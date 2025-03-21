import json
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Tuple, TypeVar, Union, overload

import einops
import torch
from jaxtyping import Float
from safetensors.torch import save_file
from torch import nn
from transformer_lens.hook_points import HookedRootModule, HookPoint

from sae_lens.config import DTYPE_MAP
from sae_lens.toolkit.pretrained_sae_loaders import (
    NAMED_PRETRAINED_SAE_LOADERS,
    get_conversion_loader_name,
    handle_config_defaulting,
    read_sae_from_disk,
)
from sae_lens.toolkit.pretrained_saes_directory import (
    get_norm_scaling_factor,
    get_pretrained_saes_directory,
)


from kan_ae import KANAutoencoder

SPARSITY_FILENAME = "sparsity.safetensors"
SAE_WEIGHTS_FILENAME = "sae_weights.safetensors"
SAE_CFG_FILENAME = "cfg.json"


T = TypeVar("T", bound="KANSAE")


@dataclass
class KANSAEConfig:
    d_in: int
    d_hidden: int
    d_bottleneck: int

    # KAN hyperparameters
    grid_size: int = 5
    spline_order: int = 3
    scale_noise: float = 0.1
    scale_base: float = 1.0
    scale_spline: float = 1.0

    # misc
    model_name: str
    hook_name: str
    hook_layer: int
    prepend_bos: bool
    dataset_path: str
    dtype: str
    device: str

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "KANSAEConfig":
        # rename dict:
        rename_dict = {  # old : new
            "hook_point": "hook_name",
            "hook_point_head_index": "hook_head_index",
            "hook_point_layer": "hook_layer",
            "activation_fn": "activation_fn_str",
        }
        config_dict = {rename_dict.get(k, k): v for k, v in config_dict.items()}

        # use only config terms that are in the dataclass
        config_dict = {
            k: v
            for k, v in config_dict.items()
            if k in cls.__dataclass_fields__
        }

        if "seqpos_slice" in config_dict:
            config_dict["seqpos_slice"] = tuple(config_dict["seqpos_slice"])

        return cls(**config_dict)


    def to_dict(self) -> dict[str, Any]:
        return {
            "d_in": self.d_in,
            "dtype": self.dtype,
            "device": self.device,
            "model_name": self.model_name,
            "hook_name": self.hook_name,
            "hook_layer": self.hook_layer,
            "prepend_bos": self.prepend_bos,
            "dataset_path": self.dataset_path,
        }


class KANSAE(HookedRootModule):
    cfg: KANSAEConfig
    dtype: torch.dtype
    device: torch.device
    x_norm_coeff: torch.Tensor

    # analysis
    use_error_term: bool

    def __init__(
        self,
        cfg: KANSAEConfig,
        use_error_term: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.dtype = DTYPE_MAP[cfg.dtype]
        self.device = torch.device(cfg.device)
        self.use_error_term = use_error_term
        
        self.autoencoder = KANAutoencoder(
            input_size=cfg.d_in,
            hidden_size=cfg.d_hidden,
            bottleneck_size=cfg.d_bottleneck
        )

        self.hook_input = HookPoint()
        self.hook_encoded = HookPoint()
        self.hook_output = HookPoint()

        self.setup()  # Required for `HookedRootModule`s

    def initialize_weights_basic(self):
        # no config changes encoder bias init for now.
        self.b_enc = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

        # Start with the default init strategy:
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device
                )
            )
        )

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.d_in, self.cfg.d_sae, dtype=self.dtype, device=self.device
                )
            )
        )

        # methdods which change b_dec as a function of the dataset are implemented after init.
        self.b_dec = nn.Parameter(
            torch.zeros(self.cfg.d_in, dtype=self.dtype, device=self.device)
        )

        # scaling factor for fine-tuning (not to be used in initial training)
        # TODO: Make this optional and not included with all SAEs by default (but maintain backwards compatibility)
        if self.cfg.finetuning_scaling_factor:
            self.finetuning_scaling_factor = nn.Parameter(
                torch.ones(self.cfg.d_sae, dtype=self.dtype, device=self.device)
            )

    def initialize_weights_gated(self):
        # Initialize the weights and biases for the gated encoder
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.d_in, self.cfg.d_sae, dtype=self.dtype, device=self.device
                )
            )
        )

        self.b_gate = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

        self.r_mag = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

        self.b_mag = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device
                )
            )
        )

        self.b_dec = nn.Parameter(
            torch.zeros(self.cfg.d_in, dtype=self.dtype, device=self.device)
        )

    def initialize_weights_jumprelu(self):
        # The params are identical to the standard SAE
        # except we use a threshold parameter too
        self.threshold = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )
        self.initialize_weights_basic()

    @overload
    def to(
        self: T,
        device: Optional[Union[torch.device, str]] = ...,
        dtype: Optional[torch.dtype] = ...,
        non_blocking: bool = ...,
    ) -> T: ...

    @overload
    def to(self: T, dtype: torch.dtype, non_blocking: bool = ...) -> T: ...

    @overload
    def to(self: T, tensor: torch.Tensor, non_blocking: bool = ...) -> T: ...

    def to(self, *args, **kwargs):
        """
        Override .to() so that if we change device or dtype, we also update self.cfg accordingly.
        """
        device_arg = kwargs.get("device", None)
        dtype_arg = kwargs.get("dtype", None)
        if device_arg is not None:
            self.cfg.device = str(device_arg)
        if dtype_arg is not None:
            self.cfg.dtype = str(dtype_arg)

        return super().to(*args, **kwargs)

    # Basic Forward Pass Functionality.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with hook points for hooking the input, the encoded vector, and the final output.
        """
        # 1) Hook the input
        x = self.hook_input(x)

        # 2) Encode
        encoded = self.autoencoder.encoder(x)
        encoded = self.hook_encoded(encoded)

        # 3) Decode
        decoded = self.autoencoder.decoder(encoded)

        # 4) Hook final output
        decoded = self.hook_output(decoded)
        return decoded
   

    @overload
    def save_model(self, path: str | Path) -> Tuple[Path, Path]: ...

    @overload
    def save_model(
        self, path: str | Path, sparsity: torch.Tensor
    ) -> Tuple[Path, Path, Path]: ...

    def save_model(self, path: str | Path, sparsity: Optional[torch.Tensor] = None):
        path = Path(path)

        if not path.exists():
            path.mkdir(parents=True)

        # generate the weights
        state_dict = self.state_dict()
        self.process_state_dict_for_saving(state_dict)
        model_weights_path = path / SAE_WEIGHTS_FILENAME
        save_file(state_dict, model_weights_path)

        # save the config
        config = self.cfg.to_dict()

        cfg_path = path / SAE_CFG_FILENAME
        with open(cfg_path, "w") as f:
            json.dump(config, f)

        if sparsity is not None:
            sparsity_in_dict = {"sparsity": sparsity}
            sparsity_path = path / SPARSITY_FILENAME
            save_file(sparsity_in_dict, sparsity_path)
            return model_weights_path, cfg_path, sparsity_path

        return model_weights_path, cfg_path

    # overwrite this in subclasses to modify the state_dict in-place before saving
    def process_state_dict_for_saving(self, state_dict: dict[str, Any]) -> None:
        pass

    # overwrite this in subclasses to modify the state_dict in-place after loading
    def process_state_dict_for_loading(self, state_dict: dict[str, Any]) -> None:
        pass

    @classmethod
    def load_from_pretrained(
        cls, path: str, device: str = "cpu", dtype: str | None = None
    ) -> "KANSAE":
        # get the config
        config_path = os.path.join(path, SAE_CFG_FILENAME)
        with open(config_path) as f:
            cfg_dict = json.load(f)
        cfg_dict = handle_config_defaulting(cfg_dict)
        cfg_dict["device"] = device
        if dtype is not None:
            cfg_dict["dtype"] = dtype

        weight_path = os.path.join(path, SAE_WEIGHTS_FILENAME)
        cfg_dict, state_dict = read_sae_from_disk(
            cfg_dict=cfg_dict,
            weight_path=weight_path,
            device=device,
        )
        sae_cfg = KANSAEConfig.from_dict(cfg_dict)
        sae = cls(sae_cfg)
        sae.process_state_dict_for_loading(state_dict)
        sae.load_state_dict(state_dict)

        return sae

    @classmethod
    def from_pretrained(
        cls,
        release: str,
        sae_id: str,
        device: str = "cpu",
    ) -> Tuple["KANSAE", dict[str, Any], Optional[torch.Tensor]]:
    
        # get sae directory
        sae_directory = get_pretrained_saes_directory()

        # get the repo id and path to the SAE
        if release not in sae_directory:
            if "/" not in release:
                raise ValueError(
                    f"Release {release} not found in pretrained SAEs directory, and is not a valid huggingface repo."
                )
        elif sae_id not in sae_directory[release].saes_map:
            # If using Gemma Scope and not the canonical release, give a hint to use it
            if (
                "gemma-scope" in release
                and "canonical" not in release
                and f"{release}-canonical" in sae_directory
            ):
                canonical_ids = list(
                    sae_directory[release + "-canonical"].saes_map.keys()
                )
                # Shorten the lengthy string of valid IDs
                if len(canonical_ids) > 5:
                    str_canonical_ids = str(canonical_ids[:5])[:-1] + ", ...]"
                else:
                    str_canonical_ids = str(canonical_ids)
                value_suffix = f" If you don't want to specify an L0 value, consider using release {release}-canonical which has valid IDs {str_canonical_ids}"
            else:
                value_suffix = ""

            valid_ids = list(sae_directory[release].saes_map.keys())
            # Shorten the lengthy string of valid IDs
            if len(valid_ids) > 5:
                str_valid_ids = str(valid_ids[:5])[:-1] + ", ...]"
            else:
                str_valid_ids = str(valid_ids)

            raise ValueError(
                f"ID {sae_id} not found in release {release}. Valid IDs are {str_valid_ids}."
                + value_suffix
            )
        sae_info = sae_directory.get(release, None)
        config_overrides = sae_info.config_overrides if sae_info is not None else None

        conversion_loader_name = get_conversion_loader_name(sae_info)
        conversion_loader = NAMED_PRETRAINED_SAE_LOADERS[conversion_loader_name]

        cfg_dict, state_dict, log_sparsities = conversion_loader(
            release,
            sae_id=sae_id,
            device=device,
            force_download=False,
            cfg_overrides=config_overrides,
        )

        sae = cls(KANSAEConfig.from_dict(cfg_dict))
        sae.process_state_dict_for_loading(state_dict)
        sae.load_state_dict(state_dict)
        return sae, cfg_dict, log_sparsities

    def get_name(self):
        return f"sae_{self.cfg.model_name}_{self.cfg.hook_name}_{self.cfg.d_sae}"

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "KANSAE":
        return cls(KANSAEConfig.from_dict(config_dict))
