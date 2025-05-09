"""Most of this is just copied over from Arthur's code and slightly simplified:
https://github.com/ArthurConmy/sae/blob/main/sae/model.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import torch.nn.functional as F
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

SPARSITY_FILENAME = "sparsity.safetensors"
SAE_WEIGHTS_FILENAME = "sae_weights.safetensors"
SAE_CFG_FILENAME = "cfg.json"

T = TypeVar("T", bound="SAE")


@dataclass
class SAEConfig:
    # architecture details
    architecture: Literal["standard", "gated", "jumprelu", "topk", "kan"]

    # forward pass details.
    d_in: int
    d_sae: int  # 'kan': this is the bottleneck size
    activation_fn_str: str
    apply_b_dec_to_input: bool
    finetuning_scaling_factor: bool

    # dataset it was trained on details.
    context_size: int
    model_name: str
    hook_name: str
    hook_layer: int
    hook_head_index: Optional[int]
    prepend_bos: bool
    dataset_path: str
    dataset_trust_remote_code: bool
    normalize_activations: str

    # misc
    dtype: str
    device: str
    sae_lens_training_version: Optional[str]
    activation_fn_kwargs: dict[str, Any] = field(default_factory=dict)
    neuronpedia_id: Optional[str] = None
    model_from_pretrained_kwargs: dict[str, Any] = field(default_factory=dict)
    seqpos_slice: tuple[int | None, ...] = (None,)
    ae_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SAEConfig":
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
            if k in cls.__dataclass_fields__  # pylint: disable=no-member
        }

        if "seqpos_slice" in config_dict:
            config_dict["seqpos_slice"] = tuple(config_dict["seqpos_slice"])

        return cls(**config_dict)


    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "dtype": self.dtype,
            "device": self.device,
            "model_name": self.model_name,
            "hook_name": self.hook_name,
            "hook_layer": self.hook_layer,
            "hook_head_index": self.hook_head_index,
            "activation_fn_str": self.activation_fn_str,  # use string for serialization
            "activation_fn_kwargs": self.activation_fn_kwargs or {},
            "apply_b_dec_to_input": self.apply_b_dec_to_input,
            "finetuning_scaling_factor": self.finetuning_scaling_factor,
            "sae_lens_training_version": self.sae_lens_training_version,
            "prepend_bos": self.prepend_bos,
            "dataset_path": self.dataset_path,
            "dataset_trust_remote_code": self.dataset_trust_remote_code,
            "context_size": self.context_size,
            "normalize_activations": self.normalize_activations,
            "neuronpedia_id": self.neuronpedia_id,
            "model_from_pretrained_kwargs": self.model_from_pretrained_kwargs,
            "seqpos_slice": self.seqpos_slice,
            "ae_kwargs": self.ae_kwargs,
        }


class SAE(HookedRootModule):
    """
    Core Sparse Autoencoder (SAE) class used for inference. For training, see `TrainingSAE`.
    """

    cfg: SAEConfig
    dtype: torch.dtype
    device: torch.device
    x_norm_coeff: torch.Tensor

    # analysis
    use_error_term: bool

    def __init__(
        self,
        cfg: SAEConfig,
        use_error_term: bool = False,
    ):
        super().__init__()

        self.cfg = cfg

        if cfg.model_from_pretrained_kwargs:
            warnings.warn(
                "\nThis SAE has non-empty model_from_pretrained_kwargs. "
                "\nFor optimal performance, load the model like so:\n"
                "model = HookedSAETransformer.from_pretrained_no_processing(..., **cfg.model_from_pretrained_kwargs)",
                category=UserWarning,
                stacklevel=1,
            )

        self.activation_fn = get_activation_fn(
            cfg.activation_fn_str, **cfg.activation_fn_kwargs or {}
        )
        self.dtype = DTYPE_MAP[cfg.dtype]
        self.device = torch.device(cfg.device)
        self.use_error_term = use_error_term

        if self.cfg.architecture == "standard" or self.cfg.architecture == "topk":
            self.initialize_weights_basic()
            self.encode = self.encode_standard
        elif self.cfg.architecture == "gated":
            self.initialize_weights_gated()
            self.encode = self.encode_gated
        elif self.cfg.architecture == "jumprelu":
            self.initialize_weights_jumprelu()
            self.encode = self.encode_jumprelu
        elif self.cfg.architecture == "kan":
            self.initialize_weights_kan()
            self.encode = self.encode_kan
        elif self.cfg.architecture == "step":
            self.initialize_weights_step()
            self.encode = self.encode_step
        else:
            raise ValueError(f"Invalid architecture: {self.cfg.architecture}")

        # handle presence / absence of scaling factor.
        if self.cfg.finetuning_scaling_factor:
            self.apply_finetuning_scaling_factor = (
                lambda x: x * self.finetuning_scaling_factor
            )
        else:
            self.apply_finetuning_scaling_factor = lambda x: x

        # set up hooks
        self.hook_sae_input = HookPoint()
        self.hook_sae_acts_pre = HookPoint()
        self.hook_sae_acts_post = HookPoint()
        self.hook_sae_output = HookPoint()
        self.hook_sae_recons = HookPoint()
        self.hook_sae_error = HookPoint()

        # handle hook_z reshaping if needed.
        # this is very cursed and should be refactored. it exists so that we can reshape out
        # the z activations for hook_z SAEs. but don't know d_head if we split up the forward pass
        # into a separate encode and decode function.
        # this will cause errors if we call decode before encode.
        if self.cfg.hook_name.endswith("_z"):
            self.turn_on_forward_pass_hook_z_reshaping()
        else:
            # need to default the reshape fns
            self.turn_off_forward_pass_hook_z_reshaping()

        # handle run time activation normalization if needed:
        if self.cfg.normalize_activations == "constant_norm_rescale":
            #  we need to scale the norm of the input and store the scaling factor
            def run_time_activation_norm_fn_in(x: torch.Tensor) -> torch.Tensor:
                self.x_norm_coeff = (self.cfg.d_in**0.5) / x.norm(dim=-1, keepdim=True)
                return x * self.x_norm_coeff

            def run_time_activation_norm_fn_out(x: torch.Tensor) -> torch.Tensor:  #
                x = x / self.x_norm_coeff
                del self.x_norm_coeff  # prevents reusing
                return x

            self.run_time_activation_norm_fn_in = run_time_activation_norm_fn_in
            self.run_time_activation_norm_fn_out = run_time_activation_norm_fn_out

        elif self.cfg.normalize_activations == "layer_norm":
            #  we need to scale the norm of the input and store the scaling factor
            def run_time_activation_ln_in(
                x: torch.Tensor, eps: float = 1e-5
            ) -> torch.Tensor:
                mu = x.mean(dim=-1, keepdim=True)
                x = x - mu
                std = x.std(dim=-1, keepdim=True)
                x = x / (std + eps)
                self.ln_mu = mu
                self.ln_std = std
                return x

            def run_time_activation_ln_out(
                x: torch.Tensor,
                eps: float = 1e-5,  # noqa: ARG001
            ) -> torch.Tensor:
                return x * self.ln_std + self.ln_mu  # type: ignore

            self.run_time_activation_norm_fn_in = run_time_activation_ln_in
            self.run_time_activation_norm_fn_out = run_time_activation_ln_out
        else:
            self.run_time_activation_norm_fn_in = lambda x: x
            self.run_time_activation_norm_fn_out = lambda x: x

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
    
    def initialize_weights_kan(self):
        print(f"sae.py:332 {self.cfg.activation_fn_kwargs=}")
        # create the KANAutoencoder
        self.kan_autoencoder = KANAutoencoder(
            input_size=self.cfg.d_in,
            hidden_size=self.cfg.activation_fn_kwargs["kan_hidden_size"],
            kan_ae_type=self.cfg.activation_fn_kwargs["kan_ae_type"],
            bottleneck_size=self.cfg.d_sae,
        ).to(self.device, self.dtype)

    def initialize_weights_step(self):
        self.step_size = nn.Parameter(
            torch.full((self.cfg.d_sae,), 0.1, dtype=self.dtype, device=self.device)
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

    def to(self, *args: Any, **kwargs: Any) -> "SAE":  # type: ignore
        device_arg = None
        dtype_arg = None

        # Check args
        for arg in args:
            if isinstance(arg, (torch.device, str)):
                device_arg = arg
            elif isinstance(arg, torch.dtype):
                dtype_arg = arg
            elif isinstance(arg, torch.Tensor):
                device_arg = arg.device
                dtype_arg = arg.dtype

        # Check kwargs
        device_arg = kwargs.get("device", device_arg)
        dtype_arg = kwargs.get("dtype", dtype_arg)

        if device_arg is not None:
            # Convert device to torch.device if it's a string
            device = (
                torch.device(device_arg) if isinstance(device_arg, str) else device_arg
            )

            # Update the cfg.device
            self.cfg.device = str(device)

            # Update the .device property
            self.device = device

        if dtype_arg is not None:
            # Update the cfg.dtype
            self.cfg.dtype = str(dtype_arg)

            # Update the .dtype property
            self.dtype = dtype_arg

        # Call the parent class's to() method to handle all cases (device, dtype, tensor)
        return super().to(*args, **kwargs)

    # Basic Forward Pass Functionality.
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        feature_acts = self.encode(x)
        sae_out = self.decode(feature_acts)

        # TEMP
        if self.use_error_term:
            with torch.no_grad():
                # Recompute everything without hooks to get true error term
                # Otherwise, the output with error term will always equal input, even for causal interventions that affect x_reconstruct
                # This is in a no_grad context to detach the error, so we can compute SAE feature gradients (eg for attribution patching). See A.3 in https://arxiv.org/pdf/2403.19647.pdf for more detail
                # NOTE: we can't just use `sae_error = input - x_reconstruct.detach()` or something simpler, since this would mean intervening on features would mean ablating features still results in perfect reconstruction.
                with _disable_hooks(self):
                    feature_acts_clean = self.encode(x)
                    x_reconstruct_clean = self.decode(feature_acts_clean)
                sae_error = self.hook_sae_error(x - x_reconstruct_clean)
            sae_out = sae_out + sae_error
        return self.hook_sae_output(sae_out)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate SAE features from inputs
        """
        feature_acts = self.encode(x)
        return feature_acts
    
    def process_sae_in(self, sae_in: torch.Tensor) -> torch.Tensor:
        sae_in = sae_in.to(self.dtype)
        sae_in = self.reshape_fn_in(sae_in)
        sae_in = self.hook_sae_input(sae_in)
        sae_in = self.run_time_activation_norm_fn_in(sae_in)
        return sae_in

    def encode_standard(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        """
        Calculate SAE features from inputs
        """
        sae_in = self.process_sae_in(x)

        # "... d_in, d_in d_sae -> ... d_sae",
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        return self.hook_sae_acts_post(self.activation_fn(hidden_pre))

    def encode_jumprelu(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        """
        Calculate SAE features from inputs
        """
        sae_in = self.process_sae_in(x)

        # "... d_in, d_in d_sae -> ... d_sae",
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        return self.hook_sae_acts_post(
            self.activation_fn(hidden_pre) * (hidden_pre > self.threshold)
        )

    def encode_gated(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        sae_in = self.process_sae_in(x)

        # Gating path
        gating_pre_activation = sae_in @ self.W_enc + self.b_gate
        active_features = (gating_pre_activation > 0).to(self.dtype)

        # Magnitude path with weight sharing
        magnitude_pre_activation = self.hook_sae_acts_pre(
            sae_in @ (self.W_enc * self.r_mag.exp()) + self.b_mag
        )
        feature_magnitudes = self.activation_fn(magnitude_pre_activation)

        return self.hook_sae_acts_post(active_features * feature_magnitudes)

    def encode_kan(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        sae_in = x.to(self.dtype)
        sae_in = self.reshape_fn_in(sae_in)
        sae_in = self.hook_sae_input(sae_in)
        sae_in = self.run_time_activation_norm_fn_in(sae_in)

        hidden_pre = self.hook_sae_acts_pre(sae_in)
        bottleneck = self.kan_autoencoder.encoder(hidden_pre)
        return self.hook_sae_acts_post(bottleneck)
    
    def encode_step(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        sae_in = self.process_sae_in(x)
        
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        return self.hook_sae_acts_post(self.activation_fn(hidden_pre))
        

    def decode(
        self, feature_acts: Float[torch.Tensor, "... d_sae"]
    ) -> Float[torch.Tensor, "... d_in"]:
        """Decodes SAE feature activation tensor into a reconstructed input activation tensor."""
        
        if self.cfg.architecture in ["standard", "topk", "gated", "jumprelu", "step"]:
            # "... d_sae, d_sae d_in -> ... d_in",
            sae_out = self.hook_sae_recons(
                self.apply_finetuning_scaling_factor(feature_acts) @ self.W_dec + self.b_dec
            )

            # handle run time activation normalization if needed
            # will fail if you call this twice without calling encode in between.
            sae_out = self.run_time_activation_norm_fn_out(sae_out)

            # handle hook z reshaping if needed.
            return self.reshape_fn_out(sae_out, self.d_head)  # type: ignore

        elif self.cfg.architecture == "kan":
            return self.decode_kan(feature_acts)
        
        else:
            raise ValueError(f"decode not defined for {self.cfg.architecture}")

    def decode_kan(
        self, feature_acts: Float[torch.Tensor, "... d_sae"]
    ) -> Float[torch.Tensor, "... d_in"]:
        dec_pre = self.hook_sae_recons(
            self.kan_autoencoder.decoder(feature_acts)
        )
        dec_out = self.run_time_activation_norm_fn_out(dec_pre)
        return self.reshape_fn_out(dec_out, self.d_head)  # type: ignore



    @torch.no_grad()
    def fold_W_dec_norm(self):
        if self.cfg.architecture == "kan":
            return
        W_dec_norms = self.W_dec.norm(dim=-1).unsqueeze(1)
        self.W_dec.data = self.W_dec.data / W_dec_norms
        self.W_enc.data = self.W_enc.data * W_dec_norms.T
        if self.cfg.architecture == "gated":
            self.r_mag.data = self.r_mag.data * W_dec_norms.squeeze()
            self.b_gate.data = self.b_gate.data * W_dec_norms.squeeze()
            self.b_mag.data = self.b_mag.data * W_dec_norms.squeeze()
        elif self.cfg.architecture == "jumprelu":
            self.threshold.data = self.threshold.data * W_dec_norms.squeeze()
            self.b_enc.data = self.b_enc.data * W_dec_norms.squeeze()
        else:
            self.b_enc.data = self.b_enc.data * W_dec_norms.squeeze()

    @torch.no_grad()
    def fold_activation_norm_scaling_factor(
        self, activation_norm_scaling_factor: float
    ):
        if self.cfg.architecture == "kan":
            return
        self.W_enc.data = self.W_enc.data * activation_norm_scaling_factor
        # previously weren't doing this.
        self.W_dec.data = self.W_dec.data / activation_norm_scaling_factor
        self.b_dec.data = self.b_dec.data / activation_norm_scaling_factor

        # once we normalize, we shouldn't need to scale activations.
        self.cfg.normalize_activations = "none"

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
        if self.cfg.architecture == "kan":
            # for KAN architecture, drop the standard W_enc, b_enc, W_dec, b_dec if they exist
            for key in ["W_enc", "b_enc", "W_dec", "b_dec", "r_mag", "b_gate", "b_mag"]:
                if key in state_dict:
                    del state_dict[key]
        

    # overwrite this in subclasses to modify the state_dict in-place after loading
    def process_state_dict_for_loading(self, state_dict: dict[str, Any]) -> None:
        if self.cfg.architecture == "kan":
            # If the old checkpoint has e.g. W_enc, b_enc, etc. remove them
            for key in ["W_enc", "b_enc", "W_dec", "b_dec", "r_mag", "b_gate", "b_mag"]:
                if key in state_dict:
                    del state_dict[key]


    @classmethod
    def load_from_pretrained(
        cls, path: str, device: str = "cpu", dtype: str | None = None
    ) -> "SAE":
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

        sae_cfg = SAEConfig.from_dict(cfg_dict)

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
    ) -> Tuple["SAE", dict[str, Any], Optional[torch.Tensor]]:
        """

        Load a pretrained SAE from the Hugging Face model hub.

        Args:
            release: The release name. This will be mapped to a huggingface repo id based on the pretrained_saes.yaml file.
            id: The id of the SAE to load. This will be mapped to a path in the huggingface repo.
            device: The device to load the SAE on.
            return_sparsity_if_present: If True, will return the log sparsity tensor if it is present in the model directory in the Hugging Face model hub.
        """

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

        sae = cls(SAEConfig.from_dict(cfg_dict))
        sae.process_state_dict_for_loading(state_dict)
        sae.load_state_dict(state_dict)

        # Check if normalization is 'expected_average_only_in'
        if cfg_dict.get("normalize_activations") == "expected_average_only_in":
            norm_scaling_factor = get_norm_scaling_factor(release, sae_id)
            if norm_scaling_factor is not None:
                sae.fold_activation_norm_scaling_factor(norm_scaling_factor)
                cfg_dict["normalize_activations"] = "none"
            else:
                warnings.warn(
                    f"norm_scaling_factor not found for {release} and {sae_id}, but normalize_activations is 'expected_average_only_in'. Skipping normalization folding."
                )

        return sae, cfg_dict, log_sparsities

    def get_name(self):
        return f"sae_{self.cfg.model_name}_{self.cfg.hook_name}_{self.cfg.d_sae}"

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SAE":
        return cls(SAEConfig.from_dict(config_dict))

    def turn_on_forward_pass_hook_z_reshaping(self):
        if not self.cfg.hook_name.endswith("_z"):
            raise ValueError("This method should only be called for hook_z SAEs.")

        def reshape_fn_in(x: torch.Tensor):
            self.d_head = x.shape[-1]  # type: ignore
            self.reshape_fn_in = lambda x: einops.rearrange(
                x, "... n_heads d_head -> ... (n_heads d_head)"
            )
            return einops.rearrange(x, "... n_heads d_head -> ... (n_heads d_head)")

        self.reshape_fn_in = reshape_fn_in

        self.reshape_fn_out = lambda x, d_head: einops.rearrange(
            x, "... (n_heads d_head) -> ... n_heads d_head", d_head=d_head
        )
        self.hook_z_reshaping_mode = True

    def turn_off_forward_pass_hook_z_reshaping(self):
        self.reshape_fn_in = lambda x: x
        self.reshape_fn_out = lambda x, d_head: x  # noqa: ARG005
        self.d_head = None
        self.hook_z_reshaping_mode = False


class TopK(nn.Module):
    def __init__(
        self, k: int, postact_fn: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()
    ):
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    # TODO: Use a fused kernel to speed up topk decoding like https://github.com/EleutherAI/sae/blob/main/sae/kernels.py
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result


class StepsActivation(nn.Module):
    def __init__(self, step_size: float = 1.0):

        super().__init__()
        self.step_size = step_size


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the step function
        x = torch.floor(x / self.step_size) * self.step_size

        # Make sure the output is the same shape as the input
        return x



class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):

        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # number of control points used for B-spline interpolation 
        self.grid_size = grid_size

        # order of the B-spline basis functions
        self.spline_order = spline_order

        # h calculates the grid step size, which determines how spaced the control points are
        h = (grid_range[1] - grid_range[0]) / grid_size

        # the grid tensor stores the control points for the B-spline basis functions
        # control points define where the B-spline basis functions are centered
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # the matrix responsible for the standard linear transformation applied to the base activation function
        self.base_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features))
        
        # the matrix responsible for the B-spline interpolation
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )

        # if True, introduces an additional trainable scaling parameter for spline weights
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        # scaling factor for adding random noise to spline weights during initialization
        self.scale_noise = scale_noise

        # scaling factor for initializing the base linear transformation weights
        self.scale_base = scale_base
        
        # scaling factor for spline weight initialization
        self.scale_spline = scale_spline

        # if True, introduces an additional trainable scaling parameter for spline weights
        self.enable_standalone_scale_spline = enable_standalone_scale_spline

        # activation function applied before the base linear transformation
        self.base_activation = base_activation()

        # a small factor that controls how much the grid adapts to input distributions
        self.grid_eps = grid_eps

        # initialize the weights properly here
        self.reset_parameters()

    def reset_parameters(self):
        
        # use Kaiming uniform initialization to set up self.base_weight
        torch.nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():

            # add random noise to the spline weights during initialization
            noise = (
                (
                    torch.rand(self.grid_size + 1,
                               self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )

            # initialize the spline weights using the curve2coeff function
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )

            # if enabled, initialize the standalone scaling parameter for spline weights
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x)
                / (grid[:, k + 1:] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()


    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.view(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(
            splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] +
                        2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + \
            (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1,
                               device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1,
                               device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(
            self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )
    
class Encoder(nn.Module):
    """
        encoder part: KANLinear -> ReLU -> Linear
        KANLinear is a custom linear layer that uses a combination of
        B-splines and a base activation function to transform the input.
        The output is then passed through a ReLU activation and a final
        linear layer to produce the bottleneck representation.
    """
    def __init__(
        self,
        input_size,
        hidden_size,
        bottleneck_size,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(Encoder, self).__init__()
        self.kan = KANLinear(
            input_size,
            hidden_size,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )
        self.relu = nn.ReLU()
        self.dense = nn.Linear(hidden_size, bottleneck_size)

    def forward(self, x):
        x = self.kan(x)
        x = self.relu(x)
        x = self.dense(x)
        return x


class Decoder(nn.Module):
    """
        decoder part: Linear -> ReLU -> KANLinear
        The input is first passed through a linear layer, then a ReLU activation,
        and finally through a KANLinear layer to produce the output.
    """
    def __init__(
        self,
        bottleneck_size,
        hidden_size,
        output_size,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(Decoder, self).__init__()
        self.dense = nn.Linear(bottleneck_size, hidden_size)
        self.relu = nn.ReLU()
        self.kan = KANLinear(
            hidden_size,
            output_size,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )

    def forward(self, x):
        x = self.dense(x)
        x = self.relu(x)
        x = self.kan(x)
        return x


class KANAutoencoder(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        bottleneck_size,
        kan_ae_type="kan_relu_dense",
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANAutoencoder, self).__init__()
        self.kan_ae_type = kan_ae_type
        if kan_ae_type == "only_kan":
            hidden_size = bottleneck_size
            self.encoder = KANLinear(
                input_size,
                hidden_size,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            self.decoder = KANLinear(
                hidden_size,
                input_size,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
        elif kan_ae_type == "kan_relu_dense":
            self.encoder = Encoder(
                input_size, hidden_size, bottleneck_size,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            self.decoder = Decoder(
                bottleneck_size, hidden_size, input_size,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
        else:
            raise ValueError(f"Unknown KAN Autoencoder type: {self.kan_ae_type}")

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_activation_fn(
    activation_fn: str, **kwargs: Any
) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation_fn == "relu":
        return torch.nn.ReLU()
    if activation_fn == "tanh-relu":

        def tanh_relu(input: torch.Tensor) -> torch.Tensor:
            input = torch.relu(input)
            return torch.tanh(input)

        return tanh_relu
    if activation_fn == "topk":
        if "k" not in kwargs:
            raise ValueError("TopK activation function requires a k value.")
        k = kwargs.get("k", 1)  # Default k to 1 if not provided
        postact_fn = kwargs.get(
            "postact_fn", nn.ReLU()
        )  # Default post-activation to ReLU if not provided

        return TopK(k, postact_fn)
    raise ValueError(f"Unknown activation function: {activation_fn}")


_blank_hook = nn.Identity()


@contextmanager
def _disable_hooks(sae: SAE):
    """
    Temporarily disable hooks for the SAE. Swaps out all the hooks with a fake modules that does nothing.
    """
    try:
        for hook_name in sae.hook_dict:
            setattr(sae, hook_name, _blank_hook)
        yield
    finally:
        for hook_name, hook in sae.hook_dict.items():
            setattr(sae, hook_name, hook)
