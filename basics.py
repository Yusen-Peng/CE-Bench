# Standard imports
import os
import torch
from tqdm import tqdm
import plotly.express as px
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE

import webbrowser
import http.server
import socketserver
import threading

PORT = 8000

torch.set_grad_enabled(False)

def display_vis_inline(filename: str, height: int = 850):
        """
        Displays the HTML files in Colab. Uses global `PORT` variable defined in prev cell, so that each
        vis has a unique port without having to define a port within the function.
        """
        webbrowser.open(filename)

def main():

    # For the most part I'll try to import functions and classes near where they are used
    # to make it clear where they come from.

    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")


    model = HookedTransformer.from_pretrained("gpt2-small", device=device)

    # the cfg dict is returned alongside the SAE since it may contain useful information for analysing the SAE (eg: instantiating an activation store)
    # Note that this is not the same as the SAEs config dict, rather it is whatever was in the HF repo, from which we can extract the SAE config dict
    # We also return the feature sparsities which are stored in HF for convenience.
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
        sae_id="blocks.8.hook_resid_pre",  # won't always be a hook point
        device=device,
    )

    from transformer_lens.utils import tokenize_and_concatenate

    dataset = load_dataset(
        path="NeelNanda/pile-10k",
        split="train",
        streaming=False,
    )

    token_dataset = tokenize_and_concatenate(
        dataset=dataset,  # type: ignore
        tokenizer=model.tokenizer,  # type: ignore
        streaming=True,
        max_length=sae.cfg.context_size,
        add_bos_token=sae.cfg.prepend_bos,
    )

    sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads

    with torch.no_grad():
        # activation store can give us tokens.
        batch_tokens = token_dataset[:32]["tokens"]
        _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)

        # Use the SAE
        feature_acts = sae.encode(cache[sae.cfg.hook_name])
        sae_out = sae.decode(feature_acts)

        # save some room
        del cache

        # ignore the bos token, get the number of features that activated in each token, averaged accross batch and position
        l0 = (feature_acts[:, 1:] > 0).float().sum(-1).detach()
        print("average l0", l0.mean().item())
        print("------------------------LO test ------------------------")
        px.histogram(l0.flatten().cpu().numpy()).show()

    from transformer_lens import utils
    from functools import partial


    # next we want to do a reconstruction test.
    def reconstr_hook(activation, hook, sae_out):
        return sae_out


    def zero_abl_hook(activation, hook):
        return torch.zeros_like(activation)


    print("Orig", model(batch_tokens, return_type="loss").item())
    print(
        "reconstr",
        model.run_with_hooks(
            batch_tokens,
            fwd_hooks=[
                (
                    sae.cfg.hook_name,
                    partial(reconstr_hook, sae_out=sae_out),
                )
            ],
            return_type="loss",
        ).item(),
    )
    print(
        "Zero",
        model.run_with_hooks(
            batch_tokens,
            return_type="loss",
            fwd_hooks=[(sae.cfg.hook_name, zero_abl_hook)],
        ).item(),
    )

    example_prompt = "When John and Mary went to the shops, John gave the bag to"
    example_answer = " Mary"
    utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)

    logits, cache = model.run_with_cache(example_prompt, prepend_bos=True)
    tokens = model.to_tokens(example_prompt)
    sae_out = sae(cache[sae.cfg.hook_name])


    def reconstr_hook(activations, hook, sae_out):
        return sae_out


    def zero_abl_hook(mlp_out, hook):
        return torch.zeros_like(mlp_out)


    hook_name = sae.cfg.hook_name

    print("Orig", model(tokens, return_type="loss").item())
    print(
        "reconstr",
        model.run_with_hooks(
            tokens,
            fwd_hooks=[
                (
                    hook_name,
                    partial(reconstr_hook, sae_out=sae_out),
                )
            ],
            return_type="loss",
        ).item(),
    )
    print(
        "Zero",
        model.run_with_hooks(
            tokens,
            return_type="loss",
            fwd_hooks=[(hook_name, zero_abl_hook)],
        ).item(),
    )


    with model.hooks(
        fwd_hooks=[
            (
                hook_name,
                partial(reconstr_hook, sae_out=sae_out),
            )
        ]
    ):
        utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)

    from sae_dashboard.sae_vis_data import SaeVisConfig
    from sae_dashboard.sae_vis_runner import SaeVisRunner

    test_feature_idx_gpt = list(range(30)) + [14057]

    feature_vis_config_gpt = SaeVisConfig(
        hook_point=hook_name,
        features=test_feature_idx_gpt,
        minibatch_size_features=64,
        minibatch_size_tokens=256,
        verbose=True,
        device=device,
    )

    visualization_data_gpt = SaeVisRunner(
        feature_vis_config_gpt
    ).run(
        encoder=sae,
        model=model,
        tokens=token_dataset[:10000]["tokens"], 
    )
    
    print(token_dataset[:20]["tokens"])


    from sae_dashboard.data_writing_fns import save_feature_centric_vis

    filename = f"demo_feature_dashboards.html"
    save_feature_centric_vis(sae_vis_data=visualization_data_gpt, filename=filename)

    from sae_lens.analysis.neuronpedia_integration import get_neuronpedia_quick_list

    # this function should open
    get_neuronpedia_quick_list(sae, test_feature_idx_gpt)


if __name__ == "__main__":
    main()