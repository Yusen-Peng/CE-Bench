import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from sae_lens import SAE
from transformer_lens import HookedTransformer
from transformer_lens import utils
from functools import partial

torch.set_grad_enabled(False)
def main():    
    # next we want to do a reconstruction test
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
    get_neuronpedia_quick_list(sae, test_feature_idx_gpt)


if __name__ == "__main__":
    main()