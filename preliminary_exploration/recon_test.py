import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from sae_lens import SAE, HookedSAETransformer
from transformer_lens import HookedTransformer
from transformer_lens import utils
from functools import partial
from transformers import AutoTokenizer


torch.set_grad_enabled(False)
def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load LLaMA 3.2 tokenizer
    model_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token


    # Load the trained SAE from checkpoints
    architecture = "standard"
    sae_checkpoint_path = f"checkpoints/{architecture}/final_1000448"
    sae = SAE.load_from_pretrained(path=sae_checkpoint_path, device=device)
    sae.eval()

    # Load model using HookedSAETransformer with SAE's kwargs
    model = HookedSAETransformer.from_pretrained_no_processing(
        model_name,
        device=device,
        **sae.cfg.model_from_pretrained_kwargs
    )



    batch_tokens = token_dataset[:32]["tokens"]

    
   

    

    

    
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