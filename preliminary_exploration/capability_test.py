import torch
import gc
import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Tokenizer
from sae_lens import SAE, HookedSAETransformer
from transformer_lens import utils
from functools import partial


def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    architecture = "GPT_cache_gated"
    steps = "1k"
    best_model = "best_3686400_ce_2.39366_ori_2.33838"
    log_file = f"figures/{architecture}_{steps}_{best_model}_capability.log"
    sys.stdout = open(log_file, "w")

    # Load LLaMA 3.2 tokenizer
    #model_name = "meta-llama/Llama-3.2-1B"
    model_name = "gpt2-small"
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load the trained SAE from checkpoints
    sae_checkpoint_path = f"checkpoints/{architecture}/{steps}/{best_model}/"
    sae = SAE.load_from_pretrained(path=sae_checkpoint_path, device=device)
    sae.eval()

    # Load model using HookedSAETransformer with SAE's kwargs
    model = HookedSAETransformer.from_pretrained_no_processing(
        model_name,
        device=device,
        **sae.cfg.model_from_pretrained_kwargs
    )

    example_prompt = "If the glass falls off the table, it will"
    example_answer = " break"
    utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)

    logits, cache = model.run_with_cache(example_prompt, prepend_bos=False)
    tokens = model.to_tokens(example_prompt)
    sae_out = sae(cache[sae.cfg.hook_name])

    def reconstr_hook(activations, hook, sae_out):
        expected_seq_len = activations.shape[1]
        if sae_out.shape[1] < expected_seq_len:
            padding = torch.zeros(
                sae_out.shape[0], expected_seq_len - sae_out.shape[1], sae_out.shape[2],
                device=sae_out.device
            )
            sae_out_padded = torch.cat([sae_out, padding], dim=1)
            print(f"Padded sae_out from {sae_out.shape} to {sae_out_padded.shape}")
            return sae_out_padded
        return sae_out

    hook_name = sae.cfg.hook_name
    with model.hooks(
        fwd_hooks=[
            (
                hook_name,
                partial(reconstr_hook, sae_out=sae_out),
            )
        ]
    ):
        utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)




if __name__ == "__main__":
    main()
