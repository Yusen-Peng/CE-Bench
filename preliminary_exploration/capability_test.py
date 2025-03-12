import torch
import gc
import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
from sae_lens import SAE, HookedSAETransformer
from transformer_lens import utils
from functools import partial


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

    # Verify SAE config
    print(f"Loaded SAE with d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}, hook={sae.cfg.hook_name}")


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




    # # Load and downsample the pile-10k dataset
    # dataset = load_dataset("NeelNanda/pile-10k", split="train")
    # desired_sample_size = 400 # FIXME: experiment with this!
    # downsampled_dataset = dataset.shuffle(seed=42).select(range(desired_sample_size))

    # # Tokenization function
    # def tokenize_function(examples):
    #     return tokenizer(
    #         examples["text"],
    #         padding="max_length",
    #         truncation=True,
    #         max_length=128,
    #         return_tensors="pt",
    #     )

    # tokenized_dataset = downsampled_dataset.map(tokenize_function, batched=True)

    # batch_size = 8
    # all_activations = []
    # avg_l0_scores = []
    # with torch.no_grad():
    #     for i in range(0, len(tokenized_dataset), batch_size):
    #         torch.cuda.empty_cache()
    #         gc.collect()

    #         # Select batch
    #         batch = tokenized_dataset[i : i + batch_size]
    #         inputs = torch.tensor(batch["input_ids"]).to(device, non_blocking=True)

    #         # Use autocast for mixed precision (memory efficient)
    #         with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    #             _, cache = model.run_with_cache(inputs, return_type=None)
    #             hidden_states = cache["blocks.0.hook_mlp_out"]

    #             # Encode activations using the SAE
    #             feature_acts = sae.encode(hidden_states)
    #             sae_out = sae.decode(feature_acts)

    #         # Calculate L0 sparsity
    #         avg_l0 = (feature_acts > 0).float().sum(dim=-1).mean().item()
    #         avg_l0_scores.append(avg_l0)
    #         print(f"Batch {i//batch_size}: Average L0 = {avg_l0}")

    #         # Move activations to CPU early to free GPU memory
    #         all_activations.append(feature_acts.flatten().to(torch.float32).cpu().numpy())

    #         # Free memory
    #         del inputs, cache, hidden_states, feature_acts, sae_out
    #         torch.cuda.empty_cache()
    #         gc.collect()

    # # Convert to single NumPy array
    # all_activations = np.concatenate(all_activations)

    # # different versions
    # csv_file = f"figures/{architecture}_l0_scores.csv"
    # os.makedirs(os.path.dirname(csv_file), exist_ok=True)  # Ensure directory exists
    # with open(csv_file, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Batch", "Average_L0"])  # Write header
    #     for batch_idx, l0 in enumerate(avg_l0_scores):
    #         writer.writerow([batch_idx, l0])

    # print(f"L0 scores saved to {csv_file}!")
    # print("L0 test is finished!")


if __name__ == "__main__":
    main()
