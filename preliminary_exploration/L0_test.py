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


def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load LLaMA 3.2 tokenizer
    #model_name = "meta-llama/Llama-3.2-1B"
    model_name = "gpt2-small"
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load the trained SAE from checkpoints
    architecture = "GPT_cache_jumprelu"
    steps = "1k"
    best_model = "best_3686400_ce_2.37705_ori_2.33838"
    sae_checkpoint_path = f"checkpoints/{architecture}/{steps}/{best_model}/"
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
    # Load and downsample the pile-10k dataset
    #dataset = load_dataset("GulkoA/TinyStories-tokenized-Llama-3.2", split="validation")
    dataset = load_dataset("apollo-research/roneneldan-TinyStories-tokenizer-gpt2", split="validation")
    desired_sample_size = 1_000 # 1k validation samples
    downsampled_dataset = dataset.shuffle(seed=42).select(range(desired_sample_size))
    batch_size = 8
    all_activations = []
    avg_l0_scores = []
    with torch.no_grad():
        for i in range(0, len(downsampled_dataset), batch_size):
            torch.cuda.empty_cache()
            gc.collect()

            # Select batch
            batch = downsampled_dataset[i : i + batch_size]
            inputs = torch.tensor(batch["input_ids"]).to(device, non_blocking=True)

            # Use autocast for mixed precision (memory efficient)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, cache = model.run_with_cache(inputs, return_type=None)
                hidden_states = cache["blocks.0.hook_mlp_out"]

                # Encode activations using the SAE
                feature_acts = sae.encode(hidden_states)
                sae_out = sae.decode(feature_acts)

            # Calculate L0 sparsity
            avg_l0 = (feature_acts > 0).float().sum(dim=-1).mean().item()
            avg_l0_scores.append(avg_l0)
            #print(f"Batch {i//batch_size}: Average L0 = {avg_l0}")

            # Move activations to CPU early to free GPU memory
            all_activations.append(feature_acts.flatten().to(torch.float32).cpu().numpy())

            # Free memory
            del inputs, cache, hidden_states, feature_acts, sae_out
            torch.cuda.empty_cache()
            gc.collect()

    # Convert to single NumPy array
    all_activations = np.concatenate(all_activations)

    # different versions
    csv_file = f"figures/{architecture}_{steps}_{best_model}_l0_scores.csv"
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)  # Ensure directory exists
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Batch", "Average_L0"])  # Write header
        for batch_idx, l0 in enumerate(avg_l0_scores):
            writer.writerow([batch_idx, l0])

    print(f"L0 scores saved to {csv_file}!")
    print("L0 test is finished!")

    # compute the mean, median, and std of L0 scores 
    mean_l0 = np.mean(avg_l0_scores)
    median_l0 = np.median(avg_l0_scores)
    std_l0 = np.std(avg_l0_scores)
    print(f"Mean L0 score: {mean_l0}")
    print(f"Median L0 score: {median_l0}")
    print(f"Std L0 score: {std_l0}")



if __name__ == "__main__":
    main()
