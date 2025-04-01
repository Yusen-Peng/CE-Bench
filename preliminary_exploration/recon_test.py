import os
import torch
import gc
from tqdm import tqdm
from datasets import load_dataset
from sae_lens import SAE, HookedSAETransformer
from transformer_lens import HookedTransformer
from transformer_lens import utils
from functools import partial
import time
from transformers import AutoTokenizer
from transformers import GPT2Tokenizer


torch.set_grad_enabled(False)
def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load LLaMA 3.2 tokenizer
    #model_name = "meta-llama/Llama-3.2-1B"
    #model_name = "gpt2"
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load the trained SAE from checkpoints
    architecture = "GPT_cache_kan_relu_dense"
    steps = "1k"
    best_model = "best_3686400_ce_2.34855_ori_2.33838"
    sae_checkpoint_path = f"checkpoints/{architecture}/{steps}/{best_model}"
    sae = SAE.load_from_pretrained(path=sae_checkpoint_path, device=device)
    sae.eval()

    # Load model using HookedSAETransformer with SAE's kwargs
    model = HookedSAETransformer.from_pretrained_no_processing(
        model_name="gpt2",
        device=device,
        **sae.cfg.model_from_pretrained_kwargs
    )

    # Verify SAE config
    print(f"Loaded SAE with d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}, hook={sae.cfg.hook_name}")


    #dataset = load_dataset("GulkoA/TinyStories-tokenized-Llama-3.2", split="validation")
    dataset = load_dataset("apollo-research/roneneldan-TinyStories-tokenizer-gpt2", split="validation")
    desired_sample_size = 1_000 # 1k validation samples
    downsampled_dataset = dataset.shuffle(seed=42).select(range(desired_sample_size))

    batch_size = 8
    results = []

    def reconstr_hook(activation, hook, sae_out): return sae_out

    def zero_abl_hook(activation, hook): return torch.zeros_like(activation)

    t1 = time.time()
    with torch.no_grad():
        for i in range(0, len(downsampled_dataset), batch_size):
            gc.collect()
            torch.cuda.empty_cache()

            # Select batch
            batch = downsampled_dataset[i : i + batch_size]
            inputs = torch.tensor(batch["input_ids"]).to(device, non_blocking=True)

            # Use autocast for mixed precision (memory efficient)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Forward pass with cache
                _, cache = model.run_with_cache(inputs, return_type=None)

                # NOTE: now we are probing the MLP output of the 5th block
                hidden_states = cache["blocks.5.hook_mlp_out"]

                # Encode/decode with SAE
                feature_acts = sae.encode(hidden_states)
                sae_out = sae.decode(feature_acts)

            # Compute original loss
            orig_loss = model(inputs, return_type="loss").item()

            # Compute reconstruction loss (insert decoded SAE output at hook)
            reconstr_loss = model.run_with_hooks(
                inputs,
                fwd_hooks=[
                    (sae.cfg.hook_name, partial(reconstr_hook, sae_out=sae_out))
                ],
                return_type="loss",
            ).item()

            # Compute zero-ablation loss (hooked activation -> zeros)
            zero_loss = model.run_with_hooks(
                inputs,
                fwd_hooks=[(sae.cfg.hook_name, zero_abl_hook)],
                return_type="loss",
            ).item()

            # Save results in the list
            results.append({
                "batch_index": i // batch_size,
                "orig_loss": orig_loss,
                "reconstr_loss": reconstr_loss,
                "zero_loss": zero_loss
            })

            # Free memory
            del inputs, cache, hidden_states, feature_acts, sae_out
            torch.cuda.empty_cache()
            gc.collect()

    t2 = time.time()
    print(f"Time taken for batch processing: {t2 - t1:.2f} seconds")

    print("Done with batch processing!")
    import pandas as pd
    df = pd.DataFrame(results)
    csv_path = f"figures/{architecture}_{steps}_batch_losses.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved losses to {csv_path}")



if __name__ == "__main__":
    main()