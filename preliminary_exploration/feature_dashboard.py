import os
import torch
from tqdm import tqdm
import pandas as pd
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from sae_lens import SAE, HookedSAETransformer
import json
from datasets import load_dataset
import safetensors.torch



def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load LLaMA 3.2 tokenizer
    model_name = "meta-llama/Llama-3.2-1B"  # Adjust to your specific LLaMA variant
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print("tokenizer loaded!")

    # Load the trained SAE from checkpoints
    architecture = "standard"
    sae_checkpoint_path = f"checkpoints/{architecture}/final_122880000"
    sae = SAE.load_from_pretrained(path=sae_checkpoint_path, device=device)
    print("SAE loaded!")

    # Load model using HookedSAETransformer with SAE's kwargs
    model = HookedSAETransformer.from_pretrained_no_processing(
        model_name,
        device=device,
        **sae.cfg.model_from_pretrained_kwargs
    )
    print("Model loaded!")


    from sae_dashboard.sae_vis_data import SaeVisConfig
    from sae_dashboard.sae_vis_runner import SaeVisRunner
    
    
    test_feature_idx = list(range(10)) + [14057]
    hook_name = sae.cfg.hook_name
    feature_vis_config = SaeVisConfig(
        hook_point=hook_name,
        features=test_feature_idx,
        minibatch_size_features=64,
        minibatch_size_tokens=256,
        verbose=True,
        device=device,
    )

    # Load and downsample the pile-10k dataset
    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    desired_sample_size = 400 # FIXME: experiment with this!
    downsampled_dataset = dataset.shuffle(seed=42).select(range(desired_sample_size))

    # Tokenization function
    def tokenize_function(examples):
        tokenized_output = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
        )
        return {
            "input_ids": tokenized_output["input_ids"],
            "attention_mask": tokenized_output["attention_mask"],
        }

    tokenized_dataset = downsampled_dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    tokens_tensor = tokenized_dataset["input_ids"].clone()
    print("Tokenized dataset loaded!")


    visualization_data = SaeVisRunner(
        feature_vis_config
    ).run(
        encoder=sae,
        model=model,
        tokens=tokens_tensor,
    )


    from sae_dashboard.data_writing_fns import save_feature_centric_vis

    filename = f"figures/{architecture}_demo_feature_dashboards.html"
    save_feature_centric_vis(sae_vis_data=visualization_data, filename=filename)







if __name__ == "__main__":
    main()