import torch
import torch.nn as nn
import numpy as np
import openai
import os
import pandas as pd
from dotenv import load_dotenv
from transformers import AutoTokenizer, GPT2Tokenizer
from sae_lens import SAE, HookedSAETransformer


def main():
    # Load environment variables (API Key)
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    if not openai.api_key:
        raise ValueError("OpenAI API key is missing! Make sure it's set in the .env file.")

    # Initialize OpenAI Client
    client = openai.OpenAI()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load LLaMA tokenizer
    model_name = "meta-llama/Llama-3.2-1B"
    #model_name = "gpt2-small"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded!")

    # Load the trained SAE
    architecture = "LLAMA_cache_kan_relu_dense"
    steps = "1k"
    best_model = "best_2457600_ce_2.09549_ori_2.03857"
    sae_checkpoint_path = f"checkpoints/{architecture}/{steps}/{best_model}/"
    sae = SAE.load_from_pretrained(path=sae_checkpoint_path, device=device)
    print("SAE loaded!")

    # Load the model using HookedSAETransformer
    model = HookedSAETransformer.from_pretrained_no_processing(
        model_name,
        device=device,
        **sae.cfg.model_from_pretrained_kwargs
    )
    print("Model loaded!")

    # Example input text
    example_text = "The stock market crashed during the economic crisis in 2008."
    tokens = tokenizer(example_text, return_tensors="pt").to(device)

    # Extract activations from the correct layer
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens.input_ids) 
    hidden_states = cache["blocks.0.hook_mlp_out"]

    # Pass hidden states into SAE
    with torch.no_grad():
        activations = sae(hidden_states)

    # Convert activations to NumPy
    activations = activations.to(dtype=torch.float32).detach().cpu().numpy()

    # Select top 100 activated features
    num_features = activations.shape[2]
    print(f"Number of features: {num_features}")

    feature_activations_sum = activations[0, :, :].sum(axis=0)  # Shape: (num_features,)

    # randomly select 100 features
    num_selected = min(100, num_features)
    # random select
    selected_feature_indices = np.random.choice(num_features, num_selected, replace=False)
    num_selected = min(100, num_features)
    selected_feature_indices = np.argsort(feature_activations_sum)[-num_selected:]


    print(f"Selected {num_selected} features for evaluation.")
    print(f"The first 5 features are: {selected_feature_indices[:10]}")
    print(f"corresponding activations: {feature_activations_sum[selected_feature_indices[:10]]}")


    