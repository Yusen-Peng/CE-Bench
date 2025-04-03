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
    architecture = "LLAMA_cache_only_kan"
    steps = "1k"
    best_model = "best_2457600_ce_2.13012_ori_2.03857"
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

    # contrastive text A and B
    text_A = "The stock market crashed during the economic crisis in 2008, leading to a global recession."
    text_B = "The stock market soared after the technology boom, resulting in record growth."

    tokens_A = tokenizer(text_A, return_tensors="pt").to(device)
    tokens_B = tokenizer(text_B, return_tensors="pt").to(device) 

    # Extract activations from the correct layer
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens_A.input_ids) 
    hidden_states_A = cache["blocks.5.hook_mlp_out"]  # now we are poking layer 5

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens_B.input_ids)
    hidden_states_B = cache["blocks.5.hook_mlp_out"]  # now we are poking layer 5
    
    # Pass hidden states into SAE
    with torch.no_grad():
        activations_A = sae(hidden_states_A)
        activations_B = sae(hidden_states_B)

    # Convert activations to NumPy
    activations_A = activations_A.to(dtype=torch.float32).detach().cpu().numpy()
    activations_B = activations_B.to(dtype=torch.float32).detach().cpu().numpy()

    print(activations_A.shape)
    V1 = activations_A.mean(axis=1).squeeze(0)
    V2 = activations_B.mean(axis=1).squeeze(0)

    V1 = V1 / (np.linalg.norm(V1) + 1e-12)
    V2 = V2 / (np.linalg.norm(V2) + 1e-12)


    elementwise_distance = np.abs(V1 - V2)
    SCALE = 100
    interpretability_score = np.max(elementwise_distance) * SCALE
    print("="*50)
    print(f"Contrastive interpretability score for {architecture} is {interpretability_score}")



if __name__ == "__main__":
    main()



