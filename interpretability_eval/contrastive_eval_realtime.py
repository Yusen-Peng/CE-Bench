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
        **sae.cfg.model_from_pretrained_kwargs,
    )
    print("Model loaded!")


    # load the contrastive dataset from huggingface
    from datasets import load_dataset
    dataset = load_dataset("GulkoA/contrastive-stories", split="train")
    # print three columns: story1, story2, and subject


    import re

    # Create a CSV file to store the results
    with open(f"interpretability_eval/{architecture}_interpretability_scores.csv", "w") as f:
        f.write("pair_index,interpretability_score,responsible_neuron,ground_truth_subject\n")

    # raw V1 and V2
    with open(f"interpretability_eval/{architecture}_raw_V1_V2.log", "w") as f:
        f.write(f"RAW V1 AND V2 VECTORS FOR {architecture}\n")


    # filter out marked tokens
    # text = input("Enter the text: ")
    # text = "It was a *warm* *sunny* day. Birds were chipping and everyone was hanging out at the *beach*. The air was *hot* and humid."
    text = "It was a *freezing* day. Everyone was playing *snowballs* outside and making *snowmen*. *Chill* air was blowing into our faces as we stood on top of a hill."

    # find all marked tokens
    marked_tokens = re.findall(r"\*(.*?)\*", text)

    # remove only asterisks, not the tokens
    text = text.replace("*", "")
        
    tokens = tokenizer(text, return_tensors="pt").to(device)

    # Extract activations from the correct layer
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens.input_ids) 
    hidden_states = cache["blocks.5.hook_mlp_out"]  # now we are poking layer 5

    with torch.no_grad():
        activations = sae.encode(hidden_states).cpu()
        # (B, # tokens, # features)
        print("Activations shape:", activations.shape)

     # compute V1 and V2 only for the marked tokens
    V1 = torch.zeros(activations.shape[2])
    for token_index in range(activations.shape[1]):
        # traverse each token
        token_to_traverse = tokenizer.decode(tokens["input_ids"][0][token_index]) 

        for marked_token in marked_tokens:
            # NOTE: a prefix space is added to match the marked tokens
            marked_token_prepended = " " + marked_token
            if token_to_traverse == marked_token_prepended:
                # add the activations of this token to V1
                V1 += activations[0, token_index, :]
                print(f"V1:{marked_token}")
                break
    V1 = V1 / len(marked_tokens)
    V1 = V1.numpy()

    V1 = V1 - np.min(V1)
    V1 = V1 / np.max(V1)
  
    top_10 = np.argsort(V1)[-10:]
    for i in range(10):
        print(f"Top {i+1} feature: {top_10[i]}\t{V1[top_10[i]]}")

    temperature_features = [7102, 9723, 4286, 15862, 4414, 4277, 14632, 8707, 678, 12438, 7124]
    for i in temperature_features:
        print(f" {i}\t{V1[i]}")

if __name__ == "__main__":
    main()



