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


    # load the contrastive dataset from huggingface
    from datasets import load_dataset
    dataset = load_dataset("GulkoA/contrastive-stories", split="train")
    # print three columns: story1, story2, and subject


    import re

    # Create a CSV file to store the results
    with open(f"interpretability_eval/{architecture}_interpretability_scores.csv", "w") as f:
        f.write("pair_index,interpretability_score,responsible_neuron,ground_truth_subject\n")

    for pair_index in range(len(dataset)):

        # filter out marked tokens
        text_A_original = dataset[pair_index]["story1"]
        text_B_original = dataset[pair_index]["story2"]
        ground_truth_subject = dataset[pair_index]["subject"]
        
        # find all marked tokens
        marked_tokens_A = re.findall(r"\*(.*?)\*", text_A_original)
        marked_tokens_B = re.findall(r"\*(.*?)\*", text_B_original)


        # remove only asterisks, not the tokens
        text_A = text_A_original.replace("*", "")
        text_B = text_B_original.replace("*", "")
        
        tokens_A = tokenizer(text_A, return_tensors="pt").to(device)
        tokens_B = tokenizer(text_B, return_tensors="pt").to(device)

        # Extract activations from the correct layer
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens_A.input_ids) 
        hidden_states_A = cache["blocks.5.hook_mlp_out"]  # now we are poking layer 5

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens_B.input_ids)
        hidden_states_B = cache["blocks.5.hook_mlp_out"]  # now we are poking layer 5

        with torch.no_grad():
            activations_A = sae(hidden_states_A)
            activations_B = sae(hidden_states_B)

        # Convert activations to NumPy
        activations_A = activations_A.to(dtype=torch.float32).detach().cpu().numpy()
        activations_B = activations_B.to(dtype=torch.float32).detach().cpu().numpy()

        # compute V1 and V2 only for the marked tokens
        V1 = np.zeros(activations_A.shape[2])
        for token_index in range(activations_A.shape[1]):
            # traverse each token
            token_to_traverse = tokenizer.decode(tokens_A["input_ids"][0][token_index]) 

            for marked_token in marked_tokens_A:
                # NOTE: a prefix space is added to match the marked tokens
                marked_token_prepended = " " + marked_token
                if token_to_traverse == marked_token_prepended:
                    # add the activations of this token to V1
                    V1 += activations_A[0, token_index, :]
                    print(f"V1:{marked_token}")
                    break
        
        # compute V2 only for the marked tokens
        V2 = np.zeros(activations_B.shape[2])   
        for token_index in range(activations_B.shape[1]):
            # traverse each token
            token_to_traverse = tokenizer.decode(tokens_B["input_ids"][0][token_index]) 

            for marked_token in marked_tokens_B:
                # NOTE: a prefix space is added to match the marked tokens
                marked_token_prepended = " " + marked_token
                if token_to_traverse == marked_token_prepended:
                    # add the activations of this token to V2
                    V2 += activations_B[0, token_index, :]
                    print(f"V2:{marked_token}")
                    break
        print("=" * 50)


        # do joint normalization
        V_joined = np.stack([V1, V2], axis=0)
        V_joined_normalized = V_joined / (np.linalg.norm(V_joined) + 1e-12)
        V1_joint_normalized, V2_joint_normalized = V_joined_normalized[0], V_joined_normalized[1]

        elementwise_distance = np.abs(V1_joint_normalized - V2_joint_normalized)
        SCALE = 100
        interpretability_score = np.max(elementwise_distance) * SCALE
        responsible_neuron = np.argmax(elementwise_distance)

        # wirte them to a file
        with open(f"interpretability_eval/{architecture}_interpretability_scores.csv", "a") as f:
            f.write(f"{pair_index},{interpretability_score:4f},{responsible_neuron},{ground_truth_subject}\n")


if __name__ == "__main__":
    main()



