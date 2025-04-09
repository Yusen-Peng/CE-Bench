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
    #architecture = "LLAMA_cache_kan_relu_dense"
    architecture = "LLAMA_cache_jumprelu"
    steps = "1k"
    #best_model = "best_2457600_ce_2.09549_ori_2.03857" # kan-relu-dense
    best_model = "best_2457600_ce_2.23809_ori_2.03857"  # jumprelu
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
    #dataset = 
    import re

    # Create a CSV file to store the results
    with open(f"interpretability_eval/{architecture}_interpretability_scores.csv", "w") as f:
        f.write("pair_index,interpretability_score,responsible_neuron,ground_truth_subject\n")

    # raw V1 and V2
    with open(f"interpretability_eval/{architecture}_raw_V1_V2.log", "w") as f:
        f.write(f"RAW V1 AND V2 VECTORS FOR {architecture}\n")


    contrastive_scores = []
    independent_scores = []
    interpretability_scores = []

    total_rows = len(dataset)
    for pair_index in range(total_rows):

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
            activations_A = sae.encode(hidden_states_A)
            activations_B = sae.encode(hidden_states_B)

        # Convert activations to NumPy
        activations_A = activations_A.to(dtype=torch.float32).detach().cpu().numpy()
        activations_B = activations_B.to(dtype=torch.float32).detach().cpu().numpy()


        # keep track of I1 and I2 for independent study
        I1 = np.zeros(activations_A.shape[2])
        I2 = np.zeros(activations_B.shape[2])

        # compute V1 and V2 only for the marked tokens
        V1 = np.zeros(activations_A.shape[2])
        for token_index in range(activations_A.shape[1]):
            # traverse each token
            token_to_traverse = tokenizer.decode(tokens_A["input_ids"][0][token_index]) 

            found = False
            for marked_token in marked_tokens_A:
                # NOTE: a prefix space is added to match the marked tokens
                marked_token_prepended = " " + marked_token
                if token_to_traverse == marked_token_prepended:
                    # add the activations of this token to V1
                    V1 += activations_A[0, token_index, :]
                    I1 += activations_A[0, token_index, :]
                    found = True
                    break
            if not found:
                I2 += activations_A[0, token_index, :]
        
        # compute V2 only for the marked tokens
        V2 = np.zeros(activations_B.shape[2])   
        for token_index in range(activations_B.shape[1]):
            # traverse each token
            token_to_traverse = tokenizer.decode(tokens_B["input_ids"][0][token_index]) 
            found = False
            for marked_token in marked_tokens_B:
                # NOTE: a prefix space is added to match the marked tokens
                marked_token_prepended = " " + marked_token
                if token_to_traverse == marked_token_prepended:
                    # add the activations of this token to V2
                    V2 += activations_B[0, token_index, :]
                    I1 += activations_B[0, token_index, :]
                    found = True
                    #print(f"V2:{marked_token}")
                    break
            if not found:
                I2 += activations_B[0, token_index, :]
        print("=" * 50)

        # print raw V1 and V2 to a log file
        with open(f"interpretability_eval/{architecture}_raw_V1_V2.log", "a") as f:
            f.write("="*100 + "\n")
            f.write(f"pair_index: {pair_index}\n")
            # actually print out every element of V1 and V2
            for num in V1.tolist():
                f.write(f"{num:4f},")
            f.write("\n")
            for num in V2.tolist():
                f.write(f"{num:4f},")
            f.write("\n")
        df = pd.DataFrame({"V1": V1, "V2": V2, "delta": V1 - V2, "abs_delta": np.abs(V1 - V2)})
        df.to_csv(f"interpretability_eval/{architecture}_raw_V1_V2_{pair_index}.csv", index=True)


        # take average as the last stage of condensing V1 and V2
        V1 = V1 / len(marked_tokens_A)
        V2 = V2 / len(marked_tokens_B)

        # take the average of I1 and I2
        I1 = I1 / (len(marked_tokens_A) + len(marked_tokens_B))
        num_tokens_A = len(tokens_A["input_ids"][0])
        num_tokens_B = len(tokens_B["input_ids"][0])
        I2 = I2 / (num_tokens_A + num_tokens_B - len(marked_tokens_A) - len(marked_tokens_B))


        # do joint normalization
        V_joined = np.stack([V1, V2], axis=0)
        V_joined = V_joined - np.min(V_joined)
        V_joined_normalized = V_joined / np.max(V_joined)
        V1_joint_normalized, V2_joint_normalized = V_joined_normalized[0], V_joined_normalized[1]
        elementwise_distance = np.abs(V1_joint_normalized - V2_joint_normalized)
        contrastive_score = np.max(elementwise_distance)

        # do the same thing for I1 and I2
        I_joined = np.stack([I1, I2], axis=0)
        I_joined = I_joined - np.min(I_joined)
        I_joined_normalized = I_joined / np.max(I_joined)
        I1_joint_normalized, I2_joint_normalized = I_joined_normalized[0], I_joined_normalized[1]
        elementwise_distance = np.abs(I1_joint_normalized - I2_joint_normalized)
        independent_score = np.max(elementwise_distance)


        # wirte them to a file
        # with open(f"interpretability_eval/{architecture}_interpretability_scores.csv", "a") as f:
        #     f.write(f"{pair_index},{interpretability_score:4f},{responsible_neuron},{ground_truth_subject}\n")
        print(f"pair index: {pair_index}:\n contrastive score: {contrastive_score:4f}\n independent score: {independent_score:4f}\n")
        print(f"interpretability score: {(contrastive_score + independent_score):4f}\n")

        # append the scores to the lists
        contrastive_scores.append(contrastive_score)
        independent_scores.append(independent_score)
        interpretability_scores.append(contrastive_score + independent_score)

    # compute the average for contrastive and independent scores, and overall interpretability score
    contrastive_scores = np.array(contrastive_scores)
    independent_scores = np.array(independent_scores)
    interpretability_scores = np.array(interpretability_scores)
    contrastive_score_mean = np.mean(contrastive_scores)
    independent_score_mean = np.mean(independent_scores)
    interpretability_score_mean = np.mean(interpretability_scores)
    print(f"Contrastive score mean: {contrastive_score_mean:4f}")
    print(f"Independent score mean: {independent_score_mean:4f}")
    print(f"Interpretability score mean: {interpretability_score_mean:4f}")
    
if __name__ == "__main__":
    main()



