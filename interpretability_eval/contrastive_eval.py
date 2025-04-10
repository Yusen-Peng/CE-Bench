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
    architecture = "LLAMA_cache_only_kan"
    #architecture = "LLAMA_cache_kan_relu_dense"
    #architecture = "LLAMA_cache_jumprelu"
    ##architecture = "LLAMA_cache_gated"
    steps = "1k"
    best_model = "best_2457600_ce_2.13012_ori_2.03857"
    #best_model = "best_2457600_ce_2.09549_ori_2.03857" # kan-relu-dense
    #best_model = "best_2457600_ce_2.23809_ori_2.03857"  # jumprelu
    #best_model = "best_2457600_ce_2.24055_ori_2.03857" # gated
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

    neuron_interpretability_score_subject_pairs = {}

    total_rows = len(dataset)
    for pair_index in range(total_rows):

        # filter out marked tokens
        text_A_original = dataset[pair_index]["story1"]
        text_B_original = dataset[pair_index]["story2"]
        ground_truth_subject = dataset[pair_index]["subject"]
        
        # # find all marked tokens
        # marked_tokens_A = re.findall(r"\*(.*?)\*", text_A_original)
        # marked_tokens_B = re.findall(r"\*(.*?)\*", text_B_original)


        # # remove only asterisks, not the tokens
        # text_A = text_A_original.replace("*", "")
        # text_B = text_B_original.replace("*", "")
        
        tokens_A = tokenizer(text_A_original, return_tensors="pt").to(device)
        tokens_B = tokenizer(text_B_original, return_tensors="pt").to(device)

        # find all marked tokens and record ids of all marked tokens
        marked_tokens_ids_A = []
        marked_tokens_ids_B = []
        for token_index in range(tokens_A["input_ids"].shape[0]):
            string = tokenizer.decode(tokens_A["input_ids"][0][token_index])
            print(string)
        quit()

        # ignore everything below - testing

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

        # baseline
        # activations_A = np.random.rand(*activations_A.shape)
        # activations_B = np.random.rand(*activations_B.shape)

        # keep track of I1 and I2 for independent study
        I1 = np.zeros(activations_A.shape[2])
        I1_token_num = 0
        I2 = np.zeros(activations_B.shape[2])
        I2_token_num = 0
        # compute V1 and V2 only for the marked tokens
        V1 = np.zeros(activations_A.shape[2])
        V1_token_num = 0
        V2 = np.zeros(activations_B.shape[2])
        V2_token_num = 0

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
                    V1_token_num += 1
                    I1 += activations_A[0, token_index, :]
                    I1_token_num += 1
                    found = True
                    break
            if not found:
                I2 += activations_A[0, token_index, :]
                I2_token_num += 1
        
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
                    V2_token_num += 1
                    I1 += activations_B[0, token_index, :]
                    I1_token_num += 1
                    found = True
                    break
            if not found:
                I2 += activations_B[0, token_index, :]
                I2_token_num += 1
        print("=" * 50)

        V1 = V1 / V1_token_num if V1_token_num > 0 else V1
        V2 = V2 / V2_token_num if V2_token_num > 0 else V2
        I1 = I1 / I1_token_num if I1_token_num > 0 else I1
        I2 = I2 / I2_token_num if I2_token_num > 0 else I2

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


        elementwise_contrast_distance = np.abs(V1 - V2)
        contrastive_score = np.max(elementwise_contrast_distance) / np.average(elementwise_contrast_distance)
        elementwise_independence_distance = np.abs(I1 - I2)
        independence_score = np.max(elementwise_independence_distance) / np.average(elementwise_independence_distance)
        elementwise_interpretability_score = elementwise_contrast_distance + elementwise_independence_distance

        """
            Responsibility Clustering
        """
        # clustering neurons into different interpreter groups based on their highest interpretability score 
        for neuron_index in range(len(elementwise_interpretability_score)):
            # check if the neuron index is already in the dictionary
            if neuron_index not in neuron_interpretability_score_subject_pairs:
                neuron_interpretability_score_subject_pairs[neuron_index] = [elementwise_interpretability_score[neuron_index], ground_truth_subject]
            else:
                # if it is, check if the current interpretability score is higher than the previous one
                if elementwise_interpretability_score[neuron_index] > neuron_interpretability_score_subject_pairs[neuron_index][0]:
                    neuron_interpretability_score_subject_pairs[neuron_index] = [elementwise_interpretability_score[neuron_index], ground_truth_subject]
        

        # compute the interpretability score for the entire sparse autoencoder
        interpretability_score = np.max(elementwise_interpretability_score) / np.average(elementwise_interpretability_score)

        # wirte them to a file
        # with open(f"interpretability_eval/{architecture}_interpretability_scores.csv", "a") as f:
        #     f.write(f"{pair_index},{interpretability_score:4f},{responsible_neuron},{ground_truth_subject}\n")
        print(f"pair index: {pair_index} {ground_truth_subject}:\n contrastive score: {contrastive_score:4f}\n independent score: {independence_score:4f}\n interpretability score: {interpretability_score:4f}\n")
        # print(f"interpretability score: {(contrastive_score + independence_score):4f}\n")

        # append the scores to the lists
        contrastive_scores.append(contrastive_score)
        independent_scores.append(independence_score)
        interpretability_scores.append(interpretability_score)

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

    # responsible neurons are regrouped based on subject and written to a CSV file
    # create a dataframe from the dictionary
    df = pd.DataFrame.from_dict(neuron_interpretability_score_subject_pairs, orient='index', columns=['interpretability_score', 'subject'])
    # reorder by subject
    df = df.sort_values(by='subject')
    # save to csv
    df.to_csv(f"interpretability_eval/{architecture}_responsible_neurons.csv", index=True) # we need to keep track of the indices
    
if __name__ == "__main__":
    main()



