import torch
import torch.nn as nn
import numpy as np
import openai
import os
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2Tokenizer
from sae_lens import SAE, HookedSAETransformer
from collections import defaultdict
import matplotlib.pyplot as plt
import json

def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load LLaMA tokenizer
    #model_name = "meta-llama/Llama-3.2-1B"
    model_name = "gpt2-small"
    
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded!")

    # Load the trained SAE
    experiment_name = "GPT_baseline_crop"


    #architecture = "LLAMA_cache_only_kan"
    #architecture = "LLAMA_cache_kan_relu_dense"
    #architecture = "LLAMA_cache_jumprelu"
    #architecture = "LLAMA_cache_gated"
    #architecture = "GPT_cache_jumprelu"
    #architecture = "GPT_cache_kan_relu_dense"
    #architecture = "GPT_cache_only_kan"
    architecture = "GPT_cache_gated"
    steps = "1k"

    ### LLAMA
    #best_model = "best_2457600_ce_2.13012_ori_2.03857" # only kan
    #best_model = "best_2457600_ce_2.09549_ori_2.03857" # kan-relu-dense
    #best_model = "best_2457600_ce_2.23809_ori_2.03857"  # jumprelu
    #best_model = "best_2457600_ce_2.24055_ori_2.03857" # gated

    ### GPT2
    #best_model = "best_3686400_ce_2.37705_ori_2.33838" # gpt2 jumprelu
    #best_model = "best_3686400_ce_2.34855_ori_2.33838" # gpt2 kan-relu-dense
    #best_model = "best_3686400_ce_2.35626_ori_2.33838" # gpt2 only kan
    best_model = "best_3686400_ce_2.39366_ori_2.33838" # gpt2 gated

    generate_histograms = True
    log_vectors = False

    logs_folder = f"interpretability_eval/{experiment_name}"
    os.makedirs(logs_folder, exist_ok=True)
    os.makedirs(f"{logs_folder}/histograms", exist_ok=True)
    os.makedirs(f"{logs_folder}/raw", exist_ok=True)

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
    dataset = load_dataset("GulkoA/contrastive-stories-v1", split="train")
    #dataset = 
    import re

    # # Create a CSV file to store the results
    # with open(f"{logs_folder}/interpretability_scores.csv", "w") as f:
    #     f.write("pair_index,interpretability_score,responsible_neuron,ground_truth_subject\n")

    # # raw V1 and V2
    # with open(f"{logs_folder}/raw_V1_V2.log", "w") as f:
    #     f.write(f"RAW V1 AND V2 VECTORS FOR {experiment_name}\n")


    contrastive_scores = []
    independent_scores = []
    interpretability_scores = []
    elementwise_interpretability_scores_per_subject = defaultdict(list)
    interpretability_scores_per_subject = defaultdict(list)

    neuron_interpretability_score_subject_pairs = {}

    total_rows = len(dataset)
    for pair_index in tqdm(range(total_rows)):

        # filter out marked tokens
        text_A_original = dataset[pair_index]["story1"]
        text_B_original = dataset[pair_index]["story2"]
        ground_truth_subject = dataset[pair_index]["subject"]

        if "relevance to " in ground_truth_subject:
            continue
        
        # # find all marked tokens
        # marked_tokens_A = re.findall(r"\*(.*?)\*", text_A_original)
        # marked_tokens_B = re.findall(r"\*(.*?)\*", text_B_original)


        # # remove only asterisks, not the tokens
        # text_A = text_A_original.replace("*", "")
        # text_B = text_B_original.replace("*", "")
        
        tokenizer.add_special_tokens({"additional_special_tokens": ["<subject>", "</subject>"]})
        subject_tokens = tokenizer.convert_tokens_to_ids(["<subject>", "</subject>"])
        # print(subject_tokens)
        tokens = [tokenizer(text_A_original).to(device)["input_ids"], tokenizer(text_B_original).to(device)["input_ids"]]
        clean_tokens = [[],[]]

        # find all marked tokens and record ids of all marked tokens
        marked_tokens_indices = [[], []]
        in_subject = False
        # print(tokens_A["input_ids"])
        for story_i in range(2):
            for token_index in range(len(tokens[story_i])):
                token_id = tokens[story_i][token_index]
                # string = tokenizer.decode(token_id)
                # print(f"token index: {token_id} {string}")
                if in_subject:
                    if token_id == subject_tokens[1]:
                        in_subject = False
                    else:
                        marked_tokens_indices[story_i].append(len(clean_tokens[story_i]))
                        clean_tokens[story_i].append(token_id)
                elif token_id == subject_tokens[0]:
                    in_subject = True
                else:
                    clean_tokens[story_i].append(token_id)
            
            tqdm.write(f"story {story_i} marked tokens: {len(marked_tokens_indices[story_i])}")

        # remove the subject tokens
        
        # for i, clean_token in enumerate(clean_tokens_A):
        #     if i in marked_tokens_indeces_A:
        #         print("*", end=" ")
        #     print(tokenizer.decode(clean_token), end=" ")
        
        # for marked_token_id in marked_tokens_indeces_A:
        #     print(f"marked token id: {marked_token_id} {tokenizer.decode(clean_tokens_A[marked_token_id])}")
        # quit()

        # Extract activations from the correct layer
        clean_tokens_A = torch.tensor(clean_tokens[0]).to(device)
        clean_tokens_B = torch.tensor(clean_tokens[1]).to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(clean_tokens_A) 
        hidden_states_A = cache["blocks.5.hook_mlp_out"]  # now we are poking layer 5

        with torch.no_grad():
            _, cache = model.run_with_cache(clean_tokens_B)
        hidden_states_B = cache["blocks.5.hook_mlp_out"]  # now we are poking layer 5

        with torch.no_grad():
            # activations_A = sae.encode(hidden_states_A)
            # activations_B = sae.encode(hidden_states_B)
            activations_A = hidden_states_A
            activations_B = hidden_states_B

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

        for token_index, token_id in enumerate(clean_tokens[0]):
            if token_index in marked_tokens_indices[0]:
                # add the activations of this token to V1
                # tqdm.write(f"token: {token_index} {token_id} {tokenizer.decode(token_id)}")
                V1 += activations_A[0, token_index, :]
                V1_token_num += 1
                I1 += activations_A[0, token_index, :]
                I1_token_num += 1
            else:
                I2 += activations_A[0, token_index, :]
                I2_token_num += 1
        
        for token_index, token_id in enumerate(clean_tokens[1]):
            # NOTE: a prefix space is added to match the marked tokens
            if token_index in marked_tokens_indices[1]:
                # add the activations of this token to V1
                V2 += activations_B[0, token_index, :]
                V2_token_num += 1
                I1 += activations_B[0, token_index, :]
                I1_token_num += 1
            else:
                I2 += activations_B[0, token_index, :]
                I2_token_num += 1


        # print("=" * 50)

        V1 = V1 / V1_token_num if V1_token_num > 0 else V1
        V2 = V2 / V2_token_num if V2_token_num > 0 else V2
        I1 = I1 / I1_token_num if I1_token_num > 0 else I1
        I2 = I2 / I2_token_num if I2_token_num > 0 else I2

        # print raw V1 and V2 to a log file
        # with open(f"{logs_folder}/raw_V1_V2.log", "a") as f:
        #     f.write("="*100 + "\n")
        #     f.write(f"pair_index: {pair_index}\n")
        #     # actually print out every element of V1 and V2
        #     for num in V1.tolist():
        #         f.write(f"{num:4f},")
        #     f.write("\n")
        #     for num in V2.tolist():
        #         f.write(f"{num:4f},")
        #     f.write("\n")

        if log_vectors:
            df = pd.DataFrame({"V1": V1, "V2": V2, "delta": V1 - V2, "abs_delta": np.abs(V1 - V2)})
            df.to_csv(f"{logs_folder}/raw/V1_V2_{pair_index}.csv", index=True)


        elementwise_contrast_distance = np.abs(V1 - V2)
        elementwise_contrastive_score = elementwise_contrast_distance - np.average(elementwise_contrast_distance)
        st_dev = np.std(elementwise_contrastive_score) if np.std(elementwise_contrastive_score) != 0 else 1
        elementwise_contrastive_score /= st_dev
        contrastive_score = np.max(elementwise_contrastive_score)

        elementwise_independence_distance = np.abs(I1 - I2)
        elementwise_independence_score = elementwise_independence_distance - np.average(elementwise_independence_distance)
        st_dev = np.std(elementwise_independence_score) if np.std(elementwise_independence_score) != 0 else 1
        elementwise_independence_score /= st_dev
        independence_score = np.max(elementwise_independence_score)

        elementwise_interpretability_distance = elementwise_contrast_distance + elementwise_independence_distance
        elementwise_interpretability_score = elementwise_interpretability_distance - np.average(elementwise_interpretability_distance)
        st_dev = np.std(elementwise_interpretability_distance) if np.std(elementwise_interpretability_distance) != 0 else 1
        elementwise_interpretability_score /= st_dev
        interpretability_score = np.max(elementwise_interpretability_score)


        if generate_histograms:
            # Create a single row of plots with better title structure
            plt.figure(figsize=(20, 5))  # Wider figure for one row
            
            # Set up title and subtitle
            plt.suptitle(f"Interpretability Analysis - {ground_truth_subject}", fontsize=14, y=0.98)
            plt.figtext(0.5, 0.91, 
                    f"Contrastive: {contrastive_score:.4f} | Independent: {independence_score:.4f} | Interpretability: {interpretability_score:.4f} | Story1: {text_A_original[:100]}...", 
                    ha="center", fontsize=12)
            
            # Scatter plot
            plt.subplot(1, 4, 1)
            scatter = plt.scatter(elementwise_contrastive_score, elementwise_independence_score, 
                        c=elementwise_interpretability_score, cmap='viridis')
            plt.colorbar(scatter, label="Interpretability Score")
            plt.xlabel("Contrastive Score")
            plt.ylabel("Independent Score")
            plt.title("Feature Space")
            
            # Histograms in a row
            plt.subplot(1, 4, 2)
            plt.hist(elementwise_contrastive_score, bins=50)
            plt.title("Contrastive Distribution")
            plt.xlabel("z-score")
            plt.ylabel("Frequency")
            
            plt.subplot(1, 4, 3)
            plt.hist(elementwise_independence_score, bins=50)
            plt.title("Independence Distribution")
            plt.xlabel("z-score")
            
            plt.subplot(1, 4, 4)
            plt.hist(elementwise_interpretability_score, bins=50)
            plt.title("Interpretability Distribution")
            plt.xlabel("z-score")
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)  # Make room for the titles
            
            plt.savefig(f"{logs_folder}/histograms/{pair_index}.png")
            plt.close()



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
        

        # wirte them to a file
        # with open(f"interpretability_eval/{architecture}_interpretability_scores.csv", "a") as f:
        #     f.write(f"{pair_index},{interpretability_score:4f},{responsible_neuron},{ground_truth_subject}\n")
        tqdm.write(f"pair index: {pair_index} {ground_truth_subject}:\n contrastive score: {contrastive_score:4f}\n independent score: {independence_score:4f}\n interpretability score: {interpretability_score:4f}\n")
        # print(f"interpretability score: {(contrastive_score + independence_score):4f}\n")

        # append the scores to the lists
        contrastive_scores.append(contrastive_score)
        independent_scores.append(independence_score)
        interpretability_scores.append(interpretability_score)
        elementwise_interpretability_scores_per_subject[ground_truth_subject].append(elementwise_interpretability_score)
        interpretability_scores_per_subject[ground_truth_subject].append(interpretability_score)

    # compute the average for contrastive and independent scores, and overall interpretability score
    contrastive_scores = np.array(contrastive_scores)
    independent_scores = np.array(independent_scores)
    interpretability_scores = np.array(interpretability_scores)
    contrastive_score_mean = np.mean(contrastive_scores)
    independent_score_mean = np.mean(independent_scores)
    interpretability_score_mean = np.mean(interpretability_scores)
    tqdm.write(f"Contrastive score mean: {contrastive_score_mean:4f}")
    tqdm.write(f"Independent score mean: {independent_score_mean:4f}")
    tqdm.write(f"Interpretability score mean: {interpretability_score_mean:4f}")

    interpretability_scores_per_neuron_per_subject = {}
    for subject, scores in elementwise_interpretability_scores_per_subject.items():
        all_stories = np.stack(scores, axis=0)
        interpretability_scores_per_neuron_per_subject[subject] = np.mean(all_stories, axis=0).tolist()
        # tqdm.write(f"Interpretability score mean for {subject}: {average_interpretability_scores_per_subject[subject]:4f}")
    
    average_interpretability_scores_per_subject = {}
    for subject, scores in interpretability_scores_per_subject.items():
        average_interpretability_scores_per_subject[subject] = np.mean(np.array(scores))

    # save the interpretability scores per subject to a CSV file
    df = pd.DataFrame.from_dict(interpretability_scores_per_neuron_per_subject, orient='index').T
    df.to_csv(f"{logs_folder}/interpretability_scores_per_subject.csv", index=True, header=True) # we need to keep track of the indices
    
    df = pd.DataFrame.from_dict(average_interpretability_scores_per_subject, orient='index', columns=['average_interpretability_score'])
    df.sort_values(by='average_interpretability_score', ascending=False, inplace=True)
    df.to_csv(f"{logs_folder}/average_interpretability_scores_per_subject.csv", index=True) # we need to keep track of the indices

    # responsible neurons are regrouped based on subject and written to a CSV file
    # create a dataframe from the dictionary
    df = pd.DataFrame.from_dict(neuron_interpretability_score_subject_pairs, orient='index', columns=['interpretability_score', 'subject'])
    # reorder by subject
    df = df.sort_values(by='subject')
    # save to csv
    df.to_csv(f"{logs_folder}/responsible_neurons.csv", index=True) # we need to keep track of the indices

    results = {
        "experiment_name": experiment_name,
        "architecture": architecture,
        "steps": steps,
        "best_model": best_model,
        "model_name": model_name,
    
        "contrastive_score_mean": contrastive_score_mean,
        "independent_score_mean": independent_score_mean,
        "interpretability_score_mean": interpretability_score_mean,

        "total_rows": total_rows,
    }

    with open(f"{logs_folder}/results.json", "w") as f:
        json.dump(results, f, indent=4)


    
if __name__ == "__main__":
    main()



