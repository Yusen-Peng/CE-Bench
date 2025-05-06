import argparse
import os
import time
import torch
from openai import OpenAI
from sae_lens import SAE
from tabulate import tabulate
from torch import Tensor
from tqdm import tqdm
from transformer_lens import HookedTransformer
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from sae_lens import SAE, HookedSAETransformer
from collections import defaultdict
import matplotlib.pyplot as plt
import json
from typing import Any
import sae_bench.sae_bench_utils.activation_collection as activation_collection
import sae_bench.sae_bench_utils.general_utils as general_utils
from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
from sae_bench.sae_bench_utils.sae_selection_utils import (
    get_saes_from_regex,
)
from stw import Stopwatch
from datasets import load_dataset, Dataset
import multiprocessing as mp
from multiprocessing import Pool
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory


def run_eval_once(
    dataset: Dataset,
    device: str,
    sae_release: str,
    sae_id: str,
    config: AutoInterpEvalConfig,
):
    
    model: HookedTransformer = HookedTransformer.from_pretrained_no_processing(
        config.model_name, device=device, dtype=config.llm_dtype
    )
     
    print(f"Running evaluation for {sae_release} {sae_id}")

    sae_id, sae, _ = load_sae(
        sae_release, sae_id, device, config.llm_dtype
    )
    
    generate_histograms = False
    log_vectors = False

    logs_folder = f"interpretability_eval/{sae_release}/{sae_id}"
    os.makedirs(logs_folder, exist_ok=True)
    if generate_histograms:
        os.makedirs(f"{logs_folder}/histograms", exist_ok=True)
    if log_vectors:
        os.makedirs(f"{logs_folder}/raw", exist_ok=True)


    sw = Stopwatch(verbose=True, start=True)

    contrastive_scores = []
    independent_scores = []
    interpretability_scores = []
    elementwise_interpretability_scores_per_subject = defaultdict(list)
    interpretability_scores_per_subject = defaultdict(list)
    elementwise_contrastive_scores_per_subject = defaultdict(list)
    elementwise_independence_scores_per_subject = defaultdict(list)

    shift_v_per_subect = defaultdict(list)

    neuron_interpretability_score_subject_pairs = {}

    total_rows = len(dataset)
    # total_rows = 10
    for pair_index in tqdm(range(total_rows)):

        # filter out marked tokens
        text_A_original = dataset[pair_index]["story1"]
        text_B_original = dataset[pair_index]["story2"]
        ground_truth_subject = dataset[pair_index]["subject"]

        if "relevance to " in ground_truth_subject:
            continue

        tokenizer = model.tokenizer
    
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
            

        # Extract activations from the correct layer
        clean_tokens_A = torch.tensor(clean_tokens[0]).to(device)
        clean_tokens_B = torch.tensor(clean_tokens[1]).to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(clean_tokens_A) 
        hidden_states_A = cache[sae.cfg.hook_name]

        with torch.no_grad():
            _, cache = model.run_with_cache(clean_tokens_B)
        hidden_states_B = cache[sae.cfg.hook_name]

        with torch.no_grad():
            activations_A = sae.encode(hidden_states_A).cpu()
            activations_B = sae.encode(hidden_states_B).cpu()

        # keep track of I1 and I2 for independent study
        I1 = torch.zeros(activations_A.shape[2])
        I1_token_num = 0
        I2 = torch.zeros(activations_B.shape[2])
        I2_token_num = 0    
        # compute V1 and V2 only for the marked tokens
        V1 = torch.zeros(activations_A.shape[2])
        V1_token_num = 0
        V2 = torch.zeros(activations_B.shape[2])
        V2_token_num = 0

        for token_index, token_id in enumerate(clean_tokens[0]):
            if token_index in marked_tokens_indices[0]:
                # add the activations of this token to V1
                V1 += activations_A[0, token_index, :]
                V1_token_num += 1
                I1 += activations_A[0, token_index, :]
                I1_token_num += 1
            else:
                # FIXME: if average over everything
                V1 += activations_A[0, token_index, :] 
                V1_token_num += 1
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
                # FIXME: if average over everything
                V2 += activations_B[0, token_index, :] 
                V2_token_num += 1
                I2 += activations_B[0, token_index, :]
                I2_token_num += 1

        V1 = V1 / V1_token_num if V1_token_num > 0 else V1
        V2 = V2 / V2_token_num if V2_token_num > 0 else V2
        I1 = I1 / I1_token_num if I1_token_num > 0 else I1
        I2 = I2 / I2_token_num if I2_token_num > 0 else I2


        if log_vectors:
            df = pd.DataFrame({"V1": V1, "V2": V2, "delta": V1 - V2, "abs_delta": np.abs(V1 - V2)})
            df.to_csv(f"{logs_folder}/raw/V1_V2_{pair_index}.csv", index=True)

        shift_v = V2 - V1
        shift_v_per_subect[ground_truth_subject].append(shift_v)

        elementwise_contrast_distance = torch.abs(V1 - V2)
        elementwise_contrastive_score = elementwise_contrast_distance - torch.mean(elementwise_contrast_distance)
        st_dev = torch.std(elementwise_contrastive_score) if torch.std(elementwise_contrastive_score) != 0 else 1
        elementwise_contrastive_score /= st_dev
        contrastive_score = torch.max(elementwise_contrastive_score).item()

        elementwise_independence_distance = torch.abs(I1 - I2)
        elementwise_independence_score = elementwise_independence_distance - torch.mean(elementwise_independence_distance)
        st_dev = torch.std(elementwise_independence_score) if torch.std(elementwise_independence_score) != 0 else 1
        elementwise_independence_score /= st_dev
        independence_score = torch.max(elementwise_independence_score).item()

        elementwise_interpretability_distance = elementwise_contrast_distance + elementwise_independence_distance
        elementwise_interpretability_score = elementwise_interpretability_distance - torch.mean(elementwise_interpretability_distance)
        st_dev = torch.std(elementwise_interpretability_distance) if torch.std(elementwise_interpretability_distance) != 0 else 1
        elementwise_interpretability_score /= st_dev
        interpretability_score = torch.max(elementwise_interpretability_score).item()

        # tqdm.write(f"{pair_index}-{ground_truth_subject}-{elementwise_interpretability_score[:5]}")


        elementwise_interpretability_score_np = elementwise_interpretability_score.numpy()
        elementwise_contrastive_score_np = elementwise_contrastive_score.numpy()
        elementwise_independence_score_np = elementwise_independence_score.numpy()
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
            scatter = plt.scatter(elementwise_contrastive_score_np, elementwise_independence_score_np, 
                        c=elementwise_interpretability_score_np, cmap='viridis')
            plt.colorbar(scatter, label="Interpretability Score")
            plt.xlabel("Contrastive Score")
            plt.ylabel("Independent Score")
            plt.title("Feature Space")
            
            # Histograms in a row
            plt.subplot(1, 4, 2)
            plt.hist(elementwise_contrastive_score_np, bins=50)
            plt.title("Contrastive Distribution")
            plt.xlabel("z-score")
            plt.ylabel("Frequency")
            
            plt.subplot(1, 4, 3)
            plt.hist(elementwise_independence_score_np, bins=50)
            plt.title("Independence Distribution")
            plt.xlabel("z-score")
            
            plt.subplot(1, 4, 4)
            plt.hist(elementwise_interpretability_score_np, bins=50)
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
        # for neuron_index in range(len(elementwise_interpretability_score_np)):
        #     # check if the neuron index is already in the dictionary
        #     if neuron_index not in neuron_interpretability_score_subject_pairs:
        #         neuron_interpretability_score_subject_pairs[neuron_index] = [elementwise_interpretability_score_np[neuron_index], ground_truth_subject]
        #     else:
        #         # if it is, check if the current interpretability score is higher than the previous one
        #         if elementwise_interpretability_score[neuron_index] > neuron_interpretability_score_subject_pairs[neuron_index][0]:
        #             neuron_interpretability_score_subject_pairs[neuron_index] = [elementwise_interpretability_score[neuron_index], ground_truth_subject]
        


        # append the scores to the lists
        contrastive_scores.append(contrastive_score)
        independent_scores.append(independence_score)
        interpretability_scores.append(interpretability_score)
        elementwise_interpretability_scores_per_subject[ground_truth_subject].append(elementwise_interpretability_score)
        interpretability_scores_per_subject[ground_truth_subject].append(interpretability_score)
        elementwise_contrastive_scores_per_subject[ground_truth_subject].append(elementwise_contrastive_score_np)
        elementwise_independence_scores_per_subject[ground_truth_subject].append(elementwise_independence_score_np)

    sw.lap("Finished all stories")

    # delete the model and sae to free up memory
    sae_cfg = sae.cfg.to_dict()
    del model
    del sae
    torch.cuda.empty_cache()

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

    shift_v_per_subject_mean = {}
    for subject, shifts in shift_v_per_subect.items():
        all_shifts = np.stack(shifts, axis=0)
        shift_v_per_subject_mean[subject] = np.mean(all_shifts, axis=0).tolist()
    # save the shift_v_per_subject_mean to a CSV file
    df = pd.DataFrame.from_dict(shift_v_per_subject_mean, orient='index').T
    df.to_csv(f"{logs_folder}/shift_v_per_subject_mean.csv", index=True, header=True)

    interpretability_scores_per_neuron_per_subject = {}
    for subject, scores in elementwise_interpretability_scores_per_subject.items():
        all_stories = np.stack(scores, axis=0)
        interpretability_scores_per_neuron_per_subject[subject] = np.mean(all_stories, axis=0).tolist()
    
    average_interpretability_scores_per_subject = {}
    for subject, scores in interpretability_scores_per_subject.items():
        average_interpretability_scores_per_subject[subject] = np.mean(np.array(scores))

    # save the interpretability scores per subject to a CSV file
    df = pd.DataFrame.from_dict(interpretability_scores_per_neuron_per_subject, orient='index').T
    df.to_csv(f"{logs_folder}/interpretability_scores_per_subject.csv", index=True, header=True)

    # compute and save contrastive and independence scores per neuron per subject
    contrastive_scores_per_neuron_per_subject = {}
    for subject, scores in elementwise_contrastive_scores_per_subject.items():
        all_stories = np.stack(scores, axis=0)
        contrastive_scores_per_neuron_per_subject[subject] = np.mean(all_stories, axis=0).tolist()

    independence_scores_per_neuron_per_subject = {}
    for subject, scores in elementwise_independence_scores_per_subject.items():
        all_stories = np.stack(scores, axis=0)
        independence_scores_per_neuron_per_subject[subject] = np.mean(all_stories, axis=0).tolist()

    df_contrastive = pd.DataFrame.from_dict(contrastive_scores_per_neuron_per_subject, orient='index').T
    df_contrastive.to_csv(f"{logs_folder}/contrastive_scores_per_subject.csv", index=True, header=True)

    df_independence = pd.DataFrame.from_dict(independence_scores_per_neuron_per_subject, orient='index').T
    df_independence.to_csv(f"{logs_folder}/independence_scores_per_subject.csv", index=True, header=True)

    neuron_interpretability_score_subject_pairs = {}
    for i, row in df.iterrows():
        subject_i = row.argmax()
        neuron_interpretability_score_subject_pairs[i] = row[subject_i], df.columns[subject_i]

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
        "sae_release": sae_release,
        "sae_id": sae_id,
        "contrastive_score_mean": contrastive_score_mean,
        "independent_score_mean": independent_score_mean,
        "interpretability_score_mean": interpretability_score_mean,
        "total_rows": total_rows,
        "sae_config": sae_cfg,
        "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }

    with open(f"{logs_folder}/results.json", "w") as f:
        json.dump(results, f, indent=4)

    sw.stop()
    print(sw)

def load_sae(
    sae_release: str,
    sae_object_or_id: str,
    device: str,
    llm_dtype: str,
) -> tuple[str, SAE, float]:
    """
    Load the SAE from the given release and object or ID.
    """
    llm_dtype = general_utils.str_to_dtype(llm_dtype)
    # load the SAE
    sae_id, sae, sparsity = general_utils.load_and_format_sae(
        sae_release, sae_object_or_id, device
    )  # type: ignore
    sae = sae.to(device=device).to(dtype=llm_dtype, non_blocking=False)

    # check type
    if not isinstance(sae, SAE):
        raise TypeError(
            f"Expected SAE object, but got {type(sae)}. Please provide a valid SAE object."
        )
    else:
        print(f"Success! Loaded SAE {sae_id} from {sae_release}")

    return sae_id, sae, sparsity

def run_eval_once_pool(x):
    run_eval_once(
        **x,
    )

def run_eval(
    config: AutoInterpEvalConfig,
    selected_saes: list[tuple[str, str]] | list[tuple[str, SAE]],
    device: str,
    output_path: str,
) -> dict[str, Any]:
    # os.makedirs(output_path, exist_ok=True)


    dataset = load_dataset("GulkoA/contrastive-stories-v1", split="train")

    # this is nasty - I hate this - I know there is a better way
    args = [
        {
            "dataset": dataset,
            "device": device,
            "sae_release": release,
            "sae_id": sae,
            "config": config,
        }
        for release, sae in selected_saes
    ]

    print(f"Running evaluation for {len(args)} SAEs on {device}")

    with Pool(3) as pool:
        pool.map(run_eval_once_pool, args)

def create_config_and_selected_saes(
    args,
) -> tuple[AutoInterpEvalConfig, list[tuple[str, str]]]:
    
    selected_saes = get_saes_from_regex(args.sae_regex_pattern, args.sae_block_pattern)
    assert len(selected_saes) > 0, "No SAEs selected"

    saes_directory = get_pretrained_saes_directory()
    # imply model to be from the first selected SAE
    first_release = selected_saes[0][0]
    model_name = saes_directory[first_release].model

    config = AutoInterpEvalConfig(
        model_name=model_name,
    )

    if args.llm_batch_size is not None:
        config.llm_batch_size = args.llm_batch_size
    else:
        config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[
            config.model_name
        ]

    if args.llm_dtype is not None:
        config.llm_dtype = args.llm_dtype
    else:
        config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]

    if args.random_seed is not None:
        config.random_seed = args.random_seed


    releases = set([release for release, _ in selected_saes])

    print(f"Selected SAEs from releases: {releases}")

    for release, sae in selected_saes:
        print(f"Sample SAEs: {release}, {sae}")

    return config, selected_saes

def arg_parser():
    parser = argparse.ArgumentParser(description="Run auto interp evaluation")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed")

    parser.add_argument(
        "--sae_regex_pattern",
        type=str,
        required=True,
        help="Regex pattern for SAE selection",
    )
    parser.add_argument(
        "--sae_block_pattern",
        type=str,
        required=True,
        help="Regex pattern for SAE block selection",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="eval_results/autointerp",
        help="Output folder",
    )
    parser.add_argument(
        "--artifacts_path",
        type=str,
        default="artifacts",
        help="Path to save artifacts",
    )
    parser.add_argument(
        "--force_rerun", action="store_true", help="Force rerun of experiments"
    )
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=None,
        help="Batch size for LLM. If None, will be populated using LLM_NAME_TO_BATCH_SIZE",
    )
    parser.add_argument(
        "--llm_dtype",
        type=str,
        default=None,
        choices=[None, "float32", "float64", "float16", "bfloat16"],
        help="Data type for LLM. If None, will be populated using LLM_NAME_TO_DTYPE",
    )

    return parser

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    args = arg_parser().parse_args()
    device = general_utils.setup_environment()

    start_time = time.time()

    config, selected_saes = create_config_and_selected_saes(args)

    print(selected_saes)

    # create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    api_key = "HELLO WORLD"

    # run the evaluation on all selected SAEs
    results_dict = run_eval(
        config,
        selected_saes,
        device,
        args.output_folder,
    )

    end_time = time.time()

    print(f"Finished everything in {end_time - start_time} seconds")
