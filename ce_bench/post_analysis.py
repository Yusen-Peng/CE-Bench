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
from typing import Any, List
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
from functools import partial
from typing import Tuple, Dict
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory



def depth_analysis(sae_release: str, width: str):
    base_path = os.path.expanduser(f"interpretability_eval/{sae_release}")
    sparsities_and_scores = {}

    # traverse every single layer folder
    for layer_folder in os.listdir(base_path):
        layer_path = os.path.join(base_path, layer_folder)

        # "layer_0", "layer_1", "layer_2", etc.
        layer_key = layer_folder
        sparsities_and_scores[layer_key] = {}
        width_path = os.path.join(layer_path, f"width_{width}")
        for subfolder in os.listdir(width_path):
            # retrieve the L0 sparsity value
            l0_value = int(subfolder.split("_")[-1])
            results_path = os.path.join(width_path, subfolder, "results.json")

            if os.path.exists(results_path):
                with open(results_path, "r") as f:
                    try:
                        results: dict = json.load(f)
                        contrastive = results.get("contrastive_score_mean")
                        independence = results.get("independent_score_mean")
                        interpretability = results.get("interpretability_score_mean")

                        if all(v is not None for v in [contrastive, independence, interpretability]):
                            sparsities_and_scores[layer_key][l0_value] = (
                                contrastive,
                                independence,
                                interpretability
                            )
                    except json.JSONDecodeError:
                        print(f"Invalid JSON in {results_path} found! Skipping this file.")

    layers = []
    contrastive_avgs = []
    independence_avgs = []
    interpretability_avgs = []

    contrastive_points = []
    independence_points = []
    interpretability_points = []

    for layer, sparsity_dict in sparsities_and_scores.items():
        if not sparsity_dict:
            continue

        layer_idx = int(layer.split("_")[-1])
        scores = list(sparsity_dict.values())

        # Store scatter points
        for s in scores:
            contrastive_points.append((layer_idx, s[0]))
            independence_points.append((layer_idx, s[1]))
            interpretability_points.append((layer_idx, s[2]))

        # Compute per-layer averages
        contrastive_avg = sum(s[0] for s in scores) / len(scores)
        independence_avg = sum(s[1] for s in scores) / len(scores)
        interpretability_avg = sum(s[2] for s in scores) / len(scores)

        layers.append(layer_idx)
        contrastive_avgs.append(contrastive_avg)
        independence_avgs.append(independence_avg)
        interpretability_avgs.append(interpretability_avg)

    # Sort everything by layer index
    sorted_layers, contrastive_avgs, independence_avgs, interpretability_avgs = zip(
        *sorted(zip(layers, contrastive_avgs, independence_avgs, interpretability_avgs))
    )

    # Separate scatter data
    contrastive_points.sort()
    independence_points.sort()
    interpretability_points.sort()

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Score vs. Layer Depth for {sae_release} width {width}", fontsize=14)

    # Contrastive
    axs[0].scatter(*zip(*contrastive_points), alpha=1.0, color='skyblue', label="All Sparsities")
    axs[0].plot(sorted_layers, contrastive_avgs, marker='o', color='blue', label="Layer Avg", markersize=2)
    axs[0].set_title("Contrastive Score")
    axs[0].set_xlabel("Layer")
    axs[0].set_ylabel("Score")
    axs[0].grid(True)
    axs[0].legend()

    # Independence
    axs[1].scatter(*zip(*independence_points), alpha=1.0, color='lightgreen', label="All Sparsities")
    axs[1].plot(sorted_layers, independence_avgs, marker='o', color='green', label="Layer Avg", markersize=2)
    axs[1].set_title("Independence Score")
    axs[1].set_xlabel("Layer")
    axs[1].set_ylabel("Score")
    axs[1].grid(True)
    axs[1].legend()

    # Interpretability
    axs[2].scatter(*zip(*interpretability_points), alpha=1.0, color='violet', label="All Sparsities")
    axs[2].plot(sorted_layers, interpretability_avgs, marker='o', color='purple', label="Layer Avg", markersize=2)
    axs[2].set_title("Interpretability Score")
    axs[2].set_xlabel("Layer")
    axs[2].set_ylabel("Score")
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"figures/depth_analysis_{sae_release}_{width}.png")
    plt.show()



def layer_type_analysis(sae_release_series: str, type_pool: List[str], layer: str, width: str):
    results_by_type = {}

    for layer_type in type_pool:
        sae_release = f"{sae_release_series}{layer_type}"
        base_path = os.path.expanduser(f"interpretability_eval/{sae_release}/{layer}/width_{width}")        
        scores = []

        for sparsity_folder in os.listdir(base_path):
            results_path = os.path.join(base_path, sparsity_folder, "results.json")

            try:
                with open(results_path, "r") as f:
                    results = json.load(f)
                    contrastive = results.get("contrastive_score_mean")
                    independence = results.get("independent_score_mean")
                    interpretability = results.get("interpretability_score_mean")
                    if all(v is not None for v in [contrastive, independence, interpretability]):
                        scores.append((contrastive, independence, interpretability))
            except json.JSONDecodeError:
                print(f"Invalid JSON in {results_path}")

        if scores:
            results_by_type[layer_type] = scores

    # Prepare data
    layer_type_indices = []
    contrastive_avgs = []
    independence_avgs = []
    interpretability_avgs = []

    contrastive_points = []
    independence_points = []
    interpretability_points = []

    for i, layer_type in enumerate(type_pool):
        scores = results_by_type.get(layer_type, [])
        if not scores:
            continue

        layer_type_indices.append(i)

        for score in scores:
            contrastive_points.append((i, score[0]))
            independence_points.append((i, score[1]))
            interpretability_points.append((i, score[2]))

        contrastive_avgs.append(np.mean([s[0] for s in scores]))
        independence_avgs.append(np.mean([s[1] for s in scores]))
        interpretability_avgs.append(np.mean([s[2] for s in scores]))

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Score vs. Layer Type for {sae_release_series} at {layer} with width {width}", fontsize=14)

    xticks = list(range(len(type_pool)))
    axs[0].scatter(*zip(*contrastive_points), alpha=1.0, color='skyblue', s=20, label="All Sparsities")
    axs[0].plot(layer_type_indices, contrastive_avgs, color='blue', marker='o', markersize=2, label="Avg")
    axs[0].set_title("Contrastive Score")
    axs[0].set_xticks(xticks)
    axs[0].set_xticklabels(type_pool)
    axs[0].set_ylabel("Score")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].scatter(*zip(*independence_points), alpha=1.0, color='lightgreen', s=20, label="All Sparsities")
    axs[1].plot(layer_type_indices, independence_avgs, color='green', marker='o', markersize=2, label="Avg")
    axs[1].set_title("Independence Score")
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(type_pool)
    axs[1].set_ylabel("Score")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].scatter(*zip(*interpretability_points), alpha=1.0, color='violet', s=20, label="All Sparsities")
    axs[2].plot(layer_type_indices, interpretability_avgs, color='purple', marker='o', markersize=2, label="Avg")
    axs[2].set_title("Interpretability Score")
    axs[2].set_xticks(xticks)
    axs[2].set_xticklabels(type_pool)
    axs[2].set_ylabel("Score")
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"figures/layer_type_analysis_{sae_release_series}_{layer}_{width}.png")


def width_analysis(sae_release: str, layer: str):
    base_path = os.path.expanduser(f"interpretability_eval/{sae_release}/{layer}")
    widths_and_scores = {}

    for width_folder in os.listdir(base_path):

        width_key = width_folder.split("_")[-1]
        width_path = os.path.join(base_path, width_folder)

        scores = []
        for sparsity_folder in os.listdir(width_path):
            if not sparsity_folder.startswith("average_l0_"):
                continue

            results_path = os.path.join(width_path, sparsity_folder, "results.json")
            if not os.path.exists(results_path):
                continue

            try:
                with open(results_path, "r") as f:
                    results = json.load(f)
                    contrastive = results.get("contrastive_score_mean")
                    independence = results.get("independent_score_mean")
                    interpretability = results.get("interpretability_score_mean")
                    if all(v is not None for v in [contrastive, independence, interpretability]):
                        scores.append((contrastive, independence, interpretability))
            except json.JSONDecodeError:
                print(f"Invalid JSON in {results_path}. Skipping.")

        if scores:
            widths_and_scores[width_key] = scores

    # Now process and plot
    widths = []
    contrastive_avgs = []
    independence_avgs = []
    interpretability_avgs = []

    contrastive_points = []
    independence_points = []
    interpretability_points = []

    for width_key, score_list in widths_and_scores.items():
        width_val = int(width_key.replace("k", "000"))
        for score in score_list:
            contrastive_points.append((width_val, score[0]))
            independence_points.append((width_val, score[1]))
            interpretability_points.append((width_val, score[2]))

        # Averaging
        contrastive_avg = sum(s[0] for s in score_list) / len(score_list)
        independence_avg = sum(s[1] for s in score_list) / len(score_list)
        interpretability_avg = sum(s[2] for s in score_list) / len(score_list)

        widths.append(width_val)
        contrastive_avgs.append(contrastive_avg)
        independence_avgs.append(independence_avg)
        interpretability_avgs.append(interpretability_avg)

    # Sort by width value
    sorted_data = sorted(zip(widths, contrastive_avgs, independence_avgs, interpretability_avgs))
    widths, contrastive_avgs, independence_avgs, interpretability_avgs = zip(*sorted_data)

    contrastive_points.sort()
    independence_points.sort()
    interpretability_points.sort()

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Score vs. SAE Width for {sae_release} {layer}", fontsize=14)

    axs[0].scatter(*zip(*contrastive_points), alpha=1.0, color='skyblue', s=20, label="All Sparsities")
    axs[0].plot(widths, contrastive_avgs, marker='o', color='blue', markersize=2, label="Width Avg")
    axs[0].set_title("Contrastive Score")
    axs[0].set_xlabel("Latent Width")
    axs[0].set_xscale("log", base=2)
    axs[0].set_ylabel("Score")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].scatter(*zip(*independence_points), alpha=1.0, color='lightgreen', s=20, label="All Sparsities")
    axs[1].plot(widths, independence_avgs, marker='o', color='green', markersize=2, label="Width Avg")
    axs[1].set_title("Independence Score")
    axs[1].set_xlabel("Latent Width")
    axs[1].set_xscale("log", base=2)
    axs[1].set_ylabel("Score")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].scatter(*zip(*interpretability_points), alpha=1.0, color='violet', s=20, label="All Sparsities")
    axs[2].plot(widths, interpretability_avgs, marker='o', color='purple', markersize=2, label="Width Avg")
    axs[2].set_title("Interpretability Score")
    axs[2].set_xlabel("Latent Width")
    axs[2].set_xscale("log", base=2)
    axs[2].set_ylabel("Score")
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"figures/width_analysis_{sae_release}_{layer}.png")
    plt.show()


def sae_analysis(sae_release_series: str, sae_pool: List[str], block_num: int, dataset_ver: str):
    results_by_sae = {}
    block_prefix = f"blocks.{block_num}.hook_resid_post"

    for sae_variant in sae_pool:
        sae_release = f"{sae_release_series}{sae_variant}"
        base_path = os.path.expanduser(f"{dataset_ver}_results/{sae_release}")

        scores = []
        for subfolder in os.listdir(base_path):
            results_path = os.path.join(base_path, subfolder, "results.json")
            try:
                with open(results_path, "r") as f:
                    results = json.load(f)
                    contrastive = results.get("contrastive_score_mean")
                    independence = results.get("independent_score_mean")
                    interpretability = results.get("interpretability_score_mean")
                    if all(v is not None for v in [contrastive, independence, interpretability]):
                        scores.append((contrastive, independence, interpretability))
            except json.JSONDecodeError:
                print(f"[WARN] Invalid JSON: {results_path}")

        if scores:
            results_by_sae[sae_variant] = scores

    average_scores_by_sae = {
        variant: (
            np.mean([s[0] for s in scores]),
            np.mean([s[1] for s in scores]),
            np.mean([s[2] for s in scores]),
        )
        for variant, scores in results_by_sae.items()
    }

    # Visualization (unchanged)
    sae_indices = []
    contrastive_avgs = []
    independence_avgs = []
    interpretability_avgs = []

    contrastive_points = []
    independence_points = []
    interpretability_points = []

    for i, sae_variant in enumerate(sae_pool):
        scores = results_by_sae.get(sae_variant, [])
        sae_indices.append(i)
        for s in scores:
            contrastive_points.append((i, s[0]))
            independence_points.append((i, s[1]))
            interpretability_points.append((i, s[2]))

        contrastive_avgs.append(np.mean([s[0] for s in scores]) if scores else None)
        independence_avgs.append(np.mean([s[1] for s in scores]) if scores else None)
        interpretability_avgs.append(np.mean([s[2] for s in scores]) if scores else None)

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Score vs. SAE Variant (Block {block_num}) for dataset {dataset_ver}", fontsize=14)

    # Truncate SAE variant names
    sae_labels = [sae.split("_")[0] for sae in sae_pool]
    sae_labels = ["p_anneal" if s == "p" else s for s in sae_labels]
    xticks = list(range(len(sae_labels)))

    axs[0].scatter(*zip(*contrastive_points), alpha=1.0, color='skyblue', s=20, label="All Runs")
    axs[0].plot(sae_indices, contrastive_avgs, color='blue', marker='o', markersize=2, label="Avg")
    axs[0].set_title("Contrastive Score")
    axs[0].set_xticks(xticks)
    axs[0].set_xticklabels(sae_labels, rotation=15)
    axs[0].set_ylabel("Score")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].scatter(*zip(*independence_points), alpha=1.0, color='lightgreen', s=20, label="All Runs")
    axs[1].plot(sae_indices, independence_avgs, color='green', marker='o', markersize=2, label="Avg")
    axs[1].set_title("Independence Score")
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(sae_labels, rotation=15)
    axs[1].set_ylabel("Score")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].scatter(*zip(*interpretability_points), alpha=1.0, color='violet', s=20, label="All Runs")
    axs[2].plot(sae_indices, interpretability_avgs, color='purple', marker='o', markersize=2, label="Avg")
    axs[2].set_title("Interpretability Score")
    axs[2].set_xticks(xticks)
    axs[2].set_xticklabels(sae_labels, rotation=15)
    axs[2].set_ylabel("Score")
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"figures/sae_analysis_{sae_release_series}{block_num}_{dataset_ver}.png")

    return results_by_sae, average_scores_by_sae




def plot_v2_vs_v3_scores(v2_avg: dict, v3_avg: dict, sae_pool: list):
    score_types = ["contrastive", "independence", "interpretability"]
    colors = ['skyblue', 'lightgreen', 'violet']

    # Map each SAE variant to a display label (truncate before first underscore)
    sae_labels = [s.split("_")[0] for s in sae_pool]
    sae_labels = ["p_anneal" if label == "p" else label for label in sae_labels]

    # Store unpacked score arrays
    v2_scores = {k: [] for k in score_types}
    v3_scores = {k: [] for k in score_types}
    valid_labels = []

    for sae, label in zip(sae_pool, sae_labels):
        v2_tuple = v2_avg.get(sae)
        v3_tuple = v3_avg.get(sae)

        if v2_tuple and v3_tuple:
            v2_scores["contrastive"].append(v2_tuple[0])
            v2_scores["independence"].append(v2_tuple[1])
            v2_scores["interpretability"].append(v2_tuple[2])

            v3_scores["contrastive"].append(v3_tuple[0])
            v3_scores["independence"].append(v3_tuple[1])
            v3_scores["interpretability"].append(v3_tuple[2])

            valid_labels.append(label)
        else:
            print(f"[WARN] Missing data for {sae}")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("v2 vs. v3 Score Comparison per SAE Variant", fontsize=14)

    for i, score_type in enumerate(score_types):
        x = v2_scores[score_type]
        y = v3_scores[score_type]

        axs[i].scatter(x, y, color=colors[i], s=40, label="SAE Variants")

        # Diagonal line
        min_val = min(x + y)
        max_val = max(x + y)
        axs[i].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.6, label="y = x")

        # Annotations
        for j, label in enumerate(valid_labels):
            axs[i].annotate(label, (x[j] + 0.5, y[j] + 0.5), fontsize=8)

        axs[i].set_title(f"{score_type.capitalize()} Score")
        axs[i].set_xlabel("v2")
        axs[i].set_ylabel("v3")
        axs[i].grid(True)
        axs[i].set_aspect('equal', adjustable='box')
        axs[i].legend(loc='upper left', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("figures/v2_vs_v3_score_scatter.png")
    plt.show()


def arg_parser():
    parser = argparse.ArgumentParser(description="Post analysis of neuron steering")
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="The name of the task to use.",
    )
    return parser



if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    device = general_utils.setup_environment()

    if args.task_name == "depth":
        sae_release = "gemma-scope-2b-pt-res"
        width = "16k"
        depth_analysis(
            sae_release=sae_release,
            width=width,
        )

    elif args.task_name == "layer_type":
        sae_release_series = "gemma-scope-2b-pt-"
        type_pool = ["att", "mlp", "res"]
        layer = "layer_12"
        width = "16k"

        layer_type_analysis(
            sae_release_series=sae_release_series,
            type_pool=type_pool,
            layer=layer,
            width=width,
        )
            
    elif args.task_name == "width":
        sae_release = "gemma-scope-2b-pt-res"
        layer = "layer_12"
        width_analysis(
            sae_release=sae_release,
            layer=layer,
        )
    elif args.task_name == "sae":
        sae_release_series = "sae_bench_gemma-2-2b_"
        sae_pool = ["batch_top_k_width-2pow16_date-0107", 
                    "gated_width-2pow16_date-0107", 
                    "p_anneal_width-2pow16_date-0107", 
                    "standard_new_width-2pow16_date-0107",
                    "top_k_width-2pow16_date-0107"
                    ]
        block_num = 12
        dataset_ver_1 = "v2"
        v2_runs, v2_avg = sae_analysis(
            sae_release_series=sae_release_series,
            sae_pool=sae_pool,
            block_num=block_num,
            dataset_ver=dataset_ver_1,
        )

        dataset_ver_2 = "v3"
        v3_runs, v3_avg = sae_analysis(
            sae_release_series=sae_release_series,
            sae_pool=sae_pool,
            block_num=block_num,
            dataset_ver=dataset_ver_2,
        )

        plot_v2_vs_v3_scores(v2_avg, v3_avg, sae_pool)
    else:
        raise ValueError(f"Unknown task name: {args.task_name}")
    



