import argparse
import os
import csv
import time
import torch
from openai import OpenAI
from sae_lens import SAE
from tabulate import tabulate
from torch import Tensor
from tqdm import tqdm
from matplotlib.lines import Line2D
import re
import seaborn as sns
# from transformer_lens import HookedTransformer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
import torch
# import torch.nn as nn
import numpy as np
import pandas as pd
# from transformers import AutoTokenizer
# from sae_lens import SAE, HookedSAETransformer
# from collections import defaultdict
import matplotlib.pyplot as plt
import json
# from typing import Any, List
# import sae_bench.sae_bench_utils.activation_collection as activation_collection
import sae_bench.sae_bench_utils.general_utils as general_utils
# from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
# from sae_bench.sae_bench_utils.sae_selection_utils import (
#     get_saes_from_regex,
# )
# from stw import Stopwatch
# from datasets import load_dataset, Dataset
# import multiprocessing as mp
# from multiprocessing import Pool
# from functools import partial
# from typing import Tuple, Dict
# from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

TRAINED_FEATURES = ["contrastive_score", "independent_score", "joint_score", "sparsity"]

# ABLATION STUDY FEATURE CONFIG
# TRAINED_FEATURES = ["contrastive_score", "independent_score", "sparsity"]
# TRAINED_FEATURES = ["contrastive_score", "independent_score", "joint_score"]
# TRAINED_FEATURES = ["sparsity"]

def compare_rankings(df, predictions, target_feature="ground_truth") -> float:
    scores_pred = { i: predictions[i] for i in range(len(predictions)) }
    scores_gt = { i: df[target_feature][i] for i in range(len(df[target_feature])) }
    print("Scores_pred:", scores_pred)

    rankings1 = {k: i for i, k in enumerate(sorted(scores_pred, key=scores_pred.get, reverse=True))}
    rankings2 = {k: i for i, k in enumerate(sorted(scores_gt, key=scores_gt.get, reverse=True))}

    print("Rankings1:", rankings1)
    # print("Rankings2:", rankings2)

    concordant_pairs = 0
    discordant_pairs = 0
    for i in range(len(rankings1)):
        for j in range(i + 1, len(rankings1)):
            if (rankings1[i] < rankings1[j]) == (rankings2[i] < rankings2[j]):
                concordant_pairs += 1
            else:
                discordant_pairs += 1
    total_pairs = concordant_pairs + discordant_pairs

    return concordant_pairs / total_pairs if total_pairs > 0 else 0.0

def train_linear_regression(train_csv: str):
    df = pd.read_csv(train_csv)
    X_raw = df[TRAINED_FEATURES]
    y = df["ground_truth"]

    trained_scaler = StandardScaler()
    X_scaled = trained_scaler.fit_transform(X_raw)

    trained_model = LinearRegression()
    trained_model.fit(X_scaled, y)

    return {
        "model": trained_model,
        "scaler": trained_scaler,
        "coefficients": trained_model.coef_,
        "intercept": trained_model.intercept_,
        "r2_score": trained_model.score(X_scaled, y)
    }

def predict_with_global_model(df: pd.DataFrame, trained_model: LinearRegression, trained_scaler: StandardScaler):
    X_raw = df[TRAINED_FEATURES]
    X_scaled = trained_scaler.transform(X_raw)
    predictions = trained_model.predict(X_scaled)
    return predictions

def linear_regression_sae(csv_file: str, trained_scaler: StandardScaler, trained_model: LinearRegression):
    df = pd.read_csv(csv_file)

    # Truncate sae_variant to prefix before "_width"
    df["sae_group"] = df["sae_variant"].apply(lambda x: x.split("_width")[0])

    # Features and target
    X = df[TRAINED_FEATURES]
    # Normalize features
    X_scaled = trained_scaler.transform(X)
    y = df["ground_truth"]

    df["CE-Bench prediction"] = trained_model.predict(X_scaled)

    # Plotting
    comparison_axes = TRAINED_FEATURES + ["ground_truth"]
    titles = comparison_axes.copy()

    sae_groups = sorted(df["sae_group"].unique())
    palette = sns.color_palette("tab10", n_colors=len(sae_groups))

    fig, axes = plt.subplots(1, len(TRAINED_FEATURES)+1, figsize=((len(TRAINED_FEATURES)+1)*4, 4))
    axes = axes.flatten()

    for i, column in enumerate(comparison_axes):
        ax = axes[i]
        
        # Scatter plot
        sns.scatterplot(
            data=df,
            x=column,
            y="CE-Bench prediction",
            hue="sae_group",
            hue_order=sae_groups,
            palette=palette,
            alpha=0.8,
            s=70,
            ax=ax,
            legend=False  # Suppress internal legends
        )

        # Line plot
        sns.lineplot(
            data=df.sort_values(by=["sae_group", column]),
            x=column,
            y="CE-Bench prediction",
            hue="sae_group",
            hue_order=sae_groups,
            palette=palette,
            estimator=None,
            errorbar=None,
            linewidth=0.7,
            ax=ax,
            legend=False
        )

        ax.set_title(f"CE-Bench vs. {titles[i]}")
        ax.set_xlabel(titles[i])
        ax.set_ylabel("Model Prediction")

    # Shared legend at the bottom
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8)
        for i in range(len(sae_groups))
    ]
    labels = sae_groups

    fig.legend(
        handles,
        labels,
        title="SAE Variant",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(sae_groups),
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"ce_bench/sae_analysis.png", bbox_inches='tight')
    plt.show()



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


def linear_regression_width(csv_file: str, trained_scaler: StandardScaler, trained_model: LinearRegression):
    df = pd.read_csv(csv_file)

    # Treat width as a categorical group
    df["width_group"] = df["width"].astype(str)

    # Features and target
    X_raw = df[TRAINED_FEATURES]
    y = df["ground_truth"]

    # Normalize features
    scaler = trained_scaler
    X = scaler.fit_transform(X_raw)

    df["CE-Bench prediction"] = trained_model.predict(X)

    # Plotting
    comparison_axes = TRAINED_FEATURES + ["ground_truth"]
    titles = comparison_axes.copy()

    width_groups = sorted(df["width_group"].unique())
    palette = sns.color_palette("tab10", n_colors=len(width_groups))

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes = axes.flatten()

    for i, column in enumerate(comparison_axes):
        ax = axes[i]
        
        # Scatter plot
        sns.scatterplot(
            data=df,
            x=column,
            y="CE-Bench prediction",
            hue="width_group",
            hue_order=width_groups,
            palette=palette,
            alpha=0.8,
            s=70,
            ax=ax,
            legend=False
        )

        # Line plot
        sns.lineplot(
            data=df.sort_values(by=["width_group", column]),
            x=column,
            y="CE-Bench prediction",
            hue="width_group",
            hue_order=width_groups,
            palette=palette,
            estimator=None,
            errorbar=None,
            linewidth=0.7,
            ax=ax,
            legend=False
        )

        ax.set_title(f"CE-Bench vs. {titles[i]}")
        ax.set_xlabel(titles[i])
        ax.set_ylabel("Model Prediction")

    # Shared legend
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8)
        for i in range(len(width_groups))
    ]
    labels = width_groups

    fig.legend(
        handles,
        labels,
        title="SAE Width",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(width_groups),
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("ce_bench/width_analysis.png", bbox_inches='tight')
    plt.show()


def linear_regression_layer_type(csv_file: str, trained_scaler: StandardScaler, trained_model: LinearRegression):
    df = pd.read_csv(csv_file)

    # Treat sae_type as a categorical group
    df["type_group"] = df["layer_type"]

    # Features and target
    X_raw = df[TRAINED_FEATURES]

    # Normalize features using the trained scaler
    X = trained_scaler.transform(X_raw)

    # Predict
    df["CE-Bench prediction"] = trained_model.predict(X)

    # Plotting
    comparison_axes = TRAINED_FEATURES
    titles = comparison_axes.copy()

    type_groups = sorted(df["type_group"].unique())
    palette = sns.color_palette("tab10", n_colors=len(type_groups))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes = axes.flatten()

    for i, column in enumerate(comparison_axes):
        ax = axes[i]

        # Scatter plot
        sns.scatterplot(
            data=df,
            x=column,
            y="CE-Bench prediction",
            hue="type_group",
            hue_order=type_groups,
            palette=palette,
            alpha=0.8,
            s=70,
            ax=ax,
            legend=False
        )

        # Line plot
        sns.lineplot(
            data=df.sort_values(by=["type_group", column]),
            x=column,
            y="CE-Bench prediction",
            hue="type_group",
            hue_order=type_groups,
            palette=palette,
            estimator=None,
            errorbar=None,
            linewidth=0.7,
            ax=ax,
            legend=False
        )

        ax.set_title(f"CE-Bench vs. {titles[i]}")
        ax.set_xlabel(titles[i])
        ax.set_ylabel("Model Prediction")

    # Shared legend
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8)
        for i in range(len(type_groups))
    ]

    labels = type_groups

    fig.legend(
        handles,
        labels,
        title="Layer Type",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(type_groups),
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("ce_bench/layer_type_analysis.png", bbox_inches='tight')
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

    trained_results = train_linear_regression(
        #train_csv="ce_bench/ablation_study/ABLATION_MEAN_TRAINING_DATA.csv"
        #train_csv="ce_bench/ablation_study/ABLATION_OUTLIER_TRAINING_DATA.csv"
        train_csv="ce_bench/data_processing/TRAINING_DATA.csv"
    )


    trained_scaler = trained_results["scaler"]
    trained_model = trained_results["model"]
    trained_coefficients = trained_results["coefficients"]
    trained_intercept = trained_results["intercept"]
    trained_r2_score = trained_results["r2_score"]
    print(f"Trained model R² score: {trained_r2_score:.4f}")

    training_data = pd.read_csv("ce_bench/data_processing/TRAINING_DATA.csv")

    predicted = predict_with_global_model(
        df=training_data,
        trained_model=trained_model,
        trained_scaler=trained_scaler
    )

    ranking_score = compare_rankings(
        df=training_data,
        predictions=predicted,
        target_feature="ground_truth"
    )

    print(f"Ranking score: {ranking_score:.4f}")


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
        width = "width_16k"
        metric = "max"

        # layer_type_analysis(
        #     sae_release_series=sae_release_series,
        #     type_pool=type_pool,
        #     layer=layer,
        #     width=width,
        #     metric=metric,
        # )

        linear_regression_layer_type(
            csv_file="ce_bench/LAYER_TYPE_ANALYSIS_METRICS.csv",
            trained_scaler=trained_scaler,
            trained_model=trained_model
        )
            
    elif args.task_name == "width":
        layer = "layer_12"
        pooling_metric = "max"

        # width_analysis(
        #     base_dir=".",
        #     dataset_ver="v4",
        #     metric=pooling_metric
        # )
        
        linear_regression_width(
            csv_file="ce_bench/WIDTH_ANALYSIS_METRICS.csv",
            trained_scaler=trained_scaler,
            trained_model=trained_model
        )

    elif args.task_name == "sae":
        sae_release_series = "sae_bench_gemma-2-2b_"
        sae_pool = [
                    "batch_top_k_width-2pow16_date-0107", 
                    "gated_width-2pow16_date-0107", 
                    "p_anneal_width-2pow16_date-0107", 
                    "standard_new_width-2pow16_date-0107",
                    "top_k_width-2pow16_date-0107",
                    "jump_relu_width-2pow16_date-0107",
                    "matryoshka_batch_top_k_width-2pow16_date-0107"
                    ]
        block_num = 12
        # dataset_ver_1 = "v2"
        # v2_runs, v2_avg = sae_analysis(
        #     sae_release_series=sae_release_series,
        #     sae_pool=sae_pool,
        #     block_num=block_num,
        #     dataset_ver=dataset_ver_1,
        # )

        dataset_ver_2 = "v3"
        # metric_zoo = [
        #     "max",
        #     "mean",
        #     "outlier_count_1_both",
        #     "outlier_count_1_upper",
        #     "outlier_count_2_both",
        #     "outlier_count_2_upper",
        #     "outlier_count_3_both",
        #     "outlier_count_3_upper",
        # ]

        # for metric in metric_zoo:

        #     v3_runs, v3_avg = sae_analysis(
        #         sae_release_series=sae_release_series,
        #         sae_pool=sae_pool,
        #         block_num=block_num,
        #         dataset_ver=dataset_ver_2,
        #         metric=metric
        #     )

        dataset_final = "v4"
        pooling_metric = "max"
        # sae_analysis(
        #     sae_release_series=sae_release_series,
        #     sae_pool=sae_pool,
        #     block_num=block_num,
        #     dataset_ver=dataset_final,
        #     metric=pooling_metric,
        # )

        linear_regression_sae(
            #csv_file="ce_bench/ablation_study/ABLATION_MEAN_SAE_ANALYSIS_METRICS.csv",
            #csv_file="ce_bench/ablation_study/ABLATION_OUTLIER_SAE_ANALYSIS_METRICS.csv",
            csv_file="ce_bench/data_processing/SAE_ANALYSIS_METRICS.csv",
            trained_scaler=trained_scaler,
            trained_model=trained_model
        )



        # plot_v2_vs_v3_scores(v2_avg, v3_avg, sae_pool)
    else:
        raise ValueError(f"Unknown task name: {args.task_name}")
