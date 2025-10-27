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

TRAINED_FEATURES = ["contrastive_score", "independent_score", "sparsity"]

# ABLATION STUDY FEATURE CONFIG
# TRAINED_FEATURES = ["contrastive_score", "independent_score", "joint_score"]
# TRAINED_FEATURES = ["contrastive_score", "independent_score", "joint_score", "sparsity"]
# TRAINED_FEATURES = ["sparsity"]
# TRAINED_FEATURES = ["independent_score", "sparsity"]
# TRAINED_FEATURES = ["contrastive_score", "sparsity"]



def compare_rankings(df, predictions, target_feature="ground_truth") -> float:
    scores_pred = { i: predictions[i] for i in range(len(predictions)) }
    scores_gt = { i: df[target_feature][i] for i in range(len(df[target_feature])) }
    #print("Scores_pred:", scores_pred)

    rankings1 = {k: i for i, k in enumerate(sorted(scores_pred, key=scores_pred.get, reverse=True))}
    rankings2 = {k: i for i, k in enumerate(sorted(scores_gt, key=scores_gt.get, reverse=True))}

    #print("Rankings1:", rankings1)
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



def _average_rank(values: np.ndarray) -> np.ndarray:
    """Tie-aware average ranks (0-based)."""
    n = values.size
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + j)  # average of tied positions (0-based)
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks

def spearman_correlation(df, predictions, target_feature: str = "ground_truth") -> float:
    """Spearman's ρ (Pearson on ranks), returns in [-1, 1] or NaN."""
    y_true = np.asarray(df[target_feature].to_numpy(), dtype=float)
    y_pred = np.asarray(predictions, dtype=float)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"Length mismatch: {y_true.shape[0]} ground-truth vs {y_pred.shape[0]} predictions")

    r_true = _average_rank(y_true)
    r_pred = _average_rank(y_pred)

    n = r_true.size
    x = r_true - r_true.mean()
    y = r_pred - r_pred.mean()
    sx = r_true.std(ddof=0)
    sy = r_pred.std(ddof=0)
    if sx == 0 or sy == 0:
        return np.nan

    rho = float((x @ y) / (n * sx * sy))
    # numerical guard
    return float(np.clip(rho, -1.0, 1.0))

def pearson_correlation(df, predictions, target_feature: str = "ground_truth") -> float:
    """Pearson's r, returns in [-1, 1] or NaN."""
    y_true = np.asarray(df[target_feature].to_numpy(), dtype=float)
    y_pred = np.asarray(predictions, dtype=float)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"Length mismatch: {y_true.shape[0]} ground-truth vs {y_pred.shape[0]} predictions")

    n = y_true.size
    x = y_true - y_true.mean()
    y = y_pred - y_pred.mean()
    sx = y_true.std(ddof=0)
    sy = y_pred.std(ddof=0)
    if sx == 0 or sy == 0:
        return np.nan

    r = float((x @ y) / (n * sx * sy))
    # numerical guard
    return float(np.clip(r, -1.0, 1.0))


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

    df["CE-Bench prediction"] = (df['contrastive_score'] + df['independent_score']) - 0.25 * df['sparsity']

    # Plotting
    #comparison_axes = TRAINED_FEATURES + ["ground_truth"]
    comparison_axes = TRAINED_FEATURES
    titles = comparison_axes.copy()

    sae_groups = sorted(df["sae_group"].unique())
    palette = sns.color_palette("tab10", n_colors=len(sae_groups))

    fig, axes = plt.subplots(1, len(TRAINED_FEATURES), figsize=((len(TRAINED_FEATURES))*4, 4))
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

        ax.set_title(f"CE-Bench vs. {titles[i]}", fontsize=16)
        ax.set_xlabel(titles[i], fontsize=16)
        ax.set_ylabel("CE-Bench prediction", fontsize=16)

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
        frameon=False,
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"ce_bench/sae_analysis.png", bbox_inches='tight')
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
    df["CE-Bench prediction"] = (df['contrastive_score'] + df['independent_score']) - 0.25 * df['sparsity']


    # Plotting
    comparison_axes = TRAINED_FEATURES
    titles = comparison_axes.copy()

    type_groups = sorted(df["type_group"].unique())
    palette = sns.color_palette("tab10", n_colors=len(type_groups))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
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

        ax.set_title(f"CE-Bench vs. {titles[i]}", fontsize=16)
        ax.set_xlabel(titles[i], fontsize=16)
        ax.set_ylabel("CE-Bench prediction", fontsize=16)

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
        frameon=False,
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("ce_bench/layer_type_analysis.png", bbox_inches='tight')
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

    df["CE-Bench prediction"] = (df['contrastive_score'] + df['independent_score']) - 0.25 * df['sparsity']

    # Plotting
    #comparison_axes = TRAINED_FEATURES + ["ground_truth"]
    comparison_axes = TRAINED_FEATURES
    titles = comparison_axes.copy()

    width_groups = sorted(df["width_group"].unique())
    palette = sns.color_palette("tab10", n_colors=len(width_groups))

    fig, axes = plt.subplots(1, len(TRAINED_FEATURES), figsize=(4*(len(TRAINED_FEATURES)), 4))
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

        ax.set_title(f"CE-Bench vs. {titles[i]}", fontsize=16)
        ax.set_xlabel(titles[i], fontsize=16)
        ax.set_ylabel("CE-Bench prediction", fontsize=16)

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
        frameon=False,
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("ce_bench/width_analysis.png", bbox_inches='tight')
    plt.show()



def linear_regression_depth(csv_file: str, trained_scaler: StandardScaler, trained_model: LinearRegression):
    df = pd.read_csv(csv_file)

    # Normalize and extract layer index (strip prefix if needed)
    if "layer" in df.columns and df["layer"].dtype == object:
        df["layer"] = df["layer"].apply(lambda x: int(x.split("_")[-1]))

    # Features and target
    X_raw = df[TRAINED_FEATURES]

    X = trained_scaler.transform(X_raw)
    df["CE-Bench prediction"] = (df['contrastive_score'] + df['independent_score']) - 0.25 * df['sparsity']


    # Plotting setup
    comparison_axes = TRAINED_FEATURES
    titles = comparison_axes.copy()

    layers = sorted(df["layer"].unique())
    palette = sns.color_palette("tab10", n_colors=len(layers))

    fig, axes = plt.subplots(1, len(TRAINED_FEATURES), figsize=(4*(len(TRAINED_FEATURES)), 4))
    axes = axes.flatten()

    for i, column in enumerate(comparison_axes):
        ax = axes[i]

        # Scatter by layer
        sns.scatterplot(
            data=df,
            x=column,
            y="CE-Bench prediction",
            hue="layer",
            palette=palette,
            alpha=0.8,
            s=70,
            ax=ax,
            legend=False
        )

        # Lineplot
        sns.lineplot(
            data=df.sort_values(by=["layer", column]),
            x=column,
            y="CE-Bench prediction",
            hue="layer",
            palette=palette,
            estimator=None,
            errorbar=None,
            linewidth=0.7,
            ax=ax,
            legend=False
        )

        ax.set_title(f"CE-Bench vs. {titles[i]}", fontsize=16)
        ax.set_xlabel(titles[i], fontsize=16)
        ax.set_ylabel("CE-Bench prediction", fontsize=16)

    # Shared legend
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8)
        for i in range(len(layers))
    ]
    labels = [f"Layer {l}" for l in layers]

    fig.legend(
        handles,
        labels,
        title="Layer Index",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(layers),
        frameon=False,
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("ce_bench/depth_analysis.png", bbox_inches='tight')
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



def main():
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
    #print(f"Trained model R² score: {trained_r2_score:.4f}")

    # Approach 1 (primary approach): proxy learning

    training_data = pd.read_csv("ce_bench/data_processing/TRAINING_DATA.csv")
    #training_data = pd.read_csv("ce_bench/ablation_study/ABLATION_MEAN_TRAINING_DATA.csv")
    #training_data = pd.read_csv("ce_bench/ablation_study/ABLATION_OUTLIER_TRAINING_DATA.csv")

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

    # print out the ranking score
    print(f"Ranking score of proxy learning: {ranking_score:.4f}") # should be 0.7598


    # Approach 2: simple average of contrastive and independent scores
    ranking_score = compare_rankings(
        df=training_data,
        predictions=(training_data["contrastive_score"] + training_data["independent_score"]),
        target_feature="ground_truth"
    )

    # print out the ranking score
    print(f"Ranking score of simple average: {ranking_score:.4f}")


    # Approach 3: simple average of contrastive and independent scores with sparsity penalty

    PREDICTIONS = (training_data["contrastive_score"] + training_data["independent_score"]) - 0.25 * training_data["sparsity"]
    #PREDICTIONS = training_data["contrastive_score"]
    #PREDICTIONS = training_data["independent_score"]
    #PREDICTIONS =  -training_data["sparsity"]

    ranking_score = compare_rankings(
        df=training_data,
        predictions=PREDICTIONS,
        target_feature="ground_truth"
    )

    # print out the ranking score
    print(f"Ranking score of simple average with sparsity penalty: {ranking_score:.4f}")

    rho = spearman_correlation(
        df=training_data,
        predictions=PREDICTIONS,
        target_feature="ground_truth"
    )
    r = pearson_correlation(
        df=training_data, 
        predictions=PREDICTIONS,
        target_feature="ground_truth"
    )
    print(f"Spearman correlation of simple average with sparsity penalty: {rho:.4f}")
    print(f"Pearson correlation of simple average with sparsity penalty: {r:.4f}")


    # if args.task_name == "depth":
    #     sae_release = "gemma-scope-2b-pt-res"
    #     width = "16k"
    #     # depth_analysis(
    #     #     sae_release=sae_release,
    #     #     width=width,
    #     # )
    #     linear_regression_depth(
    #         csv_file="ce_bench/data_processing/DEPTH_ANALYSIS_METRICS.csv",
    #         trained_scaler=trained_scaler,
    #         trained_model=trained_model
    #     )

    # elif args.task_name == "layer_type":
    #     sae_release_series = "gemma-scope-2b-pt-"
    #     type_pool = ["att", "mlp", "res"]
    #     layer = "layer_12"
    #     width = "width_16k"
    #     metric = "max"

    #     # layer_type_analysis(
    #     #     sae_release_series=sae_release_series,
    #     #     type_pool=type_pool,
    #     #     layer=layer,
    #     #     width=width,
    #     #     metric=metric,
    #     # )

    #     linear_regression_layer_type(
    #         csv_file="ce_bench/data_processing/LAYER_TYPE_ANALYSIS_METRICS.csv",
    #         trained_scaler=trained_scaler,
    #         trained_model=trained_model
    #     )
            
    # elif args.task_name == "width":
    #     layer = "layer_12"
    #     pooling_metric = "max"

    #     # width_analysis(
    #     #     base_dir=".",
    #     #     dataset_ver="v4",
    #     #     metric=pooling_metric
    #     # )
        
    #     linear_regression_width(
    #         csv_file="ce_bench/data_processing/WIDTH_ANALYSIS_METRICS.csv",
    #         trained_scaler=trained_scaler,
    #         trained_model=trained_model
    #     )

    # elif args.task_name == "sae":
    #     sae_release_series = "sae_bench_gemma-2-2b_"
    #     sae_pool = [
    #                 "batch_top_k_width-2pow16_date-0107", 
    #                 "gated_width-2pow16_date-0107", 
    #                 "p_anneal_width-2pow16_date-0107", 
    #                 "standard_new_width-2pow16_date-0107",
    #                 "top_k_width-2pow16_date-0107",
    #                 "jump_relu_width-2pow16_date-0107",
    #                 "matryoshka_batch_top_k_width-2pow16_date-0107"
    #                 ]
    #     block_num = 12
    #     # dataset_ver_1 = "v2"
    #     # v2_runs, v2_avg = sae_analysis(
    #     #     sae_release_series=sae_release_series,
    #     #     sae_pool=sae_pool,
    #     #     block_num=block_num,
    #     #     dataset_ver=dataset_ver_1,
    #     # )

    #     dataset_ver_2 = "v3"
    #     # metric_zoo = [
    #     #     "max",
    #     #     "mean",
    #     #     "outlier_count_1_both",
    #     #     "outlier_count_1_upper",
    #     #     "outlier_count_2_both",
    #     #     "outlier_count_2_upper",
    #     #     "outlier_count_3_both",
    #     #     "outlier_count_3_upper",
    #     # ]

    #     # for metric in metric_zoo:

    #     #     v3_runs, v3_avg = sae_analysis(
    #     #         sae_release_series=sae_release_series,
    #     #         sae_pool=sae_pool,
    #     #         block_num=block_num,
    #     #         dataset_ver=dataset_ver_2,
    #     #         metric=metric
    #     #     )

    #     dataset_final = "v4"
    #     pooling_metric = "mean"
    #     # sae_analysis(
    #     #     sae_release_series=sae_release_series,
    #     #     sae_pool=sae_pool,
    #     #     block_num=block_num,
    #     #     dataset_ver=dataset_final,
    #     #     metric=pooling_metric,
    #     # )

    #     linear_regression_sae(
    #         #csv_file="ce_bench/ablation_study/ABLATION_MEAN_SAE_ANALYSIS_METRICS.csv",
    #         #csv_file="ce_bench/ablation_study/ABLATION_OUTLIER_SAE_ANALYSIS_METRICS.csv",
    #         csv_file="ce_bench/data_processing/SAE_ANALYSIS_METRICS.csv",
    #         trained_scaler=trained_scaler,
    #         trained_model=trained_model
    #     )



    #     # plot_v2_vs_v3_scores(v2_avg, v3_avg, sae_pool)
    # else:
    #     raise ValueError(f"Unknown task name: {args.task_name}")



if __name__ == "__main__":
    main()