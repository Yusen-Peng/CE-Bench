import os
import json
import matplotlib.pyplot as plt
from typing import Dict, List
import csv
import re


def depth_analysis(
    base_dir: str,
    dataset_ver: str,
    width: str,
    metric: str,
    output_csv: str = "ce_bench/depth_scores.csv"
):
    """
    Analyzes SAE scores for jump_relu variants across different layers (depths).

    Args:
        base_dir: Path under interpretability_eval/ containing SAE folders.
        dataset_ver: Dataset version (unused, for compatibility).
        width: Width value (e.g., "4096").
        metric: Metric to extract from results.json (e.g., "mean").
        output_csv: File to write CSV output.
    """
    eval_root = os.path.expanduser(f"interpretability_eval/{base_dir}")
    layer_pattern = re.compile(r"layer_(\d+)")
    rows = []

    for folder in os.listdir(eval_root):
        layer_match = layer_pattern.search(folder)
        if not layer_match:
            continue

        layer_idx = int(layer_match.group(1))
        layer_path = os.path.join(eval_root, folder, f"width_{width}")
        if not os.path.isdir(layer_path):
            continue

        for subfolder in os.listdir(layer_path):
            results_path = os.path.join(layer_path, subfolder, "results.json")
            if not os.path.exists(results_path):
                print(f"Missing: {results_path}")
                continue

            try:
                with open(results_path, "r") as f:
                    results = json.load(f)
                    contrastive = results.get("contrastive_score_mean", {}).get(metric)
                    independence = results.get("independent_score_mean", {}).get(metric)
                    interpretability = results.get("interpretability_score_mean", {}).get(metric)

                    if contrastive is not None and independence is not None:
                        rows.append([layer_idx, folder, subfolder, contrastive, independence, interpretability])
                    else:
                        print(f"Incomplete scores in {results_path}")
            except Exception as e:
                print(f"Error reading {results_path}: {e}")

    # Sort by layer depth
    rows.sort(key=lambda x: x[0])

    # Write to CSV
    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["layer", "folder_name", "subfolder", "contrastive_score", "independent_score", "joint_score"])
        writer.writerows(rows)

    print(f"Wrote JumpReLU depth analysis results to {output_csv}")

def width_analysis(
    base_dir: str,
    dataset_ver: str,
    metric: str,
    output_csv: str = "ce_bench/width_scores.csv"
):
    """
    Analyzes SAE scores for jump_relu variants across different widths.

    Args:
        base_dir: Path under interpretability_eval/ containing SAE folders.
        dataset_ver: Dataset version (unused, for compatibility).
        metric: Metric to extract from results.json (e.g., "mean").
        output_csv: File to write CSV output.
    """
    eval_root = os.path.expanduser(f"interpretability_eval/{base_dir}")
    if not os.path.exists(eval_root):
        print(f"Base directory does not exist: {eval_root}")
        return
    import re
    width_pattern = re.compile(r"jump_relu_width-2pow(\d+)")
    rows = []

    for folder in os.listdir(eval_root):
        if "jump_relu" not in folder:
            continue

        match = width_pattern.search(folder)
        if not match:
            continue

        width_exp = int(match.group(1))
        width = 2 ** width_exp
        folder_path = os.path.join(eval_root, folder)

        if not os.path.isdir(folder_path):
            continue

        for subfolder in os.listdir(folder_path):
            results_path = os.path.join(folder_path, subfolder, "results.json")
            if not os.path.exists(results_path):
                print(f"Missing: {results_path}")
                continue

            try:
                with open(results_path, "r") as f:
                    results = json.load(f)
                    contrastive = results.get("contrastive_score_mean", {}).get(metric)
                    independence = results.get("independent_score_mean", {}).get(metric)
                    interpretability = results.get("interpretability_score_mean", {}).get(metric)

                    if contrastive is not None and independence is not None:
                        rows.append([width, folder, subfolder, contrastive, independence, interpretability])
                    else:
                        print(f"Incomplete scores in {results_path}")
            except Exception as e:
                print(f"Error reading {results_path}: {e}")

    # Write to CSV
    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["width", "folder_name", "subfolder", "contrastive_score", "independent_score", "interpretability_score"])
        writer.writerows(rows)

    print(f"Wrote JumpReLU width analysis results to {output_csv}")




def sae_analysis(
    sae_release_series: str,
    sae_pool: List[str],
    block_num: int,
    dataset_ver: str,
    metric: str,
    output_csv: str = "ce_bench/sae_scores.csv"
):
    block_prefix = f"blocks.{block_num}.hook_resid_post"
    rows = []

    for sae_variant in sae_pool:
        sae_release = f"{sae_release_series}{sae_variant}"
        base_path = os.path.expanduser(f"interpretability_eval/{sae_release}")

        if not os.path.exists(base_path):
            print(f"Warning: {base_path} does not exist. Skipping.")
            continue

        for subfolder in os.listdir(base_path):
            results_path = os.path.join(base_path, subfolder, "results.json")
            if not os.path.exists(results_path):
                print(f"Missing: {results_path}")
                continue

            try:
                with open(results_path, "r") as f:
                    results = json.load(f)
                    contrastive = results.get("contrastive_score_mean", {}).get(metric)
                    independence = results.get("independent_score_mean", {}).get(metric)
                    interpretability = results.get("interpretability_score_mean", {}).get(metric)

                    if contrastive is not None and independence is not None:
                        rows.append([sae_variant, subfolder, contrastive, independence, interpretability])
                    else:
                        print(f"Incomplete scores in {results_path}")
            except Exception as e:
                print(f"Error reading {results_path}: {e}")

            except Exception as e:
                print(f"Error reading ce_bench/saes_metrics.json: {e}")

    # Write to CSV
    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["sae_variant", "subfolder", "contrastive_score", "independent_score", "interpretability_score"])
        writer.writerows(rows)

    print(f"Wrote results to {output_csv}")


if __name__ == "__main__":
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
    

    dataset_final = "v4"
    pooling_metric = "max"
    # sae_analysis(
    #     sae_release_series=sae_release_series,
    #     sae_pool=sae_pool,
    #     block_num=block_num,
    #     dataset_ver=dataset_final,
    #     metric=pooling_metric,
    # )
    
    # width_analysis(
    #     base_dir="",
    #     dataset_ver=dataset_final,
    #     metric=pooling_metric,
    # )

    depth_analysis(
        base_dir="gemma-scope-2b-pt-res",
        dataset_ver=dataset_final,
        width="16k",
        metric=pooling_metric,
    )