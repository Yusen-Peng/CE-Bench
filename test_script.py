import os
import torch
from tqdm import tqdm
import pandas as pd
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from sae_lens.sae import SAE


@torch.no_grad()
def get_feature_property_df(sae: SAE, feature_sparsity: torch.Tensor):
    """
    feature_property_df = get_feature_property_df(sae, log_feature_density.cpu())
    """

    W_dec_normalized = (
        sae.W_dec.cpu()
    )  # / sparse_autoencoder.W_dec.cpu().norm(dim=-1, keepdim=True)
    W_enc_normalized = (sae.W_enc.cpu() / sae.W_enc.cpu().norm(dim=-1, keepdim=True)).T

    d_e_projection = (W_dec_normalized * W_enc_normalized).sum(-1)
    b_dec_projection = sae.b_dec.cpu() @ W_dec_normalized.T

    return pd.DataFrame(
        {
            "log_feature_sparsity": feature_sparsity + 1e-10,
            "d_e_projection": d_e_projection,
            # "d_e_projection_normalized": d_e_projection_normalized,
            "b_enc": sae.b_enc.detach().cpu(),
            "b_dec_projection": b_dec_projection,
            "feature": list(range(sae.cfg.d_sae)),  # type: ignore
            "dead_neuron": (feature_sparsity < -9).cpu(),
        }
    )


@torch.no_grad()
def get_stats_df(projection: torch.Tensor):
    """
    Returns a dataframe with the mean, std, skewness and kurtosis of the projection
    """
    mean = projection.mean(dim=1, keepdim=True)
    diffs = projection - mean
    var = (diffs**2).mean(dim=1, keepdim=True)
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0), dim=1)
    kurtosis = torch.mean(torch.pow(zscores, 4.0), dim=1)

    return pd.DataFrame(
        {
            "feature": range(len(skews)),
            "mean": mean.numpy().squeeze(),
            "std": std.numpy().squeeze(),
            "skewness": skews.numpy(),
            "kurtosis": kurtosis.numpy(),
        }
    )


@torch.no_grad()
def get_all_stats_dfs(
    gpt2_small_sparse_autoencoders: dict[str, SAE],  # [hook_point, sae]
    gpt2_small_sae_sparsities: dict[str, torch.Tensor],  # [hook_point, sae]
    model: HookedTransformer,
    cosine_sim: bool = False,
):
    stats_dfs = []
    pbar = tqdm(gpt2_small_sparse_autoencoders.keys())
    for key in pbar:
        layer = int(key.split(".")[1])
        sparse_autoencoder = gpt2_small_sparse_autoencoders[key]
        pbar.set_description(f"Processing layer {sparse_autoencoder.cfg.hook_name}")
        W_U_stats_df_dec, _ = get_W_U_W_dec_stats_df(
            sparse_autoencoder.W_dec.cpu(), model, cosine_sim
        )
        log_feature_sparsity = gpt2_small_sae_sparsities[key].detach().cpu()
        W_U_stats_df_dec["log_feature_sparsity"] = log_feature_sparsity
        W_U_stats_df_dec["layer"] = layer + (1 if "post" in key else 0)
        stats_dfs.append(W_U_stats_df_dec)

    return pd.concat(stats_dfs, axis=0)


@torch.no_grad()
def get_W_U_W_dec_stats_df(
    W_dec: torch.Tensor, model: HookedTransformer, cosine_sim: bool = False
) -> tuple[pd.DataFrame, torch.Tensor]:
    W_U = model.W_U.detach().cpu()
    if cosine_sim:
        W_U = W_U / W_U.norm(dim=0, keepdim=True)
    dec_projection_onto_W_U = W_dec @ W_U
    W_U_stats_df = get_stats_df(dec_projection_onto_W_U)
    return W_U_stats_df, dec_projection_onto_W_U


import numpy as np
import torch

from transformer_lens import HookedTransformer

# Model Loading
# from sae_lens.analysis.neuronpedia_integration import get_neuronpedia_quick_list

# # Enrichment Analysis Functions
# from tsea import (
#     get_enrichment_df,
#     manhattan_plot_enrichment_scores,
#     plot_top_k_feature_projections_by_token_and_category,
# )
# from tsea import (
#     get_baby_name_sets,
#     get_letter_gene_sets,
#     generate_pos_sets,
#     get_test_gene_sets,
#     get_gene_set_from_regex,
# )


def main():
    model = HookedTransformer.from_pretrained("gpt2-small")
    # this is an outdated way to load the SAE. We need to have feature spartisity loadable through the new interface to remove it.
    gpt2_small_sparse_autoencoders = {}
    gpt2_small_sae_sparsities = {}

    for layer in range(12):
        sae, original_cfg_dict, sparsity = SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id=f"blocks.{layer}.hook_resid_pre",
            device="cpu",
        )
        gpt2_small_sparse_autoencoders[f"blocks.{layer}.hook_resid_pre"] = sae
        gpt2_small_sae_sparsities[f"blocks.{layer}.hook_resid_pre"] = sparsity
    
    
    """
        Statistical Properties of Feature Logit Distributions.

    """
    
    # In the post, I focus on layer 8
    layer = 8

    # get the corresponding SAE and feature sparsities.
    sparse_autoencoder = gpt2_small_sparse_autoencoders[f"blocks.{layer}.hook_resid_pre"]
    log_feature_sparsity = gpt2_small_sae_sparsities[f"blocks.{layer}.hook_resid_pre"].cpu()

    W_dec = sparse_autoencoder.W_dec.detach().cpu()

    # calculate the statistics of the logit weight distributions
    W_U_stats_df_dec, dec_projection_onto_W_U = get_W_U_W_dec_stats_df(
        W_dec, model, cosine_sim=False
    )
    W_U_stats_df_dec["sparsity"] = (
        log_feature_sparsity  # add feature sparsity since it is often interesting.
    )

    # First histogram: Skewness
    plt.figure(figsize=(10, 4))
    plt.hist(W_U_stats_df_dec["skewness"], bins=1000)
    plt.title("Skewness of the Logit Weight Distributions")
    plt.xlabel("Skewness")
    plt.ylabel("Count")
    plt.savefig("skewness_histogram.png")
    plt.close()

    # Second histogram: Kurtosis (log scale)
    plt.figure(figsize=(10, 4))
    plt.hist(np.log10(W_U_stats_df_dec["kurtosis"]), bins=1000)
    plt.title("Kurtosis of the Logit Weight Distributions")
    plt.xlabel("Log10(Kurtosis)")
    plt.ylabel("Count")
    plt.savefig("kurtosis_histogram.png")
    plt.close()

    # Scatter plot: Skewness vs Kurtosis
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        W_U_stats_df_dec["skewness"],
        W_U_stats_df_dec["kurtosis"],
        c=W_U_stats_df_dec["std"],
        cmap="viridis",
        s=3  # point size
    )
    plt.yscale('log')  # Log scale for y-axis
    plt.xlabel("Skewness")
    plt.ylabel("Kurtosis")
    plt.title(f"Layer {8}: Skewness vs Kurtosis of the Logit Weight Distributions")
    plt.colorbar(scatter, label="Standard Deviation")
    plt.savefig("skewness_vs_kurtosis_scatter.png")
    plt.close()



    # # then you can query accross combinations of the statistics to find features of interest and open them in neuronpedia.
    # tmp_df = W_U_stats_df_dec[["feature", "skewness", "kurtosis", "std"]]
    # # tmp_df = tmp_df[(tmp_df["std"] > 0.04)]
    # # tmp_df = tmp_df[(tmp_df["skewness"] > 0.65)]
    # tmp_df = tmp_df[(tmp_df["skewness"] > 3)]
    # tmp_df = tmp_df.sort_values("skewness", ascending=False).head(50)
    # print(tmp_df)

    # # if desired, open the features in neuronpedia
    # get_neuronpedia_quick_list(sparse_autoencoder, list(tmp_df.feature))

if __name__ == '__main__':
    main()