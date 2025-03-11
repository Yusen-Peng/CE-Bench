import os
import torch
from tqdm import tqdm
import pandas as pd
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
import plotly.express as px
import json
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
import plotly_express as px

from transformer_lens import HookedTransformer

# Model Loading
from sae_lens.analysis.neuronpedia_integration import get_neuronpedia_quick_list

# Enrichment Analysis Functions
from tsea import (
    get_enrichment_df,
    manhattan_plot_enrichment_scores,
    plot_top_k_feature_projections_by_token_and_category,
)
from tsea import (
    get_baby_name_sets,
    get_letter_gene_sets,
    generate_pos_sets,
    get_test_gene_sets,
    get_gene_set_from_regex,
)


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
    print(W_U_stats_df_dec)
    # Let's look at the distribution of the 3rd / 4th moments. I found these aren't as useful on their own as joint distributions can be.
    px.histogram(
        W_U_stats_df_dec,
        x="skewness",
        width=800,
        height=300,
        nbins=1000,
        title="Skewness of the Logit Weight Distributions",
    ).show()

    px.histogram(
        W_U_stats_df_dec,
        x=np.log10(W_U_stats_df_dec["kurtosis"]),
        width=800,
        height=300,
        nbins=1000,
        title="Kurtosis of the Logit Weight Distributions",
    ).show()

    fig = px.scatter(
        W_U_stats_df_dec,
        x="skewness",
        y="kurtosis",
        color="std",
        color_continuous_scale="Portland",
        hover_name="feature",
        width=800,
        height=500,
        log_y=True,  # Kurtosis has larger outliers so logging creates a nicer scale.
        labels={"x": "Skewness", "y": "Kurtosis", "color": "Standard Deviation"},
        title=f"Layer {8}: Skewness vs Kurtosis of the Logit Weight Distributions",
    )

    # decrease point size
    fig.update_traces(marker=dict(size=3))
    fig.show()


    # then you can query accross combinations of the statistics to find features of interest and open them in neuronpedia.
    tmp_df = W_U_stats_df_dec[["feature", "skewness", "kurtosis", "std"]]
    # tmp_df = tmp_df[(tmp_df["std"] > 0.04)]
    # tmp_df = tmp_df[(tmp_df["skewness"] > 0.65)]
    tmp_df = tmp_df[(tmp_df["skewness"] > 3)]
    tmp_df = tmp_df.sort_values("skewness", ascending=False).head(50)
    print(tmp_df)

    # if desired, open the features in neuronpedia
    get_neuronpedia_quick_list(sparse_autoencoder, list(tmp_df.feature))


    # """
    #     Token Set Enrichment Analysis.

    # """

    # import nltk

    # nltk.download()
    # # get the vocab we need to filter to formulate token sets.
    # vocab = model.tokenizer.get_vocab()  # type: ignore

    # # make a regex dictionary to specify more sets.
    # regex_dict = {
    #     "starts_with_space": r"Ġ.*",
    #     "starts_with_capital": r"^Ġ*[A-Z].*",
    #     "starts_with_lower": r"^Ġ*[a-z].*",
    #     "all_digits": r"^Ġ*\d+$",
    #     "is_punctuation": r"^[^\w\s]+$",
    #     "contains_close_bracket": r".*\).*",
    #     "contains_open_bracket": r".*\(.*",
    #     "all_caps": r"Ġ*[A-Z]+$",
    #     "1 digit": r"Ġ*\d{1}$",
    #     "2 digits": r"Ġ*\d{2}$",
    #     "3 digits": r"Ġ*\d{3}$",
    #     "4 digits": r"Ġ*\d{4}$",
    #     "length_1": r"^Ġ*\w{1}$",
    #     "length_2": r"^Ġ*\w{2}$",
    #     "length_3": r"^Ġ*\w{3}$",
    #     "length_4": r"^Ġ*\w{4}$",
    #     "length_5": r"^Ġ*\w{5}$",
    # }

    # # print size of gene sets
    # all_token_sets = get_letter_gene_sets(vocab)
    # for key, value in regex_dict.items():
    #     gene_set = get_gene_set_from_regex(vocab, value)
    #     all_token_sets[key] = gene_set

    # # some other sets that can be interesting
    # baby_name_sets = get_baby_name_sets(vocab)
    # pos_sets = generate_pos_sets(vocab)
    # arbitrary_sets = get_test_gene_sets(model)

    # all_token_sets = {**all_token_sets, **pos_sets}
    # all_token_sets = {**all_token_sets, **arbitrary_sets}
    # all_token_sets = {**all_token_sets, **baby_name_sets}

    # # for each gene set, convert to string and  print the first 5 tokens
    # for token_set_name, gene_set in sorted(
    #     all_token_sets.items(), key=lambda x: len(x[1]), reverse=True
    # ):
    #     tokens = [model.to_string(id) for id in list(gene_set)][:10]  # type: ignore
    #     print(f"{token_set_name}, has {len(gene_set)} genes")
    #     print(tokens)
    #     print("----")

    # features_ordered_by_skew = (
    #     W_U_stats_df_dec["skewness"].sort_values(ascending=False).head(5000).index.to_list()
    # )
    # # filter our list.
    # token_sets_index = [
    #     "starts_with_space",
    #     "starts_with_capital",
    #     "all_digits",
    #     "is_punctuation",
    #     "all_caps",
    # ]
    # token_set_selected = {
    #     k: set(v) for k, v in all_token_sets.items() if k in token_sets_index
    # }

    # # calculate the enrichment scores
    # df_enrichment_scores = get_enrichment_df(
    #     dec_projection_onto_W_U,  # use the logit weight values as our rankings over tokens.
    #     features_ordered_by_skew,  # subset by these features
    #     token_set_selected,  # use token_sets
    # )

    # manhattan_plot_enrichment_scores(
    #     df_enrichment_scores,
    #     label_threshold=0,
    #     top_n=3,  # use our enrichment scores
    # ).show()
    # fig = px.scatter(
    #     df_enrichment_scores.apply(lambda x: -1 * np.log(1 - x)).T,
    #     x="starts_with_space",
    #     y="starts_with_capital",
    #     marginal_x="histogram",
    #     marginal_y="histogram",
    #     labels={
    #         "starts_with_space": "Starts with Space",
    #         "starts_with_capital": "Starts with Capital",
    #     },
    #     title="Enrichment Scores for Starts with Space vs Starts with Capital",
    #     height=800,
    #     width=800,
    # )
    # # reduce point size on the scatter only
    # fig.update_traces(marker=dict(size=2), selector=dict(mode="markers"))
    # fig.show()
    # token_sets_index = ["1 digit", "2 digits", "3 digits", "4 digits"]
    # token_set_selected = {
    #     k: set(v) for k, v in all_token_sets.items() if k in token_sets_index
    # }
    # df_enrichment_scores = get_enrichment_df(
    #     dec_projection_onto_W_U, features_ordered_by_skew, token_set_selected
    # )
    # manhattan_plot_enrichment_scores(df_enrichment_scores).show()
    # token_sets_index = ["nltk_pos_PRP", "nltk_pos_VBZ", "nltk_pos_NNP"]
    # token_set_selected = {
    #     k: set(v) for k, v in all_token_sets.items() if k in token_sets_index
    # }
    # df_enrichment_scores = get_enrichment_df(
    #     dec_projection_onto_W_U, features_ordered_by_skew, token_set_selected
    # )
    # manhattan_plot_enrichment_scores(df_enrichment_scores).show()
    # token_sets_index = ["nltk_pos_VBN", "nltk_pos_VBG", "nltk_pos_VB", "nltk_pos_VBD"]
    # token_set_selected = {
    #     k: set(v) for k, v in all_token_sets.items() if k in token_sets_index
    # }
    # df_enrichment_scores = get_enrichment_df(
    #     dec_projection_onto_W_U, features_ordered_by_skew, token_set_selected
    # )
    # manhattan_plot_enrichment_scores(df_enrichment_scores).show()
    # token_sets_index = ["nltk_pos_WP", "nltk_pos_RBR", "nltk_pos_WDT", "nltk_pos_RB"]
    # token_set_selected = {
    #     k: set(v) for k, v in all_token_sets.items() if k in token_sets_index
    # }
    # df_enrichment_scores = get_enrichment_df(
    #     dec_projection_onto_W_U, features_ordered_by_skew, token_set_selected
    # )
    # manhattan_plot_enrichment_scores(df_enrichment_scores).show()
    # token_sets_index = ["a", "e", "i", "o", "u"]
    # token_set_selected = {
    #     k: set(v) for k, v in all_token_sets.items() if k in token_sets_index
    # }
    # df_enrichment_scores = get_enrichment_df(
    #     dec_projection_onto_W_U, features_ordered_by_skew, token_set_selected
    # )
    # manhattan_plot_enrichment_scores(df_enrichment_scores).show()
    # token_sets_index = ["negative_words", "positive_words"]
    # token_set_selected = {
    #     k: set(v) for k, v in all_token_sets.items() if k in token_sets_index
    # }
    # df_enrichment_scores = get_enrichment_df(
    #     dec_projection_onto_W_U, features_ordered_by_skew, token_set_selected
    # )
    # manhattan_plot_enrichment_scores(df_enrichment_scores).show()
    # fig = px.scatter(
    #     df_enrichment_scores.apply(lambda x: -1 * np.log(1 - x))
    #     .T.reset_index()
    #     .rename(columns={"index": "feature"}),
    #     x="negative_words",
    #     y="positive_words",
    #     marginal_x="histogram",
    #     marginal_y="histogram",
    #     labels={
    #         "starts_with_space": "Starts with Space",
    #         "starts_with_capital": "Starts with Capital",
    #     },
    #     title="Enrichment Scores for Starts with Space vs Starts with Capital",
    #     height=800,
    #     width=800,
    #     hover_name="feature",
    # )
    # # reduce point size on the scatter only
    # fig.update_traces(marker=dict(size=2), selector=dict(mode="markers"))
    # fig.show()
    # token_sets_index = ["contains_close_bracket", "contains_open_bracket"]
    # token_set_selected = {
    #     k: set(v) for k, v in all_token_sets.items() if k in token_sets_index
    # }
    # df_enrichment_scores = get_enrichment_df(
    #     dec_projection_onto_W_U, features_ordered_by_skew, token_set_selected
    # )
    # manhattan_plot_enrichment_scores(df_enrichment_scores).show()
    # token_sets_index = [
    #     "1910's",
    #     "1920's",
    #     "1930's",
    #     "1940's",
    #     "1950's",
    #     "1960's",
    #     "1970's",
    #     "1980's",
    #     "1990's",
    #     "2000's",
    #     "2010's",
    # ]
    # token_set_selected = {
    #     k: set(v) for k, v in all_token_sets.items() if k in token_sets_index
    # }
    # df_enrichment_scores = get_enrichment_df(
    #     dec_projection_onto_W_U, features_ordered_by_skew, token_set_selected
    # )
    # manhattan_plot_enrichment_scores(df_enrichment_scores).show()
    # token_sets_index = ["positive_words", "negative_words"]
    # token_set_selected = {
    #     k: set(v) for k, v in all_token_sets.items() if k in token_sets_index
    # }
    # df_enrichment_scores = get_enrichment_df(
    #     dec_projection_onto_W_U, features_ordered_by_skew, token_set_selected
    # )
    # manhattan_plot_enrichment_scores(df_enrichment_scores, label_threshold=0.98).show()
    # token_sets_index = ["boys_names", "girls_names"]
    # token_set_selected = {
    #     k: set(v) for k, v in all_token_sets.items() if k in token_sets_index
    # }
    # df_enrichment_scores = get_enrichment_df(
    #     dec_projection_onto_W_U, features_ordered_by_skew, token_set_selected
    # )
    # manhattan_plot_enrichment_scores(df_enrichment_scores).show()
    # tmp_df = df_enrichment_scores.apply(lambda x: -1 * np.log(1 - x)).T
    # color = (
    #     W_U_stats_df_dec.sort_values("skewness", ascending=False)
    #     .head(5000)["skewness"]
    #     .values
    # )
    # fig = px.scatter(
    #     tmp_df.reset_index().rename(columns={"index": "feature"}),
    #     x="boys_names",
    #     y="girls_names",
    #     marginal_x="histogram",
    #     marginal_y="histogram",
    #     # color = color,
    #     labels={
    #         "boys_names": "Enrichment Score (Boys Names)",
    #         "girls_names": "Enrichment Score (Girls Names)",
    #     },
    #     height=600,
    #     width=800,
    #     hover_name="feature",
    # )
    # # reduce point size on the scatter only
    # fig.update_traces(marker=dict(size=3), selector=dict(mode="markers"))
    # # annotate any features where the absolute distance between boys names and girls names > 3
    # for feature in df_enrichment_scores.columns:
    #     if abs(tmp_df["boys_names"][feature] - tmp_df["girls_names"][feature]) > 2.9:
    #         fig.add_annotation(
    #             x=tmp_df["boys_names"][feature] - 0.4,
    #             y=tmp_df["girls_names"][feature] + 0.1,
    #             text=f"{feature}",
    #             showarrow=False,
    #         )

    # fig.show()







if __name__ == '__main__':
    main()