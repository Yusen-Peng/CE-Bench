import os
import torch
from tqdm import tqdm
import pandas as pd
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from sae_lens import SAE, HookedSAETransformer
import json
import safetensors.torch

@torch.no_grad()
def get_feature_property_df(sae: SAE, feature_sparsity: torch.Tensor):
    """
    feature_property_df = get_feature_property_df(sae, log_feature_density.cpu())
    """
    W_dec_normalized = sae.W_dec.cpu()
    W_enc_normalized = (sae.W_enc.cpu() / sae.W_enc.cpu().norm(dim=-1, keepdim=True)).T
    d_e_projection = (W_dec_normalized * W_enc_normalized).sum(-1)
    b_dec_projection = sae.b_dec.cpu() @ W_dec_normalized.T
    return pd.DataFrame(
        {
            "log_feature_sparsity": feature_sparsity + 1e-10,
            "d_e_projection": d_e_projection,
            "b_enc": sae.b_enc.detach().cpu(),
            "b_dec_projection": b_dec_projection,
            "feature": list(range(sae.cfg.d_sae)),
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

    # Convert tensors to float32 before moving to NumPy
    return pd.DataFrame(
        {
            "feature": range(len(skews)),
            "mean": mean.cpu().to(torch.float).numpy().squeeze(),
            "std": std.cpu().to(torch.float).numpy().squeeze(),
            "skewness": skews.cpu().to(torch.float).numpy(),
            "kurtosis": kurtosis.cpu().to(torch.float).numpy(),
        }
    )

@torch.no_grad()
def get_W_U_W_dec_stats_df(W_dec: torch.Tensor, model: HookedTransformer, cosine_sim: bool = False) -> tuple[pd.DataFrame, torch.Tensor]:
    W_U = model.W_U.detach()

    # Ensure both tensors have the same dtype
    dtype = W_dec.dtype  # Preserve W_dec's dtype
    W_U = W_U.to(dtype)  # Convert W_U to the same dtype as W_dec

    if cosine_sim:
        W_U = W_U / W_U.norm(dim=0, keepdim=True)
    dec_projection_onto_W_U = W_dec @ W_U
    W_U_stats_df = get_stats_df(dec_projection_onto_W_U)
    return W_U_stats_df, dec_projection_onto_W_U

def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load LLaMA 3.2 tokenizer
    model_name = "meta-llama/Llama-3.2-1B"  # Adjust to your specific LLaMA variant
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print("tokenizer loaded!")

    # Load the trained SAE from checkpoints
    architecture = "jumprelu"
    sae_checkpoint_path = f"checkpoints/{architecture}/final_1000448"
    sae = SAE.load_from_pretrained(path=sae_checkpoint_path, device=device)
    print("SAE loaded!")

    # Load weights
    weights_path = os.path.join(sae_checkpoint_path, "sae_weights.safetensors")
    sae_weights = safetensors.torch.load_file(weights_path)
    sae.load_state_dict(sae_weights)
    print("Weights loaded!")
    
    # Load sparsity
    sparsity_path = os.path.join(sae_checkpoint_path, "sparsity.safetensors")
    sparsity = safetensors.torch.load_file(sparsity_path)["sparsity"]
    print("Sparsity loaded!")
    
    sae.to(device)
    sae.eval()

    # Load model using HookedSAETransformer with SAE's kwargs
    model = HookedSAETransformer.from_pretrained_no_processing(
        model_name,
        device=device,
        **sae.cfg.model_from_pretrained_kwargs
    )
    print("Model loaded!")

    # SAE was trained on layer 0
    layer = 0

    # Use the loaded SAE and sparsity
    sparse_autoencoder = sae
    log_feature_sparsity = sparsity.to(device)  # Keep on GPU
    W_dec = sparse_autoencoder.W_dec.to(device)  # Keep on GPU
    W_U_stats_df_dec, _ = get_W_U_W_dec_stats_df(W_dec, model, cosine_sim=False)
    W_U_stats_df_dec["sparsity"] = log_feature_sparsity.cpu().numpy()
    print("Statistics calculated!")


    # First histogram: Skewness
    plt.figure(figsize=(10, 4))
    plt.hist(W_U_stats_df_dec["skewness"], bins=1000)
    plt.title("Skewness of the Logit Weight Distributions")
    plt.xlabel("Skewness")
    plt.ylabel("Count")
    plt.savefig(f"figures/{architecture}_skewness_histogram.png")
    plt.close()

    # Second histogram: Kurtosis (log scale)
    plt.figure(figsize=(10, 4))
    plt.hist(np.log10(W_U_stats_df_dec["kurtosis"]), bins=1000)
    plt.title("Kurtosis of the Logit Weight Distributions")
    plt.xlabel("Log10(Kurtosis)")
    plt.ylabel("Count")
    plt.savefig(f"figures/{architecture}_kurtosis_histogram.png")
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
    plt.title(f"Layer {layer}: Skewness vs Kurtosis of the Logit Weight Distributions")
    plt.colorbar(scatter, label="Standard Deviation")
    plt.savefig(f"figures/{architecture}_skewness_vs_kurtosis_scatter.png")
    plt.close()

if __name__ == '__main__':
    main()