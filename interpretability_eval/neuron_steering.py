import pandas as pd
from functools import partial
import torch
from transformers import GPT2Tokenizer, AutoTokenizer
from sae_lens import SAE, HookedSAETransformer
import os
import json

def fire_multiple_sae_neurons(activation, hook, sae: SAE, neuron_list, scale):
    # 1. Encode to latent space
    latents = sae.encode(activation)  # shape [batch, seq_len, sae_latent_dim]

    # 2. Fire up the chosen neurons
    # latents[..., neuron_list] modifies the last dimension
    latents[..., neuron_list] *= scale

    # 3. Decode back to the original hidden dimension
    new_activation = sae.decode(latents)

    # new_activation = activation.clone()
    return new_activation

def fire_multiple_neurons(activation, hook, neuron_list, scale):
    activation[..., neuron_list] *= scale
    return activation
    

def autoregressive_generate(model, tokenizer, prompt, max_new_tokens=20, device="cuda"):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids, return_type="logits")
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_token], dim=1)

    # Decode the entire sequence of tokens
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # NOTE: CHANGE IT EVERY TIME
    experiment_name = "LLAMA_jumprelu_crop"
    print(f"Experiment name: {experiment_name}")
    print("=" * 80)
    results_json = json.load(open(f"interpretability_eval/{experiment_name}/results.json"))
    print(f"Results JSON loaded from {experiment_name}/results.json")
    print(f"Experiment name: {experiment_name}")

    # Load LLaMA tokenizer
    # model_name = "meta-llama/Llama-3.2-1B"
    #model_name = "gpt2-small"
    model_name = results_json["model_name"]
    print(model_name)
    
    if "gpt2" in model_name:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        print("Using GPT2 tokenizer")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Using AutoTokenizer")

    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded!")

    # Load the trained SAE
    # architecture = "LLAMA_cache_only_kan"
    #architecture = "LLAMA_cache_kan_relu_dense"
    #architecture = "LLAMA_cache_gated"
    #architecture = "LLAMA_cache_jumprelu"

    #architecture = "GPT_cache_only_kan"
    #architecture = "GPT_cache_kan_relu_dense"
    #architecture = "GPT_cache_gated"
    #architecture = "GPT_cache_jumprelu"

    architecture = results_json["architecture"]

    steps = "1k"
    steps = results_json["steps"]
    

    # best_model = "best_2457600_ce_2.13012_ori_2.03857" # llama only kan
    #best_model = "best_2457600_ce_2.09549_ori_2.03857" # llama kan relu dense
    #best_model = "best_2457600_ce_2.24055_ori_2.03857" # llama gated
    #best_model = "best_2457600_ce_2.23809_ori_2.03857" # llama jumprelu


    #best_model = "best_3686400_ce_2.35626_ori_2.33838" # gpt only kan
    #best_model = "best_3686400_ce_2.34855_ori_2.33838" # gpt kan relu dense
    #best_model = "best_3686400_ce_2.39366_ori_2.33838" # gpt gated
    #best_model = "best_3686400_ce_2.37705_ori_2.33838" # gpt jumprelu

    best_model = results_json["best_model"]


    sae_checkpoint_path = f"checkpoints/{architecture}/{steps}/{best_model}/"
    sae = SAE.load_from_pretrained(path=sae_checkpoint_path, device=device)
    print("SAE loaded!")

    # steer one particular set of neurons with the common subject from the csv file
    target_subject = "sense of justice"
    print(f"Target subject: {target_subject}")
    print("=" * 80)
    # Load the CSV file
    # csv_file_path = f"interpretability_eval/{experiment_name}/responsible_neurons.csv"
    # df = pd.read_csv(csv_file_path)
    # df = pd.DataFrame.from_dict(results_json["responsible_neurons"], orient='index', columns=['interpretability_score', 'subject'])

    df = pd.read_csv(f"interpretability_eval/{experiment_name}/interpretability_scores_per_subject.csv")
    # print(df.columns)
    # FIXME: alternative: take the neuron with the highest activation instead all of them in subject_neurons group
    subject_neurons = df.sort_values(by=target_subject, ascending=False).head(10)

    # take top 10 neurons rows for their interpretability scores use sort + head
    # best_neuron_row = subject_neurons.sort_values(by="interpretability_score", ascending=False).head(50)
    # best_neuron_row_ids = best_neuron_row["Unnamed: 0"].tolist()

    best_neuron_row_ids = subject_neurons["Unnamed: 0"].tolist()

    print(f"Best neuron row ids for subject '{target_subject}': {best_neuron_row_ids}")
    subject_neurons = [best_neuron_row_ids]
    
    model = HookedSAETransformer.from_pretrained_no_processing(
        model_name,
        device=device,
        **sae.cfg.model_from_pretrained_kwargs
    )
    print("Model loaded!")

    prompt = "She was walking in a park when suddenly she saw a person running towards her."
    baseline_output = autoregressive_generate(model, tokenizer, prompt, device=device)
    print("\nBaseline generation:", baseline_output)

    SCALE = 500
    model.add_hook(
        "blocks.5.hook_mlp_out",
        partial(fire_multiple_sae_neurons, sae=sae, neuron_list=subject_neurons, scale=SCALE),
        "fwd"
    )
    # model.add_hook(
    #     "blocks.5.hook_mlp_out",
    #     partial(fire_multiple_neurons, neuron_list=subject_neurons, scale=SCALE),
    #     "fwd"
    # )

    steered_output = autoregressive_generate(model, tokenizer, prompt, device=device)
    print("\nSteered generation:", steered_output)


if __name__ == "__main__":
    main()