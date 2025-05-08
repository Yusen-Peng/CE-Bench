import pandas as pd
from functools import partial
import torch
from transformers import GPT2Tokenizer, AutoTokenizer
from sae_lens import SAE, HookedSAETransformer
import os
import json
import sae_bench.sae_bench_utils.general_utils as general_utils

def fire_multiple_sae_neurons(activation, hook, sae: SAE, neuron_list, scale):
    # 1. Encode to latent space
    latents = sae.encode(activation)  # shape [batch, seq_len, sae_latent_dim]

    # 2. Fire up the chosen neurons
    # latents[..., neuron_list] modifies the last dimension
    latents[..., neuron_list] += scale

    # 3. Decode back to the original hidden dimension
    new_activation = sae.decode(latents)

    # new_activation = activation.clone()
    return new_activation

def shift_v_neurons_apply(activation, hook, sae: SAE, vector, scale):
    # 1. Encode to latent space
    latents = sae.encode(activation)  # shape [batch, seq_len, sae_latent_dim]

    # 2. Fire up the chosen neurons
    # latents[..., neuron_list] modifies the last dimension
    new_latents = latents + vector * scale

    # 3. Decode back to the original hidden dimension
    new_activation = sae.decode(new_latents)

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
    # NOTE: CHANGE THESE GUYS EVERY TIME
    target_subject = "knowledge"
    experiment_name = f"gemma-scope-2b-pt-res/layer_12/width_16k/average_l0_22"
    metric = "interpretability"
    df = pd.read_csv(f"interpretability_eval/{experiment_name}/{metric}_scores_per_subject.csv")
    top_neurons = 100
    SCALE = 0.5
    #prompt = "Shrek (Mike Myers) leads a solitary life in his swamp. A group of peasants gathers in front of his house with the goal of hunting down"
    prompt = "How did I drop my GPA? I am a student at a university and this semester, "

    # SAE baseline
    # She is a very kind person, and she has a strong sense of responsibility. She is a very good person. She is a very good person. She is

    # top 5 neurons, scale = 10, contrastive metric

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Experiment name: {experiment_name}")
    print("=" * 80)
    results_json = json.load(open(f"interpretability_eval/{experiment_name}/results.json"))
    shift_v = pd.read_csv(f"interpretability_eval/{experiment_name}/shift_v_per_subject_mean.csv")
    print(f"Results JSON loaded from {experiment_name}/results.json")
    print(f"Experiment name: {experiment_name}")

    # Load LLaMA tokenizer
    # model_name = "meta-llama/Llama-3.2-1B"
    #model_name = "gpt2-small"
    model_name = results_json["sae_config"]["model_name"]
    print(model_name)

    sae_release = results_json["sae_release"]
    sae_id = results_json["sae_id"]
    # sae = SAE.load_from_pretrained(path=sae_checkpoint_path, device=device)
    sae_id, sae, sparsity = general_utils.load_and_format_sae(
        sae_release, sae_id, device
    )  # type: ignore
    print("SAE loaded!")

    print(f"Target subject: {target_subject}")
    print("=" * 80)

    # print(df.columns)
    # FIXME: alternative: take the neuron with the highest activation instead all of them in subject_neurons group
    subject_neurons = df.sort_values(by=target_subject, ascending=False).head(top_neurons)

    best_neuron_row_ids = subject_neurons["Unnamed: 0"].tolist()

    print(f"Best neuron row ids for subject '{target_subject}': {best_neuron_row_ids}")
    subject_neurons = [best_neuron_row_ids]
    
    model = HookedSAETransformer.from_pretrained_no_processing(
        model_name,
        device=device,
        **sae.cfg.model_from_pretrained_kwargs
    )
    print("Model loaded!")

    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded!")

    baseline_output = autoregressive_generate(model, tokenizer, prompt, device=device, max_new_tokens=32)
    print("\nBaseline generation:", baseline_output)

    # model.add_hook(
    #     sae.cfg.hook_name,
    #     partial(fire_multiple_sae_neurons, sae=sae, neuron_list=subject_neurons, scale=SCALE),
    #     "fwd"
    # )

    shift_v_neurons = torch.tensor(shift_v[target_subject].tolist()).to(device)
    model.add_hook(
        sae.cfg.hook_name,
        partial(shift_v_neurons_apply, sae=sae, vector=shift_v_neurons, scale=SCALE),
        "fwd"
    )

    steered_output = autoregressive_generate(model, tokenizer, prompt, device=device, max_new_tokens=32)
    print("\nSteered generation:", steered_output)


if __name__ == "__main__":
    main()