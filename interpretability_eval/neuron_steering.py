import pandas as pd
from functools import partial
import torch
from transformers import GPT2Tokenizer, AutoTokenizer
from sae_lens import SAE, HookedSAETransformer

def fire_multiple_neurons(activation, hook, sae: SAE, neuron_list, scale):
    # 1. Encode to latent space
    latents = sae.encode(activation)  # shape [batch, seq_len, sae_latent_dim]

    # 2. Fire up the chosen neurons
    # latents[..., neuron_list] modifies the last dimension
    latents[..., neuron_list] *= scale

    # 3. Decode back to the original hidden dimension
    new_activation = sae.decode(latents)

    return new_activation
    

def autoregressive_generate(model, tokenizer, prompt, max_new_tokens=50, device="cuda"):
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
    
    # Load LLaMA tokenizer
    model_name = "meta-llama/Llama-3.2-1B"
    #model_name = "gpt2-small"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded!")

    # Load the trained SAE
    architecture = "LLAMA_cache_only_kan"
    steps = "1k"
    best_model = "best_2457600_ce_2.13012_ori_2.03857" # llama only kan
    sae_checkpoint_path = f"checkpoints/{architecture}/{steps}/{best_model}/"
    sae = SAE.load_from_pretrained(path=sae_checkpoint_path, device=device)
    print("SAE loaded!")

    # steer one particular set of neurons with the common subject from the csv file
    target_subject = "temperature"
    # Load the CSV file
    csv_file_path = f"interpretability_eval/{architecture}_responsible_neurons.csv"
    df = pd.read_csv(csv_file_path)

    # filter the DataFrame for the specific subject
    subject_neurons = df[df["subject"] == target_subject].iloc[:, 0].to_list()

    print(f"Steering neurons for subject: {target_subject}")
    #print(f"Neuron indices: {subject_neurons}")

    model = HookedSAETransformer.from_pretrained_no_processing(
        model_name,
        device=device,
        **sae.cfg.model_from_pretrained_kwargs
    )
    print("Model loaded!")


    prompt = "Once upon a time, "
    baseline_output = autoregressive_generate(model, tokenizer, prompt, device=device)
    print("\nBaseline generation:", baseline_output)

    SCALE = 50.0
    model.add_hook(
        "blocks.5.hook_mlp_out",
        partial(fire_multiple_neurons, sae=sae, neuron_list=subject_neurons, scale=SCALE),
        "fwd"
    )
    steered_output = autoregressive_generate(model, tokenizer, prompt, device=device)
    print("\nSteered generation:", steered_output)


if __name__ == "__main__":
    main()