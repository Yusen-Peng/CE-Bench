import torch
import torch.nn as nn
import numpy as np
import openai
import os
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2Tokenizer
from sae_lens import SAE, HookedSAETransformer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel


def explain(tokens: list[str], activations: np.array, client: openai.OpenAI):
    """
        Generate an explanation for the feature based on the top tokens and their activations.
    """

    token_activations = [(token, round(act.item() * 100)) for token, act in zip(tokens, activations)]
    token_activations_str = "\n".join([f"{token}\t{act}" for token, act in token_activations])
    non_zero_activations = [(token, act) for token, act in token_activations if act > 0]
    non_zero_activations_str = "\n".join(f"{token}\t{act}" for token, act in non_zero_activations)
    # print(token_activations)
    prompt = f"""
                We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is looking for. Don't list examples of words.

                The activation format is token<tab>activation. Activation values range from 0 to 10. A neuron finding what it's looking for is represented by a non-zero activation value. The higher the activation value, the stronger the match.

                Activations:
                <start>
                {
                    token_activations_str
                }
                <end>

                Same activations, but with all zeros filtered out:
                <start>
                {
                    non_zero_activations_str
                }
                <end>

                Explanation of neuron behavior: 
            """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in interpreting neural network activations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )
    explanation = response.choices[0].message.content.strip()
    return explanation

class ActivationTokens(BaseModel):
    activations: list[int]


def simulate(example_tokens: list[str], explanation: str, client: openai.OpenAI):
    """
        Calls an LLM to produce a 'simulated' set of top tokens and normalized activations (0 to 100)
    """
    text_str = "\n".join(example_tokens)
    prompt = f"""
            We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at an explanation of what the neuron does, and try to predict its activations on a particular token.
            The activations range from 0 to 100. Most activations will be 0. Make sure you return EXACTLY {len(example_tokens)} activations.
            Explanation of neuron behavior: {explanation}
            <start>
            {text_str}
            <end>
            """

    # For demonstration, using openai.Completion. You can also use ChatCompletion.
    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You simulate neuron activations in language models."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format=ActivationTokens,
    )

    content = response.choices[0].message.parsed
    # print(content.activations, content.tokens)

    return np.array(content.activations) / 100.0


def main():
    # Load environment variables (API Key)
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    if not openai.api_key:
        raise ValueError("OpenAI API key is missing! Make sure it's set in the .env file.")

    # Initialize OpenAI Client
    client = openai.OpenAI()

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
    # architecture = "LLAMA_cache_kan_relu_dense"
    architecture = "LLAMA_cache_gated"
    steps = "1k"
    best_model = "best_2457600_ce_2.24055_ori_2.03857"
    sae_checkpoint_path = f"checkpoints/{architecture}/{steps}/{best_model}/"
    sae = SAE.load_from_pretrained(path=sae_checkpoint_path, device=device)
    print("SAE loaded!")

    # Load the model using HookedSAETransformer
    model = HookedSAETransformer.from_pretrained_no_processing(
        model_name,
        device=device,
        **sae.cfg.model_from_pretrained_kwargs
    )
    print("Model loaded!")

    # Example input text
    example_text = "The stock market crashed during the economic crisis in 2008, leading to a global recession."    
    stuff = tokenizer.__call__(example_text, return_special_tokens_mask=True)
    #print(stuff)
    tokens = torch.tensor(stuff['input_ids']).to(device)
    special_mask = np.array(stuff['special_tokens_mask'])
    # Extract activations from the correct layer
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    hidden_states = cache["blocks.5.hook_mlp_out"]  # LAYER 5!

    # Pass hidden states into SAE
    with torch.no_grad():
        # sae() or sae.encode()?
        activations = sae(hidden_states)

    print(f"activations {activations[0, :, 10:15]}")

    # Convert activations to NumPy
    activations = activations.to(dtype=torch.float32).detach().cpu().numpy()

    # Select top 100 activated features
    num_features = activations.shape[2]

    feature_activations_sum = activations[0, :, :].sum(axis=0)

    # # randomly select features - approach 1
    # num_selected = min(100, num_features)
    # selected_feature_indices = np.random.choice(num_features, num_selected, replace=False)
    
    # 50 most activated features - approach 2
    num_selected = min(100, num_features)
    selected_feature_indices = np.argsort(feature_activations_sum)[-num_selected:]

    # first 100 features - approach 3

    similarities = []
    N_ITERATIONS_PER_NEURON = 5

    # Get the corresponding tokens and their activations
    tokens = tokenizer.convert_ids_to_tokens(tokens)

    for feature_idx in tqdm(selected_feature_indices):

        # Get the activations for the current feature
        feature_activations = activations[0, :, feature_idx]

        simple_tokens = [tokens[i] for i in range(len(tokens)) if special_mask[i] == 0]
        simple_activations = feature_activations[special_mask == 0]

        #normalize the activations
        simple_activations = simple_activations - np.min(simple_activations)
        # add a small value to avoid division by zero!
        simple_activations = simple_activations / (np.max(simple_activations) + 1e-10)

        similarity = 0
        for run in range(N_ITERATIONS_PER_NEURON):
            # explain
            explanation = explain(simple_tokens, simple_activations, client)
            #print("\n  Explanation from LLM:", explanation)

            # simulate
            for i in range(10):
                simulated_pairs = simulate(simple_tokens, explanation, client)
                if len(simulated_pairs) == len(simple_activations):
                    break
                else:
                    print("The model is fcking idiot! simulated_pairs length is", len(simulated_pairs), "but simple_activations length is", len(simple_activations))
            #print("\n  Simulated pairs:", simulated_pairs)

            # similarity measure
            similarity_per_run = cosine_similarity(
                simple_activations.reshape(1, -1),
                simulated_pairs.reshape(1, -1)
            )[0][0]
            print(f"\n  Cosine similarity {similarity_per_run} for neuron {feature_idx} at run {run}/{N_ITERATIONS_PER_NEURON}")
            similarity += similarity_per_run
        
        # average similarity for this neuron
        similarity /= N_ITERATIONS_PER_NEURON
        print(f"\n  Average cosine similarity for neuron {feature_idx} is {similarity}")
        similarities.append(similarity)


    # compute the average cosine similarity
    average_similarity = np.mean(np.array(similarities))
    print(f"\n  Average cosine similarity for {architecture}: {average_similarity}")

if __name__ == "__main__":
    main()