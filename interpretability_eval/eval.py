import torch
import torch.nn as nn
import numpy as np
import openai
import os
import pandas as pd
from dotenv import load_dotenv
from transformers import AutoTokenizer
from sae_lens import SAE, HookedSAETransformer

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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded!")

    # Load the trained SAE
    architecture = "kan_mini"
    sae_checkpoint_path = f"checkpoints/{architecture}/final_36864000"
    sae = SAE.load_from_pretrained(path=sae_checkpoint_path, device=device)
    print("SAE loaded!")

    # Load the LLaMA model using HookedSAETransformer
    model = HookedSAETransformer.from_pretrained_no_processing(
        model_name,
        device=device,
        **sae.cfg.model_from_pretrained_kwargs
    )
    print("Model loaded!")

    # Example input text
    example_text = "The stock market crashed during the economic crisis in 2008."
    tokens = tokenizer(example_text, return_tensors="pt").to(device)

    # Extract activations from the correct layer
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens.input_ids) 
    hidden_states = cache["blocks.0.hook_mlp_out"]

    # Pass hidden states into SAE
    with torch.no_grad():
        activations = sae(hidden_states)

    # Convert activations to NumPy
    activations = activations.to(dtype=torch.float32).detach().cpu().numpy()

    # Select top 100 activated features
    num_features = activations.shape[2]
    feature_activations_sum = activations[0, :, :].sum(axis=0)  # Shape: (num_features,)

    # Select the top 100 most activated features
    num_selected = min(100, num_features)
    selected_feature_indices = np.argsort(feature_activations_sum)[-num_selected:]  # Top activated features


    print(f"Selected {num_selected} features for evaluation.")
    print(f"The first 5 features are: {selected_feature_indices[:5]}")

    # Identify top activated tokens for each feature
    feature_prompts = []
    for feature_idx in selected_feature_indices:
        # Extract activations for this feature across all tokens
        feature_activations = activations[0, :, feature_idx]

        # Find top 5 activated tokens and insert into prompt
        top_token_indices = np.argsort(feature_activations)[-5:]  # Get top 5 token indices
        top_tokens = [tokenizer.convert_ids_to_tokens(tokens.input_ids[0, idx].item()) for idx in top_token_indices]
        top_activations = feature_activations[top_token_indices]

        # Construct interpretability prompt
        prompt = f"""
        Background
        We are analyzing the activation levels of features in a neural network, where each feature activates certain tokens in a text. 
        Each token's activation value indicates its relevance to the feature, with higher values showing stronger association.

        Features are categorized as:
        A. Low-level features (word-level polysemy, e.g., "crushed things", "Europe").
        B. High-level features (long-range patterns, e.g., "enumeration", "one of the [number/quantifier]").
        C. Undiscernible features (random noise, irrelevant activations).

        Task:
        Classify the feature as low-level, high-level, or undiscernible and assign it a monosemanticity score based on:

        **Activation Consistency Scoring**
        5: Clear pattern with no deviating examples
        4: Clear pattern with one or two deviating examples
        3: Clear overall pattern but some inconsistencies
        2: Broad theme but lacks structure
        1: No discernible pattern

        **Feature Activation Data**
        Token - Activation
        {top_tokens[0]} - {top_activations[0]:.4f}
        {top_tokens[1]} - {top_activations[1]:.4f}
        {top_tokens[2]} - {top_activations[2]:.4f}
        {top_tokens[3]} - {top_activations[3]:.4f}
        {top_tokens[4]} - {top_activations[4]:.4f}


        ### **Response Format**
        Return ONLY a valid JSON object, with NO extra text or explanation.
        Your response MUST match this exact structure:

        ```json
        {{
            "Feature category": "Low-level" | "High-level" | "Undiscernible",
            "Score": 1 | 2 | 3 | 4 | 5,
            "Explanation": "A concise explanation of why this feature was classified this way."
        }}

        **Important**
        Only return a valid JSON object without any extra text.
        """
        feature_prompts.append(prompt)

    print(f"Generated {len(feature_prompts)} prompts for LLM evaluation.")
    print("Example Prompt:", feature_prompts[0])

    # **Send Prompts to GPT-4o for Evaluation**
    responses = []
    for prompt in feature_prompts:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in feature interpretability."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )
        responses.append(response.choices[0].message.content)

    import re
    import json

    parsed_responses = []
    for i, response in enumerate(responses):
        try:
            # Extract JSON using regex (removes any extra text)
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                json_str = match.group(0)  # Extract the JSON part
                response_json = json.loads(json_str)  # Parse it
            else:
                raise json.JSONDecodeError("No valid JSON found", response, 0)

            category = response_json.get("Feature category", "Error")
            score = response_json.get("Score", -1)
            explanation = response_json.get("Explanation", "Parsing failed.")
        except (json.JSONDecodeError, KeyError):
            category, score, explanation = "Error", -1, "Parsing failed."

        parsed_responses.append({
            "Feature Index": selected_feature_indices[i],
            "Category": category,
            "Score": score,
            "Explanation": explanation
        })

    # Save Results to CSV
    results_path = f"figures/{architecture}_feature_interpretability_results.csv"
    df = pd.DataFrame(parsed_responses)
    df.to_csv(results_path, index=False)

    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
