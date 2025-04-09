import datasets
from openai import OpenAI
import os
import json
import time
from typing import List, Dict, Any
from tqdm import tqdm
from dotenv import load_dotenv

# Setup OpenAI API
def setup_openai_api():
    """Setup OpenAI API with key from environment variable"""
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client

def contrastive_prompt(subject: str, subject_prefix1: str, subject_prefix2: str, highlight_subject: bool=True) -> str:
    prompt1 = f"""
    Write a short simple story about {subject_prefix1} {subject}. Write freely, but focus especially on {subject_prefix1} {subject}. Make sure this subject is consistent across the entire story. {f"All words that are related to {subject} must be highlighted with *" if highlight_subject else ""}
    """
    prompt2 = f"""
    Now, edit this story to make it about {subject_prefix2} {subject} instead of {subject_prefix1} {subject}. Only change the required words to make it about {subject_prefix2} {subject}, but keep the story coherent. {f"All words that are related to {subject} must be highlighted with *" if highlight_subject else ""}
    """
    return prompt1, prompt2

# Function to query OpenAI API
def query_openai(prompt: list, model, client: OpenAI, temperature: float = 0.7, max_tokens: int = 256) -> str:
    """
    Query OpenAI API with a prompt
    
    Args:
        prompt: The prompt to send to OpenAI API
        model: OpenAI model to use
        temperature: Controls randomness (0.0 is deterministic, 1.0 is random)
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated text response
    """
    try:
      response = client.responses.create(
        model=model,
        instructions="You are a story teller. Write concisely and clearly. Use no more than 200 words.",
        input=prompt,
        temperature=temperature,
        # max_output_tokens=max_tokens,
      )

      return response.output_text.strip()
    except Exception as e:
        print(f"Error querying OpenAI API: {e}")
        return ""

# Generate dataset from prompts
def generate_dataset(subjects: List[str], model: str, output_file: str = "generated_dataset.json") -> List[Dict[str, Any]]:
    """
    Generate responses from OpenAI API based on prompts and save as dataset
    
    Args:
        prompts: List of prompts to send to OpenAI API
        model: OpenAI model to use
        output_file: File to save raw responses to
        
    Returns:
        List of dictionaries containing prompts and responses
    """
    client = setup_openai_api()
    results = []
    
    stories_per_subject = 5
    progress = tqdm(sorted(subjects * stories_per_subject), desc="Generating contrastive stories")
    for subject in progress:
        progress.set_postfix_str(f"Subject: {subject}")
        prompt_high, prompt_low = contrastive_prompt(subject, "high", "the opposite of high", highlight_subject=True)

        response_high = query_openai([
            {"role": "user", "content": prompt_high},
        ], model=model, client=client)
        tqdm.write(f"High: {response_high}")

        response_low = query_openai([
            {"role": "user", "content": prompt_high},
            {"role": "assistant", "content": response_high},
            {"role": "user", "content": prompt_low},
        ], model=model, client=client)
        tqdm.write(f"Low: {response_low}")

        
        if response_high and response_low:
            entry = {
                "story1": response_high,
                "story2": response_low,
                "subject": subject,
            }
            results.append(entry)
            
            # Optional: Sleep to avoid rate limits
            time.sleep(1)
    
    # Save raw results to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

# Convert results to HuggingFace dataset
def create_hf_dataset(data: List[Dict[str, Any]], dataset_name: str = "openai_generated_dataset") -> datasets.Dataset:
    """
    Convert generated data to HuggingFace dataset
    
    Args:
        data: List of dictionaries containing prompts and responses
        dataset_name: Name for the dataset
        
    Returns:
        HuggingFace Dataset object
    """
    # Create dataset from dict
    dataset = datasets.Dataset.from_dict({
        "story1": [entry["story1"] for entry in data],
        "story2": [entry["story2"] for entry in data],
        "subject": [entry["subject"] for entry in data],
    })
    
    return dataset

# Example usage
def main():
    # Example prompts
    subjects = [
        "temperature",
        "love",
        "pressure",
        "volume",
        # "entropy",
        "size",
        "mass",
        # "density",
        "darkness",
        "brightness",
        "empathy",
        "sadness",
    ]

    # Generate responses
    generated_data = generate_dataset(subjects, model="gpt-4o")
    
    # Create HuggingFace dataset
    hf_dataset = create_hf_dataset(generated_data)
    # old_dataset = datasets.load_dataset("GulkoA/contrastive-stories")
    # hf_dataset = datasets.concatenate_datasets([old_dataset["train"], hf_dataset])
    
    # Show dataset info
    print(hf_dataset)
    
    # Optional: Push to HuggingFace Hub
    if len(hf_dataset) > 0:
      hf_dataset.save_to_disk("contrastive_stories")
      hf_dataset.push_to_hub("GulkoA/contrastive-stories")
    
    return hf_dataset

if __name__ == "__main__":
    dataset = main()

