import datasets
from openai import OpenAI
import os
import json
import time
from typing import List, Dict, Any
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Setup OpenAI API
def setup_openai_api():
    """Setup OpenAI API with key from environment variable"""
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client

def wikithing_str(thing: dict) -> str:
    """
    Convert a Wikithing object to a string representation.
    
    Args:
        thing: A dictionary representing a Wikithing object.
        
    Returns:
        A string representation of the Wikithing object.
    """
    return f"{thing['name']} ({thing['description']})"

def subject_prompt(subject: str) -> str:
    prompt1 = f"""
    Write how you would describe {subject.upper()} in its high, extreme form. Rephrase things if needed, be very brief, specific, detailed, and realistic.
    For example,
    "active" -> "extremely vibrant, energetic, and lively"
    "angry" -> "extremely mad, furious, and enraged"
    """
    prompt2 = f"""
    Now, write how you would describe the exact opposite of {subject.upper()}. Rephrase things if needed, be very brief, specific, detailed, and realistic. DO NOT USE THE WORDS {subject.upper()} in your answer, instead write the opposite of the concept.
    For example,
    "active" -> "very inactive, lethargic, sluggish, and lazy"
    "angry" -> "very calm, peaceful, and relaxed"
    """
    return prompt1, prompt2

def story_prompt(subject1: str, subject2: str) -> str:
    prompt1 = f"""
    Write a short story describing the following: {subject1}.
    """
    prompt2 = f"""
    Now, rewrite this story describing the following: {subject2} (the exact opposite of the previous story).
    """
    return prompt1, prompt2

# Function to query OpenAI API
def query_openai(prompt: list, model, client: OpenAI, temperature: float = 0.7) -> str:
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
        instructions="Write concisely and clearly. Use no more than 200 words. Be as brief as possible.",
        input=prompt,
        temperature=temperature,
        # max_output_tokens=max_tokens,
      )

      return response.output_text.strip()
    except Exception as e:
        print(f"Error querying OpenAI API: {e}")
        return ""

def generate_story(subject: str, model: str, client: OpenAI) -> dict:
    subject_prompt1, subject_prompt2 = subject_prompt(subject)

    subject1 = query_openai([
        {"role": "user", "content": subject_prompt1},
    ], model=model, client=client)

    subject2 = query_openai([
        {"role": "user", "content": subject_prompt1},
        {"role": "assistant", "content": subject1},
        {"role": "user", "content": subject_prompt2},
    ], model=model, client=client)

    tqdm.write(f"Subject: {subject}")
    tqdm.write(f"High: {subject1}")
    tqdm.write(f"Low: {subject2}")

    prompt1, prompt2 = story_prompt(subject1, subject2)
    story1 = query_openai([
        {"role": "user", "content": prompt1},
    ], model=model, client=client)
    story2 = query_openai([
        {"role": "user", "content": prompt1},
        {"role": "assistant", "content": story1},
        {"role": "user", "content": prompt2},
    ], model=model, client=client)
    tqdm.write(f"Story 1: {story1}")
    tqdm.write(f"Story 2: {story2}")
    
    entry = {
        "story1": story1,
        "story2": story2,
        "subject1": subject1,
        "subject2": subject2,
    }

    return entry

# Generate dataset from prompts
def generate_dataset(subjects: datasets.Dataset, model: str, output_file: str = "generated_dataset.json", stories_per_subject: int=1) -> List[Dict[str, Any]]:
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
    
    progress = tqdm(subjects, desc="Generating contrastive stories", leave=False, unit="subject")
    file = open(output_file, 'w')
    for subject in progress:

        subject_str = f"{subject['title']} ({subject['description']})"
        progress.set_postfix_str(f"subject: {subject_str}")

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(generate_story, subject_str, model, client) for _ in range(stories_per_subject)]
            for i, future in enumerate(tqdm(futures, desc="Generating stories", leave=False, unit="story")):
                try:
                    entry = future.result()
                    entry["subject_id"] = subject["id"]
                    entry["subject_title"] = subject["title"]
                    results.append(entry)
                    tqdm.write(f"Generated story {i}")
                except Exception as e:
                    tqdm.write(f"Error generating story: {e}")
                    raise e
    
        # Save raw results to file
        file.seek(0)
        file.write(json.dumps(results, indent=2))
        file.flush()
    
    file.close()
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
        "subject1": [entry["subject1"] for entry in data],
        "subject2": [entry["subject2"] for entry in data],
        "subject_id": [entry["subject_id"] for entry in data],
        "subject_title": [entry["subject_title"] for entry in data],
    })
    
    return dataset

def load_subjects(file_path: str) -> datasets.Dataset:
    wikithings = datasets.load_dataset("GulkoA/wikithings", split="train")
    df = pd.read_csv(file_path)
    subject_ids = set(df["id"].tolist())
    subjects = wikithings.filter(lambda x: x["id"] in subject_ids)
    # df items not in wikithings
    not_subjects = set(subject_ids) - set(subjects["id"])
    print(f"Subjects not in wikithings: {not_subjects}")
    quit()
    return subjects

# Example usage
def main():
    
    subjects = load_subjects("curated_subject_list.csv")
    print(f"Subjects: {len(subjects)}")

    subjects = subjects.take(10)
    print(subjects)

    # Generate responses
    generated_data = generate_dataset(subjects, model="gpt-4.1", stories_per_subject=5)

    # with open("generated_dataset.json", "r") as f:
    #     generated_data = json.load(f)
    
    # Create HuggingFace dataset
    hf_dataset = create_hf_dataset(generated_data)
    # old_dataset = datasets.load_from_disk("contrastive-stories-v3")
    # hf_dataset = old_dataset
    
    # hf_dataset = datasets.concatenate_datasets([old_dataset, hf_dataset])
    # get last 50 entries
    # hf_dataset = old_dataset["train"].select(range(12, len(old_dataset["train"])))

    # Show dataset info
    print(hf_dataset)
    
    # Optional: Push to HuggingFace Hub
    # if len(hf_dataset) > 0:
    #   hf_dataset.save_to_disk("contrastive-stories-v4")
    #   hf_dataset.push_to_hub("GulkoA/contrastive-stories-v4")
    
    return hf_dataset

if __name__ == "__main__":
    dataset = main()

