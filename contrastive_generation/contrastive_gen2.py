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

def subject_prompt(subject: str) -> str:
    prompt1 = f"""
    Write how you would describe {subject.upper()} in its high, extreme form. Rephrase things if needed, be very brief, specific, detailed, and realistic. DO NOT USE THE WORD {subject.upper()} in your answer, but DESCRIBE the concept in a positive way EXACTLY.
    For example,
    "active" -> "extremely vibrant, energetic, and lively"
    "angry" -> "extremely mad, furious, and enraged"
    """
    prompt2 = f"""
    Now, write how you would describe {subject.upper()} in its semantically opposite form. Rephrase things if needed, be very brief, specific, detailed, and realistic. DO NOT USE THE WORD {subject.upper()} in your answer, but DESCRIBE the concept in a negative way EXACTLY.
    For example,
    "active" -> "very inactive, lethargic, sluggish, and lazy"
    "angry" -> "very calm, peaceful, and relaxed"
    """
    return prompt1, prompt2

def story_prompt(subject1: str, subject2: str) -> str:
    prompt1 = f"""
    Write a short story as if you are characterized by the following: {subject1}.
    """
    prompt2 = f"""
    Now, rewrite this short story as if you are characterized by the following: {subject2} (the exact opposite of the previous story).
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
        instructions="Write concisely and clearly. Use no more than 200 words. ",
        input=prompt,
        temperature=temperature,
        # max_output_tokens=max_tokens,
      )

      return response.output_text.strip()
    except Exception as e:
        print(f"Error querying OpenAI API: {e}")
        return ""

# Generate dataset from prompts
def generate_dataset(subjects: List[str], model: str, output_file: str = "generated_dataset.json", stories_per_subject: int=1) -> List[Dict[str, Any]]:
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
    
    progress = tqdm(sorted(subjects), desc="Generating contrastive stories", leave=False, unit="subject")
    file = open(output_file, 'w')
    for subject in progress:
        progress.set_postfix_str(f"subject: {subject}")

        subject_prompt1, subject_prompt2 = subject_prompt(subject)

        for i in tqdm(range(stories_per_subject), desc="Generating stories", leave=False, unit="story"):

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
            
            if story1 and story2:
                entry = {
                    "story1": story1,
                    "story2": story2,
                    "subject1": subject1,
                    "subject2": subject2,
                    "subject": subject,
                }
                results.append(entry)
            
            # Optional: Sleep to avoid rate limits
            # time.sleep(1)
    
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
        "subject": [entry["subject"] for entry in data],
    })
    
    return dataset

# Example usage
def main():
    subjects = [
        # "honesty",
        # "integrity",
        # "kindness",
        # "empathy",
        # "courage",
        # "harm",
        # "fear",
        # "anger",
        "illegal activity",
        # "knowledge",
        # "creativity",
        # "leadership",
        # "loyalty",
        # "self-consciousness",
        # "open-mindedness",
        # "respect",



        # "patience",
        # "curiosity",
        # "reliability",
        # "responsibility",
        # "humility",
        # "transparency",
        # "clarity",
        # "attentiveness",
        # "thoughtfulness",
        # "optimism",
        # "resilience",
        # "adaptability",
        # "fairness",
        # "impartiality",
        # "accountability",
        # "inclusivity",
        # "generosity",
        # "mindfulness",
        # "gratitude",
        # "confidence",
        # "wisdom",
        # "humor",
        # "forgiveness",
        # "discernment",
        # "tact",
        # "steadfastness",
        # "supportiveness",
        # "trustworthiness",
        # "compassion",
        # "diligence",
        # "perseverance",
        # "resourcefulness",
        # "conscientiousness",
        # "decisiveness",
        # "discipline",
        # "meticulousness",
        # "objectivity",
        # "perceptiveness",
        # "foresight",
        # "enthusiasm",
        # "benevolence",
        # "altruism",
        # "selflessness",
        # "prudence",
        # "temperance",
        # "equanimity",
        # "gentleness",
        # "mercy",
        # "frankness",
        # "collaboration",
        # "cooperation",
        # "solidarity",
        # "vigilance",
        # "acceptance",
        # "initiative",
        # "spontaneity",
        # "intuition",
        # "poise",
        # "balance",
        # "serenity",
        # "deliberation",
        # "proactivity",
        # "assertiveness",
        # "insisting",
        # "politeness",
        # "willingness to help",
        # "patriotism",
        # "sincerity",

        # "love of cheese",
        # "love of chocolate",
        # "love of coffee",
        # "love of tea",
        # "love of golden gate bridge",
        # "love of dogs",
        # "love of cats",
        # "love of nature",
        # "love of sci-fi stories",
        # "behaving like a child",

        # "poetic",
        # "artistic",
        # "whimsy",
        # "serendipity",
        # "radiance",
        # "sonder",
        # "loudness",
        # "colorfulness",
        # "brightness",
        # "skillfulness",


    ]
    subjects = list(set(subjects))
    print(f"Subjects: {len(subjects)}")

    # subjects = subjects[:3]

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
    #   hf_dataset.save_to_disk("contrastive-stories-v3.1")
    #   hf_dataset.push_to_hub("GulkoA/contrastive-stories-v3")
    
    return hf_dataset

if __name__ == "__main__":
    dataset = main()

