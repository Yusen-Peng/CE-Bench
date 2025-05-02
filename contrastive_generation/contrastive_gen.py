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
    prompt0 = f"""
    Write a short simple story focusing on {subject.upper()}. Write freely, but focus especially on {subject}. Make sure this subject is consistent across the entire story. {f"Make sure that all tokens you think are related to {subject} must be marked with <subject></subject>" if highlight_subject else ""}
    """
    prompt1 = f"""
    Now, make it about {subject.upper()} specifically being {subject_prefix1.upper()}. Write freely, but focus especially on {subject_prefix1} {subject}. Make sure this subject is consistent across the entire story. {f"Make sure that every word you think are directly related to {subject_prefix1} {subject} must be marked with <subject></subject>" if highlight_subject else ""}
    """
    prompt2 = f"""
    Now, make it about {subject.upper()} being {subject_prefix2.upper()} instead of {subject_prefix1} {subject}. Only change the required words to make it about {subject_prefix2} {subject}, but keep the story coherent. {f"Make sure that every word you think are directly related to {subject_prefix2} {subject} must be marked with <subject></subject>" if highlight_subject else ""}
    """
    return prompt0, prompt1, prompt2

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
    
    progress = tqdm(sorted(subjects * stories_per_subject), desc="Generating contrastive stories")
    file = open(output_file, 'w')
    for subject in progress:
        progress.set_postfix_str(f"subject: {subject}")

        prompt0, prompt_high, prompt_low = contrastive_prompt(subject, "extremely high", "extremely low", highlight_subject=True)
        tqdm.write(f"High: {prompt_high}")
        tqdm.write(f"Low: {prompt_low}")

        response0 = query_openai([
            {"role": "user", "content": prompt0},
        ], model=model, client=client, temperature=0.9)
        tqdm.write(f"0: {response0}")

        response_high = query_openai([
            {"role": "user", "content": prompt0},
            {"role": "assistant", "content": response0},
            {"role": "user", "content": prompt_high},
        ], model=model, client=client)
        tqdm.write(f"High: {response_high}")

        response_low = query_openai([
            {"role": "user", "content": prompt0},   
            {"role": "assistant", "content": response0},
            {"role": "user", "content": prompt_high},
            {"role": "assistant", "content": response_high},
            {"role": "user", "content": prompt_low},
        ], model=model, client=client)
        tqdm.write(f"Low: {response_low}")

        
        if response_high and response_low:
            entry = {
                "story0": response0,
                "story1": response_high,
                "story2": response_low,
                "subject": subject,
            }
            results.append(entry)
            
            # Optional: Sleep to avoid rate limits
            time.sleep(1)
    
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
        "story0": [entry["story0"] for entry in data],
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
        "entropy",
        "size",
        "mass",
        "physical density",
        "darkness",
        "brightness",
        "empathy",
        "sadness",
        "intelligence",
        "confidence",
        "clarity",
        "precision",
        "complexity",
        "intensity",
        "resolution",
        "stability",
        "robustness",
        "sensitivity",
        "emotional attachment",
        "lawfulness (legal)",
        "lawfulness (moral)",
        "integrity",
        "honesty",
        "acuity",
        "motivation",
        "creativity",
        "efficiency",
        "energy",
        "coherence",
        "capacity",
        "focus",
        # "reliability",
        # "flexibility",
        # "saturation",
        # "density",
        # "performance",
        # "resistance (to something)",
        # "reactivity",
        # "fidelity",
        # "alignment",
        # "engagement",
        # "awareness",
        # "risk tolerance",
        # "ambition",   
        # "inhibition",
        # "latency",

        "curiosity",
        "courage",
        "kindness",
        "ambition",
        "self-awareness",
        "jealousy",
        "hope",
        "desperation",
        "trust",
        "fear",
        "love",
        "honor",
        "greed",
        "loyalty",
        "arrogance",
        "wisdom",
        "emotional strength",
        "physical strength",
        "clarity of purpose",
        "self-control",
        "control over others",
        "sense of justice",
        "attachment",
        "confidence",
        "resentment",
        "chaos (internal)",
        "order (internal)",
        "imagination",
        "belief (in self, others, or ideals)",
        "patience",
        "sense of wonder",
        "faith",
        "empathy",
        "willpower",
        "detachment",
        "authenticity",

        "energy",
        "motivation",
        "focus",
        "clarity (mental or emotional)",
        "stress",
        "anxiety",
        "patience",
        "productivity",
        "creativity",
        "confidence",
        "curiosity",
        "discipline",
        "mood",
        "hope",
        "inspiration",
        "social battery",
        "self-esteem",
        "emotional bandwidth",
        "mental load",
        "burnout",
        "organization",
        "satisfaction",
        "overwhelm",
        "engagement (with work, people, ideas)",
        "drive",
        "openness",
        "resilience",
        "mental noise",
        "presence (as in “being present”)",
        "self-compassion",
        "impulsivity",
        "tiredness",
        "interest",
        "frustration",
        "connectedness",
        "relevance to golden gate bridge",
        "relevance to new york city",
        "relevance to the state of ohio",
        "compliance",
        "success",



        "social equality and justice",
        "diversity and inclusion",
        "awareness (of privilege, systems, injustice)",
        "empathy (across identity/experience)",
        "accountability",
        "allyship",
        "tolerance",
        "bias",
        "privilege",
        "justice sensitivity",
        "openness to critique",
        "willingness to learn",
        "moral clarity",
        "intersectionality awareness",
        "cultural humility",
        "activism (level of engagement)",
        "complicity",
        "civic engagement",
        "sense of fairness",
        "emotional labor",
        "social responsibility",
        "inclusiveness",
        "representation",
        "voice (feeling heard or having space)",
        "power (felt or exerted)",
        "solidarity",
        "radical compassion",
        "systemic thinking",
    ]
    subjects = list(set(subjects))
    print(f"Subjects: {len(subjects)}")

    # subjects = subjects[:3]

    # Generate responses
    # generated_data = generate_dataset(subjects, model="gpt-4o", stories_per_subject=5)
    
    # Create HuggingFace dataset
    # hf_dataset = create_hf_dataset(generated_data)
    old_dataset = datasets.load_from_disk("contrastive_stories")
    hf_dataset = old_dataset
    
    # hf_dataset = datasets.concatenate_datasets([old_dataset["train"], hf_dataset])
    # get last 50 entries
    # hf_dataset = old_dataset["train"].select(range(12, len(old_dataset["train"])))

    # Show dataset info
    print(hf_dataset)
    
    # Optional: Push to HuggingFace Hub
    # if len(hf_dataset) > 0:
    #   hf_dataset.save_to_disk("contrastive_stories")
    #   hf_dataset.push_to_hub("GulkoA/contrastive-stories-v1")
    
    return hf_dataset

if __name__ == "__main__":
    dataset = main()

