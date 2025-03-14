import openai
from dotenv import load_dotenv
import os

def main():
    # Load API key from .env file
    load_dotenv()

    # Retrieve API key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Check if the key is loaded correctly
    if not openai.api_key:
        raise ValueError("❌ OpenAI API key is missing! Make sure it's set in the .env file.")
    
    client = openai.OpenAI()
    # Test GPT-4o API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Is my API key working for GPT-4o?"}],
        max_tokens=10
    )

    print("✅ GPT-4o API is working!")
    print("Response:", response.choices[0].message.content)  # ✅ Correct format

if __name__ == "__main__":
    main()
