import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Text completion using HuggingFace models")
    parser.add_argument("model_name", type=str, help="Model name or path from HuggingFace (e.g., 'EleutherAI/pythia-70m')")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None, 
                        help="Device to use (default: auto-detect)")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    args = parser.parse_args()
    
    print(f"Loading {args.model_name} model... This may take a moment.")
    
    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        
        # Check if GPU is available
        if args.device:
            device = args.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {device}")
        model.to(device)
        
        print(f"Model loaded on {device}. You can now generate completions with {args.model_name}!")
        print("Type your prompt and press Enter. Type 'exit' or 'quit' to end.")
        
        # Completion loop
        while True:
            # Get user input
            user_input = input("> ")
            
            # Check if user wants to exit
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting completion engine. Goodbye!")
                break
            
            # Tokenize input
            inputs = tokenizer(user_input, return_tensors="pt").to(device)
            
            # Generate completion
            with torch.no_grad():
                output = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            full_output = tokenizer.decode(output[0], skip_special_tokens=True)
            completion = full_output[len(user_input):].strip()
            
            # Print the completion in purple
            print(f"\033[95m{completion}\033[0m")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()