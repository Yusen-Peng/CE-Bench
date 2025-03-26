from datasets import load_dataset

if __name__ == "__main__":
    ds = load_dataset("GulkoA/TinyStories-Llama-3.2-1B-cache-100k", split="train")
    print(ds.column_names)