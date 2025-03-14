import re

def extract_ranks(file_path):
    with open(file_path, 'r') as file:
        log_data = file.readlines()
    
    ranks = []
    
    for line in log_data:
        match = re.search(r'Rank: (\d+)', line)
        if match:
            ranks.append(int(match.group(1)))
            if len(ranks) == 2:  # Stop once we get two ranks
                break
    
    original_rank = ranks[0] if len(ranks) > 0 else None
    sae_rank = ranks[1] if len(ranks) > 1 else None
    
    print(f"Original Rank: {original_rank}")
    print(f"SAE Rank: {sae_rank}")

def main():
    architecture = "gated"
    file_path = f"figures/{architecture}_capability.log"
    extract_ranks(file_path)

if __name__ == "__main__":
    main()

