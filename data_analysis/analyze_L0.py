import pandas as pd
import numpy as np

def compute_statistics(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Compute statistics
    mean_val = np.mean(df['Average_L0'])
    median_val = np.median(df['Average_L0'])
    std_val = np.std(df['Average_L0'], ddof=1)  # Using sample std deviation (N-1)
    
    # Print results
    print(f"Mean: {mean_val:.4f}")
    print(f"Median: {median_val:.4f}")
    print(f"Standard Deviation: {std_val:.4f}")


def main():
    # Example usage
    architecture = "kan_small"
    file_path = f"figures/{architecture}_l0_scores.csv"
    compute_statistics(file_path)

if __name__ == "__main__":
    main()
