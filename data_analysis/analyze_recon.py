import pandas as pd

def compute_loss_averages(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Compute averages
    avg_orig_loss = df['orig_loss'].mean()
    avg_reconstr_loss = df['reconstr_loss'].mean()
    avg_zero_loss = df['zero_loss'].mean()
    
    # Print results
    print(f"Average Original Loss: {avg_orig_loss:.4f}")
    print(f"Average Reconstruction Loss: {avg_reconstr_loss:.4f}")
    print(f"Average Zero Loss: {avg_zero_loss:.4f}")


def main():
    architecture = "gated"
    file_path = f"figures/{architecture}_batch_losses.csv"
    compute_loss_averages(file_path)

if __name__ == "__main__":
    main()
