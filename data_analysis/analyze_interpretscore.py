import pandas as pd

if __name__ == "__main__":
    # Load the data
    architecture = "kan_relu_dense_latent"
    steps = "9k"
    file_path = f"figures/{architecture}_{steps}_feature_interpretability_results.csv"
    df = pd.read_csv(file_path)
    average_score = df["Score"].mean()
    median_score = df["Score"].median()
    std_dev_score = df["Score"].std()

    print(f"Average Interpretability Score: {average_score:.2f}")
    print(f"Median Interpretability Score: {median_score:.2f}")
    print(f"Standard Deviation of Interpretability Score: {std_dev_score:.2f}")
