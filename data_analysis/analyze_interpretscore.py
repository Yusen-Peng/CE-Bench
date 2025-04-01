import pandas as pd

if __name__ == "__main__":
    # Load the data
    architecture = "GPT_cache_only_kan"
    steps = "1k"

    best_model = "best_3686400_ce_2.35626_ori_2.33838"
    file_path = f"figures/{architecture}_{steps}_feature_interpretability_results.csv"
    df = pd.read_csv(file_path)
    average_score = df["Score"].mean()
    median_score = df["Score"].median()
    std_dev_score = df["Score"].std()

    print(f"Average Interpretability Score: {average_score:.2f}")
    print(f"Median Interpretability Score: {median_score:.2f}")
    print(f"Standard Deviation of Interpretability Score: {std_dev_score:.2f}")
