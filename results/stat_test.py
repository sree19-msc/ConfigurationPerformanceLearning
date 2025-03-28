import os
import pandas as pd
from itertools import combinations
from scipy.stats import ttest_rel

# Define relative file paths for each model's results (assuming these files are in a folder called "data")
file_paths = {
    "Linear Regression": "linear_regression.csv",
    "XGBoost": "xgboost.csv",
    "XGBoost Tuned": "xgboost_tuned.csv",
    "Random Forest": "random_forest.csv",
    "Random Forest Tuned": "random_forest_tuned.csv",
    "Voting Regressor": "voting_regressor.csv",
    "Voting Regressor Tuned": "voting_regressor_tuned.csv",
    "ANN": "ann.csv",
    "ANN Tuned": "ann_tuned.csv"
}

# Read MAE data for each model
mae_data = {}
for model, path in file_paths.items():
    df = pd.read_csv(path)
    print(f"Reading: {path}")
    # Assuming the column name for the MAE values is "Average MAE"
    mae_data[model] = df["MAE"].values

# Perform paired t-tests between all model pairs
results = []
model_names = list(mae_data.keys())
for model_a, model_b in combinations(model_names, 2):
    t_stat, p_val = ttest_rel(mae_data[model_a], mae_data[model_b])
    significance = "Yes" if p_val < 0.05 else "No"
    results.append({
        "Model A": model_a,
        "Model B": model_b,
        "T-Statistic": t_stat,
        "P-Value": p_val,
        "Significant": significance
    })

# Convert results to a DataFrame
ttest_results_df = pd.DataFrame(results)
print(ttest_results_df)

# Define output folder and file path (using relative paths)
output_path = "ttest_results.csv"

# Save the results DataFrame to a CSV file with headers
ttest_results_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")
