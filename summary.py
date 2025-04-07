import pandas as pd
import numpy as np

# Load the CSV file
file_path = "benchmark_results.csv"
df = pd.read_csv(file_path)

# Define the list of methods to include in the summary
methods = [
    "Monte Carlo",
    "Finite Difference",
    "Method of Lines",
    "Spectral Method",
    "Finite Elements",
    "Deep Ritz",
    "DeepONet",
    "Fourier Neural Operator",
    "Quantum Simulation"
]

# Initialize a dictionary to store results
summary = {}

# Compute metrics for each method
for method in methods:
    # Filter rows for the current method
    method_data = df[df["Method"] == method]
    
    # Extract relevant columns
    true_prices = method_data["True Price"]
    mean_prices = method_data["Mean Price"]
    mean_errors = method_data["Mean Error"]
    mean_runtimes = method_data["Mean Runtime"]
    
    # Compute metrics
    mae = mean_errors.mean()
    rmse = np.sqrt((mean_errors**2).mean())
    max_error = mean_errors.max()
    min_error = mean_errors.min()
    median_error = mean_errors.median()
    
    mean_runtime = mean_runtimes.mean()
    median_runtime = mean_runtimes.median()
    max_runtime = mean_runtimes.max()
    
    # Store results in the summary dictionary
    summary[method] = {
        "MAE": mae,
        "RMSE": rmse,
        "Max Error": max_error,
        "Min Error": min_error,
        "Median Error": median_error,
        "Mean Runtime": mean_runtime,
        "Median Runtime": median_runtime,
        "Max Runtime": max_runtime
    }

# Convert the summary dictionary to a DataFrame
summary_df = pd.DataFrame(summary).transpose()

# Add column names for better readability
summary_df.index.name = "Method"
summary_df.reset_index(inplace=True)

# Print the summary table
print(summary_df)

# Optionally, save the summary to a CSV file
summary_df.to_csv("summary_metrics.csv", index=False)