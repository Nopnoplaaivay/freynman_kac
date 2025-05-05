import numpy as np


def main():
    n_sets = 100  # Number of parameter sets to generate
    np.random.seed(42)  # For reproducibility

    # Generate random parameter sets
    S0_values = np.random.uniform(80, 120, n_sets)  # Stock prices between 80 and 120
    K_values = np.random.uniform(80, 120, n_sets)   # Strike prices between 80 and 120
    r_values = np.random.uniform(0.01, 0.1, n_sets) # Interest rates between 1% and 10%
    sigma_values = np.random.uniform(0.1, 0.4, n_sets) # Volatilities between 10% and 40%
    T_values = np.random.uniform(0.25, 2.0, n_sets)   # Times to maturity between 3 months and 2 years

    results_list = []

    for i in range(n_sets):
        S0 = S0_values[i]
        K = K_values[i]
        r = r_values[i]
        sigma = sigma_values[i]
        T = T_values[i]
        T = 1.0  # Set T to 1.0 for all runs
        print(f"S0: {S0}, K: {K}, r: {r}, sigma: {sigma}, T: {T}")

if __name__ == "__main__":
    main()