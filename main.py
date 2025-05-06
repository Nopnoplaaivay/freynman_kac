import numpy as np
import pandas as pd
import time
from scipy.stats import norm

# =============================================================================
# Import Pricing Methods
# =============================================================================
from src.methods import (
    monte_carlo_price,
    finite_difference_price,
    method_of_lines_price,
    spectral_method_price,
    fem_price,
    deep_ritz_price,  
    deeponet_price,
    fourier_neural_operator_price,
    quantum_price_simulation  
)

METHODS = {
    "Monte Carlo": monte_carlo_price,
    "Finite Difference": finite_difference_price,
    "Method of Lines": method_of_lines_price,
    "Spectral Method": spectral_method_price,
    "Finite Elements": fem_price,
    # "Deep Ritz": deep_ritz_price,
    # "DeepONet": deeponet_price,
    # "Fourier Neural Operator": fourier_neural_operator_price,
    "Quantum Simulation": quantum_price_simulation
}

# =============================================================================
# Analytical (Closed-Form) Black-Scholes Price for a European Call Option
# =============================================================================
def black_scholes_price(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

def run(name, func, true_price, S0=100, K=100, r=0.05, sigma=0.2, T=1.0):
    method_start = time.time()
    num_runs = 10
    prices = []
    errors = []
    runtimes = []

    fd_kwargs = {"S_max": 300, "M": 200, "N": 200}
    mol_kwargs = {"S_max": 300, "M": 200}
    spec_kwargs = {"S_max": 300, "N": 50}
    fem_kwargs = {"S_max": 300, "M": 200, "N": 200}
    deep_ritz_kwargs = {"S_max": 300, "num_collocation": 1000, "num_epochs": 5000, "lr": 1e-3}
    deeponet_kwargs = {"S_max": 300, "num_samples": 1000, "epochs": 1000, "lr": 1e-3, "m": 100, "p": 50}
    fourier_neural_operator_kwargs = {"S_max": 300, "num_samples": 1000, "epochs": 1000, "lr": 1e-3, "m": 100, "modes": 16, "width": 64, "layers": 4}

    for i in range(num_runs):
        start = time.time()

        # Directly call the synchronous function.
        if name == "Monte Carlo":
            price = func(S0, K, r, sigma, T, N=1000000)
        elif name == "Method of Lines":
            price = func(S0, K, r, sigma, T, **mol_kwargs)
        elif name == "Finite Difference":
            price = func(S0, K, r, sigma, T, **fd_kwargs)
        elif name == "Spectral Method":
            price = func(S0, K, r, sigma, T, **spec_kwargs)
        elif name == "Finite Elements":
            price = func(S0, K, r, sigma, T, **fem_kwargs)
        elif name == "Deep Ritz":
            price = func(S0, K, r, sigma, T, **deep_ritz_kwargs)
        elif name == "DeepONet":
            price = func(S0, K, r, sigma, T, **deeponet_kwargs)
        elif name == "Fourier Neural Operator":
            price = func(S0, K, r, sigma, T, **fourier_neural_operator_kwargs)
        elif name == "Quantum Simulation":
            # Simulate the quantum circuit with a normalized state vector.
            state = np.array([0.7071, 0.7071, 0.0, 0.7071])
            price = func(state, r, T, shots=100)

        end = time.time()

        runtime = end - start
        error = abs(price - true_price)

        prices.append(price)
        errors.append(error)
        runtimes.append(runtime)

    # Convert lists to NumPy arrays for easy statistics calculation.
    prices_arr = np.array(prices)
    errors_arr = np.array(errors)
    runtimes_arr = np.array(runtimes)

    # Compute mean and standard deviation.
    mean_price = np.mean(prices_arr)
    std_price = np.std(prices_arr)
    mean_error = np.mean(errors_arr)
    std_error = np.std(errors_arr)
    mean_runtime = np.mean(runtimes_arr)
    std_runtime = np.std(runtimes_arr)

    method_end = time.time()
    method_runtime = method_end - method_start

    # Save the statistics for the current method.
    return (name, mean_price, std_price, mean_error, std_error, mean_runtime, std_runtime)

# =============================================================================
# Main Benchmark Comparison with Multiple Parameter Sets
# =============================================================================
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

        # Compute the analytical Black–Scholes price.
        true_price = black_scholes_price(S0, K, r, sigma, T)

        # Run each method synchronously
        for name, func in METHODS.items():
            result = run(name, func, true_price, S0, K, r, sigma, T)

            # Store results for this parameter set
            name, mean_price, std_price, mean_error, std_error, mean_runtime, std_runtime = result
            results_list.append({
                "S0": S0,
                "K": K,
                "r": r,
                "sigma": sigma,
                "T": T,
                "True Price": true_price,
                "Method": name,
                "Mean Price": mean_price,
                "Std Price": std_price,
                "Mean Error": mean_error,
                "Std Error": std_error,
                "Mean Runtime": mean_runtime,
                "Std Runtime": std_runtime
            })
            print(f"{name} S0={S0:.2f}, K={K:.2f}, r={r:.2f}, sigma={sigma:.2f}, T={T:.2f} - Mean Price: {mean_price:.4f}, Std Price: {std_price:.4f}, Mean Error: {mean_error:.4f}, Std Error: {std_error:.4f}, Mean Runtime: {mean_runtime:.4f}, Std Runtime: {std_runtime:.4f}")

    # Convert results to a DataFrame
    df = pd.DataFrame(results_list)

    # Export to CSV
    df.to_csv("benchmark_results.csv", index=False)
    print("Results saved to benchmark_results.csv")


if __name__ == "__main__":
    main()