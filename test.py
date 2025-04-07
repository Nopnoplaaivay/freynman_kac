import asyncio
import numpy as np
import time
from scipy.stats import norm

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
    "Deep Ritz": deep_ritz_price,  
    "DeepONet": deeponet_price,
    "Fourier Neural Operator": fourier_neural_operator_price,
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

async def run_async(name, func, true_price, S0=100, K=100, r=0.05, sigma=0.2, T=1.0):
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

        # Offload the synchronous function to a thread pool.
        if name == "Monte Carlo":
            price = await asyncio.to_thread(func, S0, K, r, sigma, T, N=1000000)
        elif name == "Method of Lines":
            price = await asyncio.to_thread(func, S0, K, r, sigma, T, **mol_kwargs)
        elif name == "Finite Difference":
            price = await asyncio.to_thread(func, S0, K, r, sigma, T, **fd_kwargs)
        elif name == "Spectral Method":
            price = await asyncio.to_thread(func, S0, K, r, sigma, T, **spec_kwargs)
        elif name == "Finite Elements":
            price = await asyncio.to_thread(func, S0, K, r, sigma, T, **fem_kwargs)
        elif name == "Deep Ritz":
            price = await asyncio.to_thread(func, S0, K, r, sigma, T, **deep_ritz_kwargs)
        elif name == "DeepONet":
            price = await asyncio.to_thread(func, S0, K, r, sigma, T, **deeponet_kwargs)
        elif name == "Fourier Neural Operator":
            price = await asyncio.to_thread(func, S0, K, r, sigma, T, **fourier_neural_operator_kwargs)
        elif name == "Quantum Simulation":
            # Simulate the quantum circuit with a normalized state vector.
            state = np.array([0.7071, 0.7071, 0.0, 0.7071])
            price = await asyncio.to_thread(func, state, r, T, shots=100)

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
    print(f"Total runtime for {name}: {method_runtime:.4f} seconds")

    # Save the statistics for the current method.
    return (name, mean_price, std_price, mean_error, std_error, mean_runtime, std_runtime)

# =============================================================================
# Main Benchmark Comparison
# =============================================================================
async def main():
    # Parameters
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0

    total_time_start = time.time()

    # Compute the analytical Black–Scholes price.
    true_price = black_scholes_price(S0, K, r, sigma, T)
    print("Black-Scholes closed-form price: {:.4f}".format(true_price))


    tasks = [run_async(name, func, true_price) for name, func in METHODS.items()]

    results = await asyncio.gather(*tasks)

    # Print the benchmark comparison.
    print("\nBenchmark Comparison over {} runs:".format(10))
    header = (
        f"{'Method':30s} {'Mean Price':12s} {'Std Price':12s} "
        f"{'Mean Error':12s} {'Std Error':12s} {'Mean Runtime':15s} {'Std Runtime':15s}"
    )
    print(header)

    for result in results:
        name, mean_price, std_price, mean_error, std_error, mean_runtime, std_runtime = result
        print(
            f"{name:30s} {mean_price:12.4f} {std_price:12.4f} {mean_error:12.4f} "
            f"{std_error:12.4f} {mean_runtime:15.4f} {std_runtime:15.4f}"
        )

    total_time_end = time.time()
    total_runtime = total_time_end - total_time_start
    print(f"\nTotal runtime for all methods: {total_runtime:.4f} seconds")


if __name__ == "__main__":
    asyncio.run(main())