import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import norm
import time
import pandas as pd
from itertools import product
import signal
import functools

# Define a timeout decorator to handle long-running calculations
class TimeoutError(Exception):
    pass

def timeout(seconds, error_message="Calculation timed out"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(error_message)
            
            # Set the timeout handler
            original_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Reset the alarm and restore original handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, original_handler)
            return result
        return wrapper
    return decorator

def black_scholes_price(S0, K, r, sigma, T):
    """Analytical Black–Scholes price for a European call option."""
    d1 = (np.log(S0/K) + (r+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    price = S0 * norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return price

@timeout(10, "Spectral method calculation took too long")
def spectral_method_price(S0, K, r, sigma, T, S_max=300, N=50):
    """
    Computes the European call option price via a spectral method
    (Chebyshev collocation) for the Black–Scholes PDE.
    """
    # Number of collocation points
    N_points = N + 1
    j = np.arange(0, N_points)
    # Chebyshev nodes in [-1, 1]
    x = np.cos(np.pi * j / N)
    # Map nodes to S: S = (x+1)/2 * S_max
    S_nodes = (x + 1) / 2 * S_max

    # --- Build the Chebyshev differentiation matrix ---
    D = np.zeros((N_points, N_points))
    # Set up the scaling factors c (with c[0] and c[-1] = 2, others = 1) and sign factors.
    c = np.ones(N_points)
    c[0] = 2
    c[-1] = 2
    c = c * ((-1)**j)
    for i in range(N_points):
        for k in range(N_points):
            if i != k:
                D[i, k] = (c[i]/c[k]) / (x[i] - x[k])
    # Diagonal entries (Trefethen's formula)
    for i in range(N_points):
        if i == 0:
            D[i, i] = (2 * N**2 + 1) / 6.0
        elif i == N:
            D[i, i] = -(2 * N**2 + 1) / 6.0
        else:
            D[i, i] = -x[i] / (2*(1 - x[i]**2))
    
    # Second derivative matrix in x
    D2 = np.dot(D, D)
    
    # Transform derivatives from x to S.
    D_S = (2 / S_max) * D
    D2_S = (2 / S_max)**2 * D2

    # --- Set up the initial condition (terminal condition for Black-Scholes) ---
    U0 = np.maximum(S_nodes - K, 0)

    # --- Define the ODE system in tau ---
    def ode_system(tau, U):
        U_S_val = D_S.dot(U)
        U_SS_val = D2_S.dot(U)
        a = 0.5 * sigma**2 * S_nodes**2
        b = r * S_nodes
        F = a * U_SS_val + b * U_S_val - r * U
        return F

    # Enforce the Dirichlet boundary conditions
    def ode_system_bc(tau, U):
        F = ode_system(tau, U)
        # Enforce boundary conditions:
        U[0] = 0  # U(0,tau)=0
        U[-1] = S_max - K * np.exp(-r * tau)  # U(S_max,tau)=S_max - K*exp(-r*tau)
        F[0] = 0
        F[-1] = 0
        return F

    # --- Solve the ODE system forward in tau from 0 to T ---
    try:
        # Adjust solver parameters for better convergence
        sol = solve_ivp(
            ode_system_bc, 
            [0, T], 
            U0, 
            method='BDF', 
            t_eval=[T],
            rtol=1e-4,  # Relaxed tolerance for faster convergence
            atol=1e-6,
            max_step=T/10  # Limit maximum step size
        )
        
        # Handle different possible structures of the solution
        if hasattr(sol, 'y') and isinstance(sol.y, np.ndarray) and sol.y.size > 0:
            if sol.y.ndim == 2 and sol.y.shape[1] > 0:
                U_final = sol.y[:, -1]
            else:
                # Fall back to accessing differently
                U_final = sol.y
        elif hasattr(sol, 'y_events') and sol.y_events and len(sol.y_events[0]) > 0:
            U_final = sol.y_events[0][-1]
        else:
            # If we can't extract the solution properly, raise an error
            raise ValueError("Could not extract solution from ODE solver result")
        
        # --- Interpolate to get the price at S0 ---
        price = np.interp(S0, S_nodes, U_final)
        return price
    except Exception as e:
        print(f"Error in spectral_method_price with parameters: S0={S0}, K={K}, r={r}, sigma={sigma}, T={T}")
        print(f"Error message: {str(e)}")
        # Return NaN in case of error
        return float('nan')

def run_simulation(S0, K, r, sigma, T, num_runs=10):
    """Run multiple simulations with the given parameters and collect statistics."""
    # Compute the analytical (Black–Scholes) price.
    true_price = black_scholes_price(S0, K, r, sigma, T)
    
    # Lists to collect results from each run.
    prices = []
    errors = []
    runtimes = []
    
    # Run the spectral method num_runs times.
    for i in range(num_runs):
        start_time = time.time()
        try:
            price_spec = spectral_method_price(S0, K, r, sigma, T, S_max=300, N=50)
            end_time = time.time()
            runtime_spec = end_time - start_time
            
            # If runtime is too long, treat as failed calculation
            if runtime_spec > 10:
                print(f"Run {i} took {runtime_spec:.2f}s, which exceeds the 10s threshold. Setting result to NaN.")
                price_spec = float('nan')
                error_spec = float('nan')
            else:
                error_spec = abs(price_spec - true_price) if not np.isnan(price_spec) else float('nan')
                
        except (TimeoutError, Exception) as e:
            end_time = time.time()
            runtime_spec = end_time - start_time
            print(f"Error in run {i} with parameters: S0={S0}, K={K}, r={r}, sigma={sigma}, T={T}")
            print(f"Error message: {str(e)}")
            price_spec = float('nan')
            error_spec = float('nan')
        
        prices.append(price_spec)
        errors.append(error_spec)
        runtimes.append(runtime_spec)
    
    # Convert lists to numpy arrays to compute statistics.
    prices = np.array(prices)
    errors = np.array(errors)
    runtimes = np.array(runtimes)
    
    # Compute mean and standard deviation for each quantity, handling NaN values
    valid_prices = prices[~np.isnan(prices)]
    valid_errors = errors[~np.isnan(errors)]
    valid_runtimes = runtimes[~np.isnan(runtimes)]
    
    if len(valid_prices) > 0:
        mean_price = np.mean(valid_prices)
        std_price = np.std(valid_prices)
    else:
        mean_price = float('nan')
        std_price = float('nan')
        
    if len(valid_errors) > 0:
        mean_error = np.mean(valid_errors)
        std_error = np.std(valid_errors)
    else:
        mean_error = float('nan')
        std_error = float('nan')
        
    if len(valid_runtimes) > 0:
        mean_runtime = np.mean(valid_runtimes)
        std_runtime = np.std(valid_runtimes)
    else:
        mean_runtime = float('nan')
        std_runtime = float('nan')
    
    # Compute mean and standard deviation for each quantity.
    results = {
        "S0": S0,
        "K": K,
        "r": r,
        "sigma": sigma,
        "T": T,
        "True Price": true_price,
        "Method": "Spectral Method",
        "Mean Price": mean_price,
        "Std Price": std_price,
        "Mean Error": mean_error, 
        "Std Error": std_error,
        "Mean Runtime": mean_runtime,
        "Std Runtime": std_runtime
    }
    
    return results

def main():
    # Define parameter ranges
    S0_values = [80, 100, 120]  # Stock price
    K_values = [90, 100, 110]   # Strike price
    r_values = [0.01, 0.05, 0.1]  # Risk-free rate
    sigma_values = [0.1, 0.2, 0.3]  # Volatility
    T_values = [0.5, 1.0, 2.0]  # Time to maturity in years
    
    # Number of runs for each parameter set
    num_runs = 10  # Lower value for demonstration; increase for more reliable stats
    
    # Create a list to store results
    all_results = []
    
    # Generate all combinations of parameters
    param_combinations = list(product(S0_values, K_values, r_values, sigma_values, T_values))
    total_combinations = len(param_combinations)
    
    print(f"Running simulation for {total_combinations} different parameter combinations...")
    
    # Loop through all parameter combinations
    for i, (S0, K, r, sigma, T) in enumerate(param_combinations):
        print(f"Processing combination {i+1}/{total_combinations}: S0={S0}, K={K}, r={r}, sigma={sigma}, T={T}")
        try:
            results = run_simulation(S0, K, r, sigma, T, num_runs)
            all_results.append(results)
            
            # Save intermediate results every 10 combinations as backup
            if (i + 1) % 10 == 0:
                interim_df = pd.DataFrame(all_results)
                interim_df.to_csv(f"option_pricing_results_interim_{i+1}.csv", index=False)
                print(f"Saved interim results to option_pricing_results_interim_{i+1}.csv")
                
        except Exception as e:
            print(f"Failed processing combination {i+1}/{total_combinations}")
            print(f"Error message: {str(e)}")
            # Add a row with NaN values for this combination
            results = {
                "S0": S0,
                "K": K,
                "r": r,
                "sigma": sigma,
                "T": T,
                "True Price": black_scholes_price(S0, K, r, sigma, T),
                "Method": "Spectral Method",
                "Mean Price": float('nan'),
                "Std Price": float('nan'),
                "Mean Error": float('nan'),
                "Std Error": float('nan'),
                "Mean Runtime": float('nan'),
                "Std Runtime": float('nan')
            }
            all_results.append(results)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results to CSV
    results_df.to_csv("option_pricing_results_final.csv", index=False)
    
    # Print summary
    print("\nSimulation complete!")
    print(f"Results saved to 'option_pricing_results_final.csv'")
    
    # Display sample of the DataFrame
    print("\nSample of results DataFrame:")
    print(results_df.head())
    
    # Display summary statistics across all parameter combinations
    print("\nOverall Summary Statistics:")
    valid_errors = results_df['Mean Error'].dropna()
    valid_runtimes = results_df['Mean Runtime'].dropna()
    
    if len(valid_errors) > 0:
        print(f"Mean Price Error: {valid_errors.mean():.6f}")
    else:
        print("Mean Price Error: N/A (no valid data)")
        
    if len(valid_runtimes) > 0:
        print(f"Mean Runtime: {valid_runtimes.mean():.6f} seconds")
    else:
        print("Mean Runtime: N/A (no valid data)")

if __name__ == "__main__":
    main()