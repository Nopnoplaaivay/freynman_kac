import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import norm
import time

def black_scholes_price(S0, K, r, sigma, T):
    """Analytical Black–Scholes price for a European call option."""
    d1 = (np.log(S0/K) + (r+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    price = S0 * norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return price

def spectral_method_price(S0, K, r, sigma, T, S_max=300, N=50):
    """
    Computes the European call option price via a spectral method
    (Chebyshev collocation) for the Black–Scholes PDE.

    The Black–Scholes PDE is:
      V_t + 0.5 * sigma^2 * S^2 * V_SS + r * S * V_S - r V = 0,
      V(S,T) = max(S - K, 0),
      with boundary conditions V(0,t)=0 and V(S_max,t)=S_max - K*exp(-r*(T-t)).

    We transform to the variable tau = T-t so that the PDE becomes
      U_tau = a(S) U_SS + b(S) U_S - r U,
    with U(S,0)= payoff = max(S-K,0) and
      U(0, tau)=0, U(S_max, tau)=S_max - K * exp(-r*tau).
    
    We map S in [0,S_max] to x in [-1,1] by S = (x+1)/2 * S_max, and
    use N+1 Chebyshev collocation points.
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
    # Since S = (x+1)/2 * S_max, we have d/dS = (2/S_max) d/dx and d^2/dS^2 = (2/S_max)^2 d^2/dx^2.
    D_S = (2 / S_max) * D
    D2_S = (2 / S_max)**2 * D2

    # --- Set up the initial condition (terminal condition for Black-Scholes) ---
    # In variable tau = T - t, initial condition at tau=0 is U(S, 0) = max(S-K,0)
    U0 = np.maximum(S_nodes - K, 0)

    # --- Define the ODE system in tau ---
    # The transformed PDE is:
    #   U_tau = a(S) * U_SS + b(S) * U_S - r * U,
    # with a(S)=0.5*sigma^2*S^2, b(S)=r*S.
    def ode_system(tau, U):
        U_S_val = D_S.dot(U)
        U_SS_val = D2_S.dot(U)
        a = 0.5 * sigma**2 * S_nodes**2
        b = r * S_nodes
        F = a * U_SS_val + b * U_S_val - r * U
        # (Interior nodes: i = 1,...,N-1; boundaries are handled below.)
        return F

    # To enforce the Dirichlet boundary conditions at every time step,
    # we define a wrapper that resets the boundary values.
    def ode_system_bc(tau, U):
        F = ode_system(tau, U)
        # Enforce boundary conditions:
        U[0] = 0  # U(0,tau)=0
        U[-1] = S_max - K * np.exp(-r * tau)  # U(S_max,tau)=S_max - K*exp(-r*tau)
        F[0] = 0
        F[-1] = 0
        return F

    # --- Solve the ODE system forward in tau from 0 to T ---
    sol = solve_ivp(ode_system_bc, [0, T], U0, method='BDF', t_eval=[T])
    U_final = sol.y[:, -1]  # solution at tau=T, which corresponds to t=0



    # --- Interpolate to get the price at S0 ---
    price = np.interp(S0, S_nodes, U_final)
    # price = np.interp(S0, S_nodes_sorted, U_final_sorted)
    return price

# =============================================================================
# Example Benchmark Comparison for the Spectral Method
# =============================================================================
import time
import numpy as np

def main():

    n_sets = 100  # Number of parameter sets to generate
    np.random.seed(42)  # For reproducibility

    # Generate random parameter sets
    S0_values = np.random.uniform(90, 110, n_sets)  # Stock prices between 80 and 120
    K_values = np.random.uniform(90, 110, n_sets)   # Strike prices between 80 and 120
    r_values = np.random.uniform(0.01, 0.1, n_sets) # Interest rates between 1% and 10%
    sigma_values = np.random.uniform(0.1, 0.4, n_sets) # Volatilities between 10% and 40%
    T_values = np.random.uniform(0.25, 2.0, n_sets)   # Times to maturity between 3 months and 2 years

    for i in range(n_sets):
        # S0 = S0_values[i]
        # K = K_values[i]
        # r = r_values[i]
        # sigma = sigma_values[i]
        # T = T_values[i]

        S0 = 100
        K = 100
        r = 0.05
        sigma = 0.2
        T = 1.0
        S_max = 300

        # Compute the analytical (Black–Scholes) price.
        true_price = black_scholes_price(S0, K, r, sigma, T)
        print(f"Set {i+1}: S0={S0:.2f}, K={K:.2f}, r={r:.4f}, sigma={sigma:.4f}, T={T:.2f}")
        print("Black–Scholes closed-form price: {:.4f}".format(true_price))


        num_runs = 100

        # Lists to collect results from each run.
        prices = []
        errors = []
        runtimes = []

        # Run the spectral method num_runs times.
        for idx in range(num_runs):
            start_time = time.time()
            price_spec = spectral_method_price(S0, K, r, sigma, T, S_max=300, N=50)
            end_time = time.time()

            runtime_spec = end_time - start_time
            error_spec = abs(price_spec - true_price)

            prices.append(price_spec)
            errors.append(error_spec)
            runtimes.append(runtime_spec)

        # Convert lists to numpy arrays to compute statistics.
        prices = np.array(prices)
        errors = np.array(errors)
        runtimes = np.array(runtimes)

        # Compute mean and standard deviation for each quantity.
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        mean_runtime = np.mean(runtimes)
        std_runtime = np.std(runtimes)

        # Print the statistics.
        print("\nSpectral Method Statistics over {} runs:".format(num_runs))
        print(f"Mean Predicted Price: {mean_price:.4f}")
        print(f"Std Predicted Price: {std_price:.4f}")
        print(f"Mean Absolute Error: {mean_error:.4f}")
        print(f"Std Absolute Error: {std_error:.4f}")
        print(f"Mean Runtime: {mean_runtime:.4f} seconds")
        print(f"Std Runtime: {std_runtime:.4f} seconds")
        print("-"*50)


    
if __name__ == "__main__":
    main()
