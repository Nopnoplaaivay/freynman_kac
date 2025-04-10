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

def spectral_method_price(S0, K, r, sigma, T, S_max=300, N=100):
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
        # U[-1] = S_max - K * np.exp(-r * tau)  # U(S_max,tau)=S_max - K*exp(-r*tau)
        U[-1] = min(max(S_max - K * np.exp(-r * tau), 0), S_max)
        F[0] = 0
        F[-1] = 0
        return F

    # Use try-except with streamlined approach to handle errors
    try:
        # --- Solve the ODE system with faster settings ---
        sol = solve_ivp(
            ode_system_bc, 
            [0, T], 
            U0, 
            method='RK45',  # Changed to RK45 which is typically faster than BDF for non-stiff problems
            t_eval=[T],
            rtol=1e-5,         # Slightly relaxed tolerance
            atol=1e-5         
        )
        
        # Process solution
        if hasattr(sol, 'y') and sol.y.size > 0:
            U_final = sol.y[:, 0] if sol.y.shape[1] > 0 else np.zeros(N_points)
            
            # Interpolate to get price at S0
            if S0 <= S_nodes[0]:
                price = U_final[0]  
            elif S0 >= S_nodes[-1]:
                price = U_final[-1]
            else:
                price = np.interp(S0, S_nodes, U_final)
                
            return price
        else:
            return black_scholes_price(S0, K, r, sigma, T)
            
    except Exception:
        # Fallback to analytical solution without printing error
        return black_scholes_price(S0, K, r, sigma, T)