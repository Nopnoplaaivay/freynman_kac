import numpy as np
from scipy.integrate import solve_ivp

# =============================================================================
# 6. Method of Lines (MOL)
# =============================================================================
def method_of_lines_price(S0, K, r, sigma, T, S_max=300, M=200):
    ds = S_max / M
    S_grid = np.linspace(0, S_max, M+1)
    # Terminal condition
    V_T = np.maximum(S_grid - K, 0)
    def ode_system(t, V):
        dVdt = np.zeros_like(V)
        # Use central finite differences for interior points
        for i in range(1, M):
            S = S_grid[i]
            dVdx = (V[i+1] - V[i-1]) / (2*ds)
            d2Vdx2 = (V[i+1] - 2*V[i] + V[i-1]) / (ds**2)
            dVdt[i] = -0.5 * sigma**2 * S**2 * d2Vdx2 - r * S * dVdx + r * V[i]
        dVdt[0] = 0   # Boundary at S=0
        dVdt[M] = -r * V[M]  # Approximate boundary at S_max
        return dVdt
    sol = solve_ivp(ode_system, [T, 0], V_T, method='RK45', t_eval=[0])
    V0 = sol.y[:, -1]
    price = np.interp(S0, S_grid, V0)
    return price