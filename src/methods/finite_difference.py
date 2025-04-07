import numpy as np

def finite_difference_price(S0, K, r, sigma, T, S_max=300, M=200, N=200):
    # Set up spatial grid
    ds = S_max / M
    dt = T / N
    S_grid = np.linspace(0, S_max, M+1)
    # Terminal condition: payoff at T
    V = np.maximum(S_grid - K, 0)
    # Precompute coefficients for interior nodes
    A = np.zeros((M-1, M-1))
    B = np.zeros((M-1, M-1))
    for i in range(1, M):
        S = i * ds
        a = 0.25 * dt * (sigma**2 * (i**2) - r * i)
        b = -0.5 * dt * (sigma**2 * (i**2) + r)
        c = 0.25 * dt * (sigma**2 * (i**2) + r * i)
        idx = i - 1
        if idx > 0:
            A[idx, idx-1] = -a
            B[idx, idx-1] = a
        A[idx, idx] = 1 - b
        B[idx, idx] = 1 + b
        if idx < M-2:
            A[idx, idx+1] = -c
            B[idx, idx+1] = c
    # Time-stepping (backward in time)
    for j in range(N):
        V_interior = V[1:M]
        RHS = B.dot(V_interior)
        t = T - j * dt
        # Incorporate boundary conditions:
        # V(0)=0 and V(S_max)=S_max - K*exp(-r*(T-t))
        RHS[0] += 0  # since V(0)=0
        RHS[-1] += (0.25*dt*(sigma**2 * (M**2) + r*M)) * (S_max - K*np.exp(-r*t))
        V[1:M] = np.linalg.solve(A, RHS)
    price = np.interp(S0, S_grid, V)
    return price