import numpy as np

# =============================================================================
# 2. Monte Carlo Simulation
# =============================================================================
def monte_carlo_price(S0, K, r, sigma, T, N=1000000):
    Z = np.random.randn(N)
    ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoff = np.maximum(ST - K, 0)
    price = np.exp(-r*T) * np.mean(payoff)
    return price