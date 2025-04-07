import numpy as np
import time
from math import exp, log, sqrt
from scipy.stats import norm
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Define Black–Scholes Model Parameters
# ------------------------------------------------------------------------------
S0 = 100         # initial asset price
K = 100          # strike price
r = 0.05         # risk-free rate
T = 1.0          # time to maturity
sigma = 0.2      # volatility

# ------------------------------------------------------------------------------
# Analytical Black–Scholes Price Function
# ------------------------------------------------------------------------------
def black_scholes_price(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    price = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return price

# ------------------------------------------------------------------------------
# Quantum Circuit: Two-Step Binomial Model Encoding of an SDE
# ------------------------------------------------------------------------------
# We assume a simple two–step binomial model with the following outcomes:
#   - Outcome corresponding to |00> ("down,down"): probability 0.25, payoff = max(81-100,0) = 0.
#   - Outcome corresponding to |01> ("up,down"): probability 0.50, payoff = max(99-100,0) = 0.
#   - Outcome corresponding to |10> ("up,up"): probability 0.25, payoff = max(121-100,0) = 21.
#   - Outcome |11> is unused.
#
# Therefore, we want to prepare the state:
#   |ψ> = 0.5 |00> + 0.7071 |01> + 0.5 |10> + 0 |11>.
#
# Define the raw state vector.
raw_state = np.array([0.7071, 0.7071, 0.7071, 0.0])
# Normalize the state explicitly (to avoid floating-point issues).
norm_state = raw_state / np.linalg.norm(raw_state)

# Create a quantum circuit on 2 qubits.
qc = QuantumCircuit(2)
qc.initialize(norm_state, [0, 1])
qc.measure_all()

# ------------------------------------------------------------------------------
# Draw and Plot the Quantum Circuit
# ------------------------------------------------------------------------------
print("Text Representation of the Quantum Circuit:")
print(qc.draw(output='text'))

circuit_diagram = qc.draw(output='mpl')
plt.title("Quantum Circuit Encoding a Two-Step Binomial SDE")
plt.show()

# ------------------------------------------------------------------------------
# Simulate the Quantum Circuit
# ------------------------------------------------------------------------------
shots = 10000
backend = Aer.get_backend("qasm_simulator")
start_time_qc = time.time()
job = execute(qc, backend, shots=shots)
result = job.result()
counts = result.get_counts(qc)
end_time_qc = time.time()
runtime_qc = end_time_qc - start_time_qc

# ------------------------------------------------------------------------------
# Compute Option Price from the Quantum Simulation (Binomial Model)
# ------------------------------------------------------------------------------
# In our encoding, we assume that only the branch |10> (representing "up,up")
# yields a nonzero payoff (which is 21).
# Therefore, the estimated probability for the up–up branch is:
prob_upup = counts.get('10', 0) / shots

# The binomial model price (under risk-neutral pricing) is then given by:
# V_binomial = exp(-r*T) * (payoff) * (probability)
price_quantum = exp(-r * T) * 21 * prob_upup

# ------------------------------------------------------------------------------
# Compute the Analytic Black–Scholes Price
# ------------------------------------------------------------------------------
start_time_bs = time.time()
price_analytic = black_scholes_price(S0, K, r, sigma, T)
end_time_bs = time.time()
runtime_bs = end_time_bs - start_time_bs

# ------------------------------------------------------------------------------
# Print the Comparison Results
# ------------------------------------------------------------------------------
abs_error = abs(price_quantum - price_analytic)

print("\nBenchmark Comparison:")
print(f"{'Method':40s} {'Price':>10s} {'Abs Error':>12s} {'Runtime (sec)':>15s}")
print(f"{'Quantum Simulation (2-step binomial)':40s} {price_quantum:10.4f} {abs_error:12.4f} {runtime_qc:15.4f}")
print(f"{'Analytic Black-Scholes':40s} {price_analytic:10.4f} {0.0:12.4f} {runtime_bs:15.4f}")
