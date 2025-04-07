import numpy as np
import time
from math import exp, log, sqrt
from scipy.optimize import minimize
from scipy.stats import norm
from qiskit import QuantumCircuit, Aer, execute
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

# Compute the target (analytic) Black–Scholes price.
target_price = black_scholes_price(S0, K, r, sigma, T)
print("Target (Black–Scholes) Price: {:.12f}".format(target_price))

# ------------------------------------------------------------------------------
# Quantum Price Prediction Function
# ------------------------------------------------------------------------------
def predicted_price(state, r, T, payoff=21):
    """
    Given a (normalized) state vector of 4 real amplitudes for a 2–qubit system,
    assume that the outcome |10> (index 2) is associated with a nonzero payoff.
    The predicted price is:
         V = exp(-r*T) * payoff * (|a_2|^2).
    """
    discount = exp(-r * T)
    return discount * payoff * (state[2] ** 2)

# ------------------------------------------------------------------------------
# Optimization Setup: Optimize the Raw Quantum State
# ------------------------------------------------------------------------------
# Objective: minimize the squared error between the predicted price and target price.
def objective(x):
    price = predicted_price(x, r, T, payoff=21)
    return (price - target_price) ** 2

# Constraint: state vector must be normalized: sum(x_i^2) == 1.
def norm_constraint(x):
    return np.sum(x**2) - 1.0

constraints = ({'type': 'eq', 'fun': norm_constraint})
# Set bounds for each amplitude. (Here we use a wider bound, e.g. angles, if needed.)
bounds = [(-2*np.pi, 2*np.pi)] * 4

# Initial guess for the raw state vector.

def quantum_price_simulation(state, r, T, shots=100):
    """
    Simulate the quantum circuit with the given state vector and return the predicted price.
    """
    state = np.array([0.7071, 0.7071, 0.0, 0.7071])
    result = minimize(objective, state, method='SLSQP', bounds=bounds,
                      constraints=constraints, options={'ftol': 1e-12})

    # Normalize the optimized state (should be nearly normalized already).
    opt_state = result.x / np.linalg.norm(result.x)

    # Create a quantum circuit with 2 qubits.
    qc = QuantumCircuit(2)
    qc.initialize(opt_state, [0, 1])
    qc.measure_all()

    # Execute the circuit on a qasm simulator backend.
    backend = Aer.get_backend("qasm_simulator")
    job = execute(qc, backend, shots=shots)
    result = job.result()
    
    # Get the counts of measurement outcomes.
    counts = result.get_counts(qc)
    
    # In our encoding, only outcome |10> yields a payoff.
    prob_upup = counts.get('10', 0) / shots
    price_sim = exp(-r * T) * 21 * prob_upup
    return price_sim