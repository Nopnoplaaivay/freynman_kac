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
initial_state = np.array([0.7071, 0.7071, 0.0, 0.7071])

# ------------------------------------------------------------------------------
# Run Loop: Optimize and Simulate the Quantum Circuit
# ------------------------------------------------------------------------------
num_runs = 10  # Adjust the number of runs as needed

# Lists for optimization results.
predicted_prices = []
opt_abs_errors = []
opt_runtimes = []
optimized_states = []

# Lists for quantum simulation results.
simulated_prices = []
sim_abs_errors = []
sim_runtimes = []

# Set simulation parameters.
shots = 100
backend = Aer.get_backend("qasm_simulator")

for i in range(num_runs):
    # --- Optimization ---
    start_opt = time.time()
    result = minimize(objective, initial_state, method='SLSQP', bounds=bounds,
                      constraints=constraints, options={'ftol': 1e-12})
    elapsed_opt = time.time() - start_opt

    # Normalize the optimized state (should be nearly normalized already).
    opt_state = result.x / np.linalg.norm(result.x)
    price_opt = predicted_price(opt_state, r, T, payoff=21)
    error_opt = abs(price_opt - target_price)

    predicted_prices.append(price_opt)
    opt_abs_errors.append(error_opt)
    opt_runtimes.append(elapsed_opt)
    optimized_states.append(opt_state)

    # --- Quantum Circuit Simulation ---
    qc = QuantumCircuit(2)
    qc.initialize(opt_state, [0, 1])
    qc.measure_all()
    
    start_sim = time.time()
    job = execute(qc, backend, shots=shots)
    result_sim = job.result()
    counts = result_sim.get_counts(qc)
    elapsed_sim = time.time() - start_sim

    # In our encoding, only outcome |10> yields a payoff.
    prob_upup = counts.get('10', 0) / shots
    price_sim = exp(-r * T) * 21 * prob_upup
    error_sim = abs(price_sim - target_price)
    
    simulated_prices.append(price_sim)
    sim_abs_errors.append(error_sim)
    sim_runtimes.append(elapsed_sim)

# Convert lists to NumPy arrays for aggregated statistics.
predicted_prices = np.array(predicted_prices)
opt_abs_errors = np.array(opt_abs_errors)
opt_runtimes = np.array(opt_runtimes)

simulated_prices = np.array(simulated_prices)
sim_abs_errors = np.array(sim_abs_errors)
sim_runtimes = np.array(sim_runtimes)

# Aggregated statistics for optimization.
mean_opt_price = np.mean(predicted_prices)
std_opt_price = np.std(predicted_prices)
mean_opt_error = np.mean(opt_abs_errors)
std_opt_error = np.std(opt_abs_errors)
mean_opt_runtime = np.mean(opt_runtimes)
std_opt_runtime = np.std(opt_runtimes)

# Aggregated statistics for simulation.
mean_sim_price = np.mean(simulated_prices)
std_sim_price = np.std(simulated_prices)
mean_sim_error = np.mean(sim_abs_errors)
std_sim_error = np.std(sim_abs_errors)
mean_sim_runtime = np.mean(sim_runtimes)
std_sim_runtime = np.std(sim_runtimes)

# ------------------------------------------------------------------------------
# Print Aggregated Results
# ------------------------------------------------------------------------------
print("\nOptimization Results over {} runs:".format(num_runs))
print(f"Mean Predicted Price: {mean_opt_price:.12f}")
print(f"Std Predicted Price:  {std_opt_price:.12f}")
print(f"Mean Abs Error:       {mean_opt_error:.12f}")
print(f"Std Abs Error:        {std_opt_error:.12f}")
print(f"Mean Runtime:         {mean_opt_runtime:.12f} seconds")
print(f"Std Runtime:          {std_opt_runtime:.12f} seconds")

print("\nQuantum Simulation Results over {} runs:".format(num_runs))
print(f"Mean Simulated Price: {mean_sim_price:.12f}")
print(f"Std Simulated Price:  {std_sim_price:.12f}")
print(f"Mean Abs Error:       {mean_sim_error:.12f}")
print(f"Std Abs Error:        {std_sim_error:.12f}")
print(f"Mean Runtime:         {mean_sim_runtime:.12f} seconds")
print(f"Std Runtime:          {std_sim_runtime:.12f} seconds")

# ------------------------------------------------------------------------------
# (Optional) Display the Circuit from the First Run
# ------------------------------------------------------------------------------
best_state = optimized_states[0]
print("\nOptimized Quantum State Vector (from first run):")
for i, amp in enumerate(best_state):
    print(f"Amplitude for |{i:02b}>: {amp:.12f}, probability: {amp**2:.12f}")

qc_best = QuantumCircuit(2)
qc_best.initialize(best_state, [0, 1])
qc_best.measure_all()

print("\nText Representation of the Quantum Circuit (first run):")
print(qc_best.draw(output='text'))
circuit_diagram = qc_best.draw(output='mpl')
plt.title("Quantum Circuit with Optimized State (first run)")
plt.show()
