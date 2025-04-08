import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import norm

# ------------------------------
# Analytical Black–Scholes Formula
# ------------------------------
def black_scholes_price(S, K, r, sigma, T):
    """
    Computes the Black–Scholes price for a European call option.
    S can be a scalar or a numpy array.
    """
    S = np.array(S)
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T) + 1e-8)
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

# ------------------------------
# Define the Branch and Trunk Networks
# ------------------------------
class BranchNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=50):
        super(BranchNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )
    def forward(self, x):
        return self.net(x)

class TrunkNet(nn.Module):
    def __init__(self, output_size, hidden_dim=50):
        super(TrunkNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )
    def forward(self, x):
        return self.net(x)

class DeepONet(nn.Module):
    def __init__(self, branch_input_size, output_size, hidden_dim=50):
        """
        branch_input_size: Dimension of the discretized payoff function.
        output_size: Dimension p of the latent representations.
        """
        super(DeepONet, self).__init__()
        self.branch_net = BranchNet(branch_input_size, output_size, hidden_dim)
        self.trunk_net = TrunkNet(output_size, hidden_dim)
    def forward(self, branch_input, trunk_input):
        # branch_input: shape (batch_size, branch_input_size)
        # trunk_input: shape (batch_size, 1)
        branch_out = self.branch_net(branch_input)  # shape (batch_size, p)
        trunk_out = self.trunk_net(trunk_input)       # shape (batch_size, p)
        # Dot product (inner product) along the latent dimension:
        output = torch.sum(branch_out * trunk_out, dim=1, keepdim=True)
        return output

# ------------------------------
# Training Routine for DeepONet
# ------------------------------
def train_deeponet(num_samples=1000, epochs=5000, lr=1e-3, m=100, p=50, S_max=300, r=0.05, sigma=0.2, T=1.0):
    """
    Train a DeepONet to learn the mapping from the terminal payoff function
      g(S) = max(S - K, 0)
    to the solution V(S) given by the Black–Scholes formula.
    
    The branch input is the discretized payoff function on a grid of m points in [0, S_max].
    The trunk input is the query point S.
    
    Training data:
      - Sample K uniformly from [80, 120].
      - For each K, compute the payoff function g(S) and target function V(S) on S_grid.
      - For each training sample, randomly select a query point from S_grid.
    
    Returns:
      - Trained DeepONet model.
      - The grid S_grid (as a numpy array) used for discretizing the payoff function.
    """
    device = torch.device("cpu")
    
    # Create a fixed spatial grid for the branch input (discretization of S)
    S_grid = np.linspace(0, S_max, m)
    
    # Generate training data by sampling strikes K in [80, 120]
    Ks = np.random.uniform(80, 120, num_samples)
    branch_inputs = []    # each entry is g(S) = max(S-K, 0) on S_grid, shape (m,)
    target_functions = [] # corresponding target V(S) on S_grid (from Black-Scholes)
    
    for K_val in Ks:
        g = np.maximum(S_grid - K_val, 0)  # payoff function
        branch_inputs.append(g)
        # Compute target: V(S) on S_grid using Black–Scholes formula for given K_val
        V = black_scholes_price(S_grid, K_val, r, sigma, T)
        target_functions.append(V)
    
    branch_inputs = np.array(branch_inputs)         # shape (num_samples, m)
    target_functions = np.array(target_functions)       # shape (num_samples, m)
    
    # For training, for each sample pick a random query point from the grid
    query_indices = np.random.randint(0, m, size=num_samples)
    trunk_inputs = S_grid[query_indices].reshape(-1, 1)  # shape (num_samples, 1)
    # Targets: the corresponding V(S) values at the chosen query points
    targets = np.array([target_functions[i, query_indices[i]] for i in range(num_samples)]).reshape(-1, 1)
    
    # Convert training data to torch tensors
    branch_inputs_tensor = torch.tensor(branch_inputs, dtype=torch.float32, device=device)
    trunk_inputs_tensor = torch.tensor(trunk_inputs, dtype=torch.float32, device=device)
    targets_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
    
    # Initialize DeepONet model
    model = DeepONet(branch_input_size=m, output_size=p, hidden_dim=50).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(branch_inputs_tensor, trunk_inputs_tensor)  # shape (num_samples, 1)
        loss = loss_fn(outputs, targets_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"Epoch {epoch:5d}, Loss: {loss.item():.6f}")
    
    return model, S_grid

# ------------------------------
# DeepONet Price Prediction Function
# ------------------------------
def deeponet_price(S0, K, r, sigma, T, **kwargs):
    """
    Predict the option price at S = S0 for a given strike K using DeepONet.
    This function trains DeepONet on synthetic data mapping from the payoff function
    g(S)=max(S-K,0) to the solution V(S) (computed via the Black–Scholes formula) and then
    evaluates the trained network at the query point S0.
    
    Hyperparameters (can be passed via kwargs):
      - num_samples: number of training samples (default 1000)
      - epochs: number of training epochs (default 5000)
      - lr: learning rate (default 1e-3)
      - m: number of discretization points for payoff function (default 100)
      - p: latent dimension (default 50)
      - S_max: maximum asset price for the grid (default 300)
    
    Returns:
      - Predicted option price at S = S0.
    """
    num_samples = kwargs.get("num_samples", 1000)
    epochs = kwargs.get("epochs", 5000)
    lr = kwargs.get("lr", 1e-3)
    m = kwargs.get("m", 100)
    p = kwargs.get("p", 50)
    S_max = kwargs.get("S_max", 300)
    
    print("Training DeepONet...")
    model, S_grid = train_deeponet(num_samples=num_samples, epochs=epochs, lr=lr,
                                    m=m, p=p, S_max=S_max, r=r, sigma=sigma, T=T)
    
    # Build branch input for the given strike K: g(S)=max(S-K,0) on S_grid
    branch_input = np.maximum(S_grid - K, 0).reshape(1, -1)  # shape (1, m)
    branch_input_tensor = torch.tensor(branch_input, dtype=torch.float32)
    
    # Build trunk input for the query point S0
    trunk_input = np.array([[S0]], dtype=np.float32)  # shape (1, 1)
    trunk_input_tensor = torch.tensor(trunk_input, dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        price_tensor = model(branch_input_tensor, trunk_input_tensor)
    price = price_tensor.item()
    return price