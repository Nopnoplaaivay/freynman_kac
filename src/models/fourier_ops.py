import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d
import time

# ---------------------------
# Analytical Black-Scholes Formula
# ---------------------------
def black_scholes_price(S, K, r, sigma, T):
    """
    Computes the Black–Scholes price for a European call option.
    S can be a scalar or a numpy array.
    """
    S = np.array(S)
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T) + 1e-8)
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return price

# ---------------------------
# Spectral Convolution for 1D
# ---------------------------
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        """
        1D Fourier layer. It does FFT, multiplies some Fourier modes by learned weights, and returns via inverse FFT.
        """
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # number of Fourier modes to keep
        self.scale = 1 / (in_channels * out_channels)
        # weights is a complex tensor: shape (in_channels, out_channels, modes)
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat))
    
    def compl_mul1d(self, input, weights):
        # input: (batch, in_channels, n_ft), weights: (in_channels, out_channels, modes)
        # returns: (batch, out_channels, modes)
        return torch.einsum("bix, iox -> box", input, weights)
    
    def forward(self, x):
        """
        x: shape (batch, in_channels, n)
        """
        batchsize = x.shape[0]
        n = x.shape[-1]
        # Compute Fourier transform (rfft returns half-spectrum)
        x_ft = torch.fft.rfft(x)  # shape (batch, in_channels, n//2 + 1)
        # Allocate output in Fourier space, same shape as x_ft
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[-1], device=x.device, dtype=torch.cfloat)
        # Multiply only the first self.modes frequencies
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)
        # Return inverse FFT to get back to physical space
        x = torch.fft.irfft(out_ft, n=n)
        return x

# ---------------------------
# Fourier Neural Operator (FNO) for 1D
# ---------------------------
class FNO1d(nn.Module):
    def __init__(self, modes, width, layers=4):
        """
        FNO that maps an input function defined on a 1D grid to an output function.
        The input is augmented with the spatial coordinate.
        - modes: number of Fourier modes to keep in each spectral layer.
        - width: number of channels (feature dimension) in the lifted space.
        - layers: number of Fourier layers.
        """
        super(FNO1d, self).__init__()
        self.modes = modes
        self.width = width
        self.layers = layers
        # Lift the input (which has 2 channels: function value and coordinate) to 'width' channels.
        self.fc0 = nn.Linear(2, width)
        self.spectral_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        for _ in range(layers):
            self.spectral_layers.append(SpectralConv1d(width, width, modes))
            self.w_layers.append(nn.Conv1d(width, width, 1))
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        """
        x: input tensor of shape (batch, n), representing the discretized function (e.g., terminal payoff) on the grid.
        We also append the normalized spatial coordinate.
        """
        batchsize, n = x.shape
        # Create a grid of coordinates normalized to [0,1]
        grid = torch.linspace(0, 1, n, device=x.device).unsqueeze(0).repeat(batchsize, 1)  # shape (batch, n)
        # Concatenate input function and grid as features.
        x = x.unsqueeze(-1)  # shape (batch, n, 1)
        inp = torch.cat([x, grid.unsqueeze(-1)], dim=-1)  # shape (batch, n, 2)
        # Lift to higher dimension
        x = self.fc0(inp)  # shape (batch, n, width)
        x = x.permute(0, 2, 1)  # shape (batch, width, n)
        # Apply Fourier layers with residual connections.
        for i in range(self.layers):
            x1 = self.spectral_layers[i](x)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            x = self.activation(x)
        x = x.permute(0, 2, 1)  # shape (batch, n, width)
        x = self.fc1(x)         # shape (batch, n, 128)
        x = self.activation(x)
        x = self.fc2(x)         # shape (batch, n, 1)
        x = x.squeeze(-1)       # shape (batch, n)
        return x

# ---------------------------
# Training Routine for FNO
# ---------------------------
def train_fno(num_samples=1000, epochs=5000, lr=1e-3, m=100, modes=16, width=64, layers=4, S_max=300, r=0.05, sigma=0.2, T=1.0):
    """
    Train an FNO to learn the mapping from the terminal payoff function
      g(S) = max(S - K, 0)
    to the solution function V(S) (given by the Black–Scholes formula) on the domain [0,S_max].
    
    Training data:
      - Sample strikes K uniformly from [80, 120].
      - For each sample, compute the terminal payoff function g(S) on a grid of m points over [0, S_max].
      - Compute the target solution V(S) on the same grid using the Black–Scholes formula.
    
    Returns:
      - Trained FNO model.
      - The grid S_grid (as a numpy array) used for discretization.
    """
    device = torch.device("cpu")
    # Create a fixed spatial grid on [0, S_max]
    S_grid = np.linspace(0, S_max, m)
    
    # Generate training data by sampling strikes in [80, 120]
    branch_inputs = []  # terminal payoff functions, shape (num_samples, m)
    targets = []        # corresponding Black–Scholes solutions, shape (num_samples, m)
    strikes = np.random.uniform(80, 120, num_samples)
    for K_val in strikes:
        g = np.maximum(S_grid - K_val, 0)  # payoff function
        branch_inputs.append(g)
        V = black_scholes_price(S_grid, K_val, r, sigma, T)
        targets.append(V)
    branch_inputs = np.array(branch_inputs)  # shape (num_samples, m)
    targets = np.array(targets)              # shape (num_samples, m)
    
    # Convert training data to torch tensors
    X = torch.tensor(branch_inputs, dtype=torch.float32, device=device)
    Y = torch.tensor(targets, dtype=torch.float32, device=device)
    
    model = FNO1d(modes, width, layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X)  # shape (num_samples, m)
        loss = loss_fn(pred, Y)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"Epoch {epoch:5d}, Loss: {loss.item():.6f}")
    return model, S_grid


