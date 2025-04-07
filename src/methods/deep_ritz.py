import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def deep_ritz_price(S0, K, r, sigma, T, **kwargs):
    """
    Solves the stationary Black–Scholes ODE for a European call option using a Deep Ritz approach.
    
    We solve:
      0.5 * sigma^2 * S^2 * V''(S) + r * S * V'(S) - r * V(S) = 0,  for S in (0, S_max),
    with boundary conditions
      V(0) = 0,  V(S_max) = S_max - K.
    
    We construct a trial solution of the form:
      V(S) = A(S) + B(S) * N(S; theta),
    where A(S) = (S/S_max)*(S_max-K) and B(S) = S*(S_max-S) so that the boundary conditions are satisfied.
    
    The loss function is the mean squared PDE residual at collocation points.
    
    Parameters (via kwargs):
      - S_max: maximum asset price (default 300)
      - num_collocation: number of collocation points (default 1000)
      - num_epochs: number of training epochs (default 5000)
      - lr: learning rate (default 1e-3)
    
    Returns:
      - Approximate option price at S = S0.
    """
    device = torch.device("cpu")
    
    S_max = kwargs.get("S_max", 300.0)
    num_collocation = kwargs.get("num_collocation", 1000)
    num_epochs = kwargs.get("num_epochs", 5000)
    lr = kwargs.get("lr", 1e-3)
    
    # Define the neural network N(S; theta)
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(1, 50),
                nn.Tanh(),
                nn.Linear(50, 50),
                nn.Tanh(),
                nn.Linear(50, 1)
            )
        def forward(self, x):
            return self.net(x)
    
    net = Net().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    # Define the trial solution: V(S) = A(S) + B(S)*N(S)
    def trial_solution(S):
        # S is a tensor of shape (N, 1)
        A = (S / S_max) * (S_max - K)  # satisfies V(0)=0, V(S_max)=S_max-K
        B = S * (S_max - S)            # vanishes at S=0 and S=S_max
        return A + B * net(S)
    
    # Generate collocation points in (0, S_max)
    S_colloc = torch.linspace(0.0, S_max, num_collocation, device=device).view(-1, 1)
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # Enable gradient computation for S_colloc (needed for derivatives)
        S_colloc.requires_grad = True
        V_pred = trial_solution(S_colloc)  # shape (num_collocation, 1)
        
        # First derivative dV/dS
        grad_V = torch.autograd.grad(V_pred, S_colloc, grad_outputs=torch.ones_like(V_pred),
                                     create_graph=True, retain_graph=True)[0]
        # Second derivative d^2V/dS^2
        grad_V2 = torch.autograd.grad(grad_V, S_colloc, grad_outputs=torch.ones_like(grad_V),
                                      create_graph=True, retain_graph=True)[0]
        
        # PDE residual: R(S) = 0.5 * sigma^2 * S^2 * V'' + r * S * V' - r * V
        residual = 0.5 * sigma**2 * S_colloc**2 * grad_V2 + r * S_colloc * grad_V - r * V_pred
        
        loss = torch.mean(residual**2)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:5d}, Loss: {loss.item():.6f}")
    
    # Evaluate the trained network at S = S0
    S0_tensor = torch.tensor([[S0]], device=device, dtype=torch.float32)
    price = trial_solution(S0_tensor).item()
    return price