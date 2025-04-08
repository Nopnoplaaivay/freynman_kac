import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train_deep_ritz(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, **kwargs):
    device = torch.device("cpu")

    S_max = kwargs.get("S_max", 300.0)
    num_collocation = kwargs.get("num_collocation", 1000)
    num_epochs = kwargs.get("num_epochs", 5000)
    lr = kwargs.get("lr", 1e-3)

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

    def trial_solution(S, net_out):
        A = (S / S_max) * (S_max - K)
        B = S * (S_max - S)
        return A + B * net_out

    def pde_residual(S_tensor):
        S_tensor.requires_grad_(True)
        net_out = net(S_tensor)
        V = trial_solution(S_tensor, net_out)

        dV_dS = torch.autograd.grad(V, S_tensor, grad_outputs=torch.ones_like(V), create_graph=True)[0]
        d2V_dS2 = torch.autograd.grad(dV_dS, S_tensor, grad_outputs=torch.ones_like(dV_dS), create_graph=True)[0]

        res = 0.5 * sigma**2 * S_tensor**2 * d2V_dS2 + r * S_tensor * dV_dS - r * V
        return res

    S_train = torch.linspace(1e-5, S_max, num_collocation).view(-1, 1).to(device)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        residual = pde_residual(S_train)
        loss = torch.mean(residual**2)
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    save_path = "src/models/weights/deep_ritz_weights.pth" 
    torch.save(net.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")
