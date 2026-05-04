import torch
import torch.nn as nn
import numpy as np

class Actor(nn.Module):
    def __init__(self, action_dim, T_max):
        super().__init__()
        self.T_max = T_max
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.LayerNorm(64), nn.SiLU(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.SiLU(),
            nn.Linear(64, 1)
        )
        self.xi_mu = nn.Parameter(torch.tensor(0.0))
        self.xi_log_sigma = nn.Parameter(torch.tensor(np.log(0.2)))
        self.log_theta = nn.Parameter(torch.tensor(np.log(0.4)))
        self.register_buffer('actions', torch.linspace(-3, 3, action_dim))

    def sample_xi(self, n_samples=1):
        sigma = torch.exp(self.xi_log_sigma)
        epsilon = torch.randn(n_samples, 1, device=self.xi_mu.device)
        return self.xi_mu + sigma * epsilon

    def forward(self, tx, xi_val):
        t, x = tx[:, 0:1], tx[:, 1:2]
        if xi_val.shape[0] != t.shape[0]:
            xi_val = xi_val.expand(t.shape[0], 1)
        inp = torch.cat([t, x, xi_val], dim=1)
        mu = torch.clamp(self.net(inp), -3.0, 3.0)
        theta = torch.exp(self.log_theta) + 1e-4
        sq_diff = (self.actions.unsqueeze(0) - mu) ** 2
        exponent = -sq_diff / (2 * theta ** 2)
        return torch.exp(exponent - torch.logsumexp(exponent, dim=1, keepdim=True))
