import torch
import numpy as np
from dataclasses import dataclass

# Ορίζουμε το device κεντρικά
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class MFGConfig:
    name: str = "Default"
    T_max: float = 5.0
    dt: float = 0.01
    dx: float = 0.1
    sigma: float = 1.0
    epsilon: float = 1.0
    alpha_term: float = 2.0
    beta_term: float = 1.0
    s_min: float = -4.0
    s_max: float = 4.0
    m0_mean: float = 0.1
    m0_std: float = 0.11
    activation: str = "silu"
    #activation: str = "relu"
class MFGEngine:
    def __init__(self, cfg: MFGConfig):
        self.cfg = cfg
        self.T = np.arange(0, cfg.T_max + cfg.dt, cfg.dt)
        self.S = np.arange(cfg.s_min, cfg.s_max + cfg.dx, cfg.dx)
        self.A = np.linspace(-3, 3, 50)
        self.Nt = len(self.T)
        self.Ns = len(self.S)
        self.device = device

    def get_initial_distribution(self):
        m0 = np.exp(-0.5 * (self.S - self.cfg.m0_mean)**2 / (self.cfg.m0_std**2))
        return m0 / (np.sum(m0) * self.cfg.dx)
