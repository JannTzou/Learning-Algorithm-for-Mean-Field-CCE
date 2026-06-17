@dataclass(frozen=True)
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

# Μια κλάση/δομή για να κρατάμε τα στατικά δεδομένα (grid) που περνάμε στο JIT
class MFGEngineStatic:
    def __init__(self, cfg):
        self.cfg = cfg
        self.T = jnp.arange(0, cfg.T_max + cfg.dt, cfg.dt)
        self.S = jnp.arange(cfg.s_min, cfg.s_max + cfg.dx, cfg.dx)
        self.A = jnp.linspace(-3.0, 3.0, 50)
        self.Nt = len(self.T)
        self.Ns = len(self.S)

def get_initial_distribution(S, mean, std, dx):
    m0 = jnp.exp(-0.5 * (S - mean)**2 / (std**2))
    return m0 / (jnp.sum(m0) * dx)
