import torch
import numpy as np

def solve_hjb_differentiable(M1, M2, eng):
    cfg = eng.cfg
    dt, dx = cfg.dt, cfg.dx
    Ns, Nt = eng.Ns, eng.Nt
    S_ten = torch.tensor(eng.S, device=eng.device).float()

    F = (cfg.alpha_term + cfg.epsilon * S_ten.unsqueeze(0)) * M1.unsqueeze(1) \
        - 0.5 * (cfg.beta_term + cfg.epsilon) * M2.unsqueeze(1) \
        - 0.5 * cfg.epsilon * (S_ten**2).unsqueeze(0)

    V = torch.zeros(Ns, device=eng.device)
    nu = 0.5 * cfg.sigma**2
    dt_sub = dt / 5

    for t in range(Nt - 2, -1, -1):
        F_t = F[t]
        for _ in range(5):
            V_x = (V[2:] - V[:-2]) / (2 * dx)
            V_xx = (V[2:] - 2 * V[1:-1] + V[:-2]) / (dx**2)

            a_opt = torch.clamp(V_x, -3.0, 3.0)
            H = a_opt * V_x - 0.5 * (a_opt**2)

            V_inner = V[1:-1] + dt_sub * (nu * V_xx + H + F_t[1:-1])
            V = torch.cat([V_inner[0:1], V_inner, V_inner[-1:]])

    m0 = torch.tensor(eng.get_initial_distribution(), device=eng.device).float()
    V0 = torch.sum(V * m0) * dx
    return V0


def compute_actor_loss_components(actor, eng, xi_val):
    cfg, dev = eng.cfg, eng.device
    T_ten = torch.tensor(eng.T, device=dev).float()
    S_ten = torch.tensor(eng.S, device=dev).float()
    A_ten = torch.tensor(eng.A, device=dev).float()
    Nt, Ns = len(T_ten), len(S_ten)

    T_g, S_g = torch.meshgrid(T_ten, S_ten, indexing='ij')
    tx = torch.stack([T_g.flatten(), S_g.flatten()], dim=1)

    probs = actor(tx, xi_val.reshape(1,1)).reshape(Nt, Ns, -1)
    drift = torch.clamp(torch.sum(probs * A_ten, dim=2), -3.0, 3.0)

    m_list = [torch.tensor(eng.get_initial_distribution(), device=dev).float()]

    alpha = (cfg.sigma**2 * cfg.dt) / (2 * cfg.dx**2)
    idx = torch.arange(1, Ns-1, device=dev)
    L_diff = torch.zeros((Ns, Ns), device=dev)
    L_diff[idx, idx] = -2; L_diff[idx, idx-1] = 1; L_diff[idx, idx+1] = 1
    A_base = torch.eye(Ns, device=dev) - alpha * L_diff

    for t in range(Nt - 1):
        b = drift[t]; b_p, b_n = torch.relu(b), torch.relu(-b)
        L_adv = torch.zeros((Ns, Ns), device=dev)
        L_adv[idx, idx] = -(b_p[idx]+b_n[idx])/cfg.dx
        L_adv[idx, idx+1] = b_n[idx+1]/cfg.dx
        L_adv[idx, idx-1] = b_p[idx-1]/cfg.dx

        A_sys = A_base - cfg.dt * L_adv
        A_sys[0,0] = 1; A_sys[-1,-1] = 1

        m_next = torch.relu(torch.linalg.solve(A_sys, m_list[-1]))
        m_next = m_next / (torch.sum(m_next)*cfg.dx + 1e-8)
        m_list.append(m_next)

    m = torch.stack(m_list)

    m_bar = torch.sum(m * S_ten, dim=1) * cfg.dx

    running_reward = (cfg.alpha_term * m_bar.unsqueeze(1)) - (0.5 * cfg.beta_term * m_bar.unsqueeze(1)**2) - (cfg.epsilon / 2) * (S_ten.unsqueeze(0) - m_bar.unsqueeze(1))**2

    L = running_reward - 0.5 * (drift**2)
    J = torch.sum(torch.sum(m * L, dim=1)[:-1]) * cfg.dx * cfg.dt

    return J, m_bar
