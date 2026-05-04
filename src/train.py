import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os

# Κάνουμε import τα εργαλεία από τα άλλα αρχεία μας!
from config import MFGConfig, MFGEngine, device
from models import Actor
from physics import compute_actor_loss_components, solve_hjb_differentiable

def train_actor_cce(cfg: MFGConfig, epochs=5000, mc_samples=20, burn_in=0, checkpoint_path=None):
    eng = MFGEngine(cfg)
    actor = Actor(len(eng.A), cfg.T_max).to(device)
    opt_a = optim.Adam(actor.parameters(), lr=1e-4)

    current_lambda = 1.0
    prev_reg = 0.0

    sum_tau = 0.0
    sum_tau_reward = 0.0
    sum_tau_regret = 0.0
    sum_tau_E_mu_T = 0.0

    hist = {
        'Reward': [], 'Regret': [], 'Lambda': [],
        'Weighted_Reward': [], 'Weighted_Regret': [],
        'E_mu_T': [], 'Weighted_E_mu_T': []
    }

    start_epoch = 1

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"🔄 Βρέθηκε αρχείο σωτηρίας! Φόρτωση από: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        actor.load_state_dict(checkpoint['model_state_dict'])
        hist = checkpoint['hist']
        start_epoch = checkpoint['epoch'] + 1

        if len(hist['Lambda']) > 0:
            current_lambda = hist['Lambda'][-1]
            prev_reg = hist['Regret'][-1]

        if 'E_mu_T' not in hist:
            hist['E_mu_T'] = [float('nan')] * (start_epoch - 1)

        if 'Weighted_E_mu_T' not in hist:
            hist['Weighted_E_mu_T'] = [float('nan')] * (start_epoch - 1)
            for i in range(burn_in, start_epoch):
                idx = i - 1
                if idx < len(hist['Reward']):
                    t_n = 1.0 / np.sqrt(i * (i + 1))
                    sum_tau += t_n
                    sum_tau_reward += t_n * hist['Reward'][idx]
                    sum_tau_regret += t_n * hist['Regret'][idx]
                    if not np.isnan(hist['E_mu_T'][idx]):
                        sum_tau_E_mu_T += t_n * hist['E_mu_T'][idx]

        if 'sum_tau_E_mu_T' in checkpoint:
            sum_tau_E_mu_T = checkpoint['sum_tau_E_mu_T']

        print(f"▶️ Η εκπαίδευση θα συνεχίσει από την εποχή {start_epoch}...")
    else:
        print("▶️ Δεν βρέθηκε αρχείο (ή δεν δόθηκε). Κανονική έναρξη από την εποχή 1...")

    pbar = tqdm(range(start_epoch, epochs + 1), desc=f"Training {cfg.name}")

    M1_prev = torch.zeros(eng.Nt, device=device)

    try:
        for k in pbar:
            xi_b = actor.sample_xi(mc_samples)
            J_list, m_bar_list = [], []

            opt_a.zero_grad()

            for i in range(mc_samples):
                try:
                    J, mb = compute_actor_loss_components(actor, eng, xi_b[i])
                    J_list.append(J)
                    m_bar_list.append(mb)
                except Exception:
                    continue

            if len(J_list) == 0: continue

            J_tensor = torch.stack(J_list)
            m_bar_tensor = torch.stack(m_bar_list)

            M1 = torch.mean(m_bar_tensor, dim=0)
            M2 = torch.mean(m_bar_tensor**2, dim=0)

            current_E_mu_T = M1[-1].item()

            V0 = solve_hjb_differentiable(M1, M2, eng)

            J_mean = torch.mean(J_tensor)
            reg = V0 - J_mean

            theta_eff = 100.0 if k < burn_in else 200.0
            cw = 1

            tau_n = 1.0
            sigma_n = 1.0/np.sqrt(epochs)

            integral_diff = torch.sum(M1 - M1_prev) * cfg.dt
            prox_term = (1.0 / (2.0 * tau_n)) * (integral_diff ** 2)

            loss = (cw * (-J_mean)) + (current_lambda * reg) + (theta_eff * torch.relu(reg)**2) + prox_term
            loss.backward()

            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            opt_a.step()

            M1_prev = M1.detach().clone()

            current_lambda = max(0.0, current_lambda + sigma_n * reg.item())
            prev_reg = reg.item()

            if k >= burn_in:
                sum_tau += tau_n
                sum_tau_reward += tau_n * J_mean.item()
                sum_tau_regret += tau_n * reg.item()
                sum_tau_E_mu_T += tau_n * current_E_mu_T

                weighted_avg_reward = sum_tau_reward / sum_tau
                weighted_avg_regret = sum_tau_regret / sum_tau
                weighted_avg_E_mu_T = sum_tau_E_mu_T / sum_tau
            else:
                weighted_avg_reward = float('nan')
                weighted_avg_regret = float('nan')
                weighted_avg_E_mu_T = float('nan')

            hist['Reward'].append(J_mean.item())
            hist['Regret'].append(reg.item())
            hist['Lambda'].append(current_lambda)
            hist['Weighted_Reward'].append(weighted_avg_reward)
            hist['Weighted_Regret'].append(weighted_avg_regret)
            hist['E_mu_T'].append(current_E_mu_T)
            hist['Weighted_E_mu_T'].append(weighted_avg_E_mu_T)

            if k % 10 == 0:
                pbar.set_description(f"[{cfg.name}] W_Rwd={weighted_avg_reward:.3f} | W_Reg={weighted_avg_regret:.4f}")

            if k % 500 == 0:
                save_path = checkpoint_path if checkpoint_path else 'rescue_data.pth'
                checkpoint = {
                    'epoch': k,
                    'model_state_dict': actor.state_dict(),
                    'hist': hist,
                    'current_lambda': current_lambda,
                    'prev_reg': prev_reg,
                    'sum_tau': sum_tau,
                    'sum_tau_reward': sum_tau_reward,
                    'sum_tau_regret': sum_tau_regret,
                    'sum_tau_E_mu_T': sum_tau_E_mu_T
                }
                torch.save(checkpoint, save_path)

        # Final save after the loop
        save_path = checkpoint_path if checkpoint_path else 'rescue_data.pth'
        checkpoint = {
            'epoch': epochs,
            'model_state_dict': actor.state_dict(),
            'hist': hist,
            'current_lambda': current_lambda,
            'prev_reg': prev_reg,
            'sum_tau': sum_tau,
            'sum_tau_reward': sum_tau_reward,
            'sum_tau_regret': sum_tau_regret,
            'sum_tau_E_mu_T': sum_tau_E_mu_T
        }
        torch.save(checkpoint, save_path)
        print(f"\n✅ Η εκπαίδευση ολοκληρώθηκε και το τελικό μοντέλο αποθηκεύτηκε στο '{save_path}'!")

    except KeyboardInterrupt:
        print("\n⚠️ Διακόπηκε από τον χρήστη! Αποθήκευση προόδου για αύριο...")

        save_path = checkpoint_path if checkpoint_path else 'rescue_data.pth'
        checkpoint = {
            'epoch': k,
            'model_state_dict': actor.state_dict(),
            'hist': hist,
            'current_lambda': current_lambda,
            'prev_reg': prev_reg,
            'sum_tau': sum_tau,
            'sum_tau_reward': sum_tau_reward,
            'sum_tau_regret': sum_tau_regret,
            'sum_tau_E_mu_T': sum_tau_E_mu_T
        }
        torch.save(checkpoint, save_path)
        print(f"✅ Η πρόοδος (Epoch {k}) σώθηκε με επιτυχία στο '{save_path}'!")

    return hist, actor, eng
