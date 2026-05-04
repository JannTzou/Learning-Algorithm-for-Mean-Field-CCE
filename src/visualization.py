import numpy as np
import matplotlib.pyplot as plt

def calculate_baselines(eng):
    if "Emissions Abatement Game" in eng.cfg.name:
        j_nash = -1.2
        j_social = 5.8
        j_CCE = 4.7
    else:
        j_nash = -1.2
        j_social = 5.8
        j_CCE = 4.7

    return j_nash, j_social, j_CCE

def plot_enhanced_results(hist, j_n, j_s, j_explicit_CCE, example_name):
    c = np.array(hist['Reward'])
    r = np.array(hist['Regret'])
    w_c = np.array(hist['Weighted_Reward'])
    w_r = np.array(hist['Weighted_Regret'])

    e_mu_t = np.array(hist.get('E_mu_T', []))
    w_e_mu_t = np.array(hist.get('Weighted_E_mu_T', []))

    epochs = np.arange(1, len(c) + 1)

    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"{example_name}", fontsize=15, fontweight='bold')

    ax[0].plot(epochs, c, color='gray', alpha=0.3, label='Actual Cost/Reward_n')
    ax[0].plot(epochs, w_c, color='blue', linewidth=2.5, label='Weighted Avg')
    ax[0].axhline(j_n, color='red', ls='--', label='Nash')
    ax[0].axhline(j_s, color='green', ls='--', label='MFC')
    ax[0].axhline(j_explicit_CCE, color='orange', ls='--', label='Optimal CCE')
    ax[0].set_title('Weighted Avg Cost / Reward', fontsize=12)
    ax[0].set_xlabel('Epochs (n)')
    ax[0].set_ylabel('Cost / Reward')
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(epochs, r, color='gray', alpha=0.3, label='Actual Regret_n')
    ax[1].plot(epochs, w_r, color='purple', linewidth=2.5, label='Weighted Avg Regret')
    ax[1].axhline(0, color='black', ls='--', linewidth=1.5)
    ax[1].set_title('Actual Regret vs. Weighted Avg Regret', fontsize=12)
    ax[1].set_xlabel('Epochs (n)')
    ax[1].set_ylabel('Regret')
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    if len(e_mu_t) > 0:
        ax[2].plot(epochs, e_mu_t, color='gray', alpha=0.3, label=r'Actual $\mathbb{E}[\overline{\mu}_T]$')
        if len(w_e_mu_t) > 0:
             ax[2].plot(epochs, w_e_mu_t, color='teal', linewidth=2.5, label='Weighted Avg')

        ax[2].set_title('Avg. Cumulative Terminal Abatement', fontsize=12)
        ax[2].set_xlabel('Epochs (n)')
        ax[2].set_ylabel('Terminal Abatement')
        ax[2].grid(True, alpha=0.3)
        ax[2].legend()

    plt.tight_layout()
    plt.show()
