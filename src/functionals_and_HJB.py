def compute_loss(params, actor_def, coord_def, cfg, T, S, A, Nt, Ns, mc_samples, rng, lambda_val, prev_reg, tau_n, theta_eff):

    # --- 1. Δειγματοληψία ξ ---
    xi_mu, xi_log_sigma = coord_def.apply({'params': params['coord']})
    sigma_xi = jnp.exp(xi_log_sigma)
    epsilon_xi = jax.random.normal(rng, (mc_samples, 1))
    xi_b = xi_mu + sigma_xi * epsilon_xi

    # --- 2. Προετοιμασία Πλέγματος ---
    T_g, S_g = jnp.meshgrid(T, S, indexing='ij')
    tx = jnp.stack([T_g.flatten(), S_g.flatten()], axis=1)

    tx_batch = jnp.broadcast_to(tx[None, :, :], (mc_samples, tx.shape[0], 2)).reshape(-1, 2)
    xi_batch = jnp.broadcast_to(xi_b[:, None, :], (mc_samples, tx.shape[0], 1)).reshape(-1, 1)

    # --- 3. Forward Pass του Actor ---
    probs = actor_def.apply({'params': params['actor']}, tx_batch, xi_batch).reshape(mc_samples, Nt, Ns, -1)
    drift = jnp.clip(jnp.sum(probs * A, axis=-1), -3.0, 3.0)

    # --- 4. Fokker-Planck Εξίσωση ---
    m0 = get_initial_distribution(S, cfg.m0_mean, cfg.m0_std, cfg.dx)
    m0_batch = jnp.broadcast_to(m0[None, :], (mc_samples, Ns))

    alpha = (cfg.sigma**2 * cfg.dt) / (2 * cfg.dx**2)
    idx = jnp.arange(1, Ns-1)
    L_diff = jnp.zeros((Ns, Ns))
    L_diff = L_diff.at[idx, idx].set(-2).at[idx, idx-1].set(1).at[idx, idx+1].set(1)
    A_base = jnp.eye(Ns) - alpha * L_diff

    def fp_step(m_curr, b):
        b_p = jax.nn.relu(b)
        b_n = jax.nn.relu(-b)

        L_adv = jnp.zeros((mc_samples, Ns, Ns))
        L_adv = L_adv.at[:, idx, idx].set(-(b_p[:, idx] + b_n[:, idx]) / cfg.dx)
        L_adv = L_adv.at[:, idx, idx+1].set(b_n[:, idx+1] / cfg.dx)
        L_adv = L_adv.at[:, idx, idx-1].set(b_p[:, idx-1] / cfg.dx)

        A_sys = A_base[None, :, :] - cfg.dt * L_adv
        # Προσθέτουμε μια διάσταση [..., None] και μετά την αφαιρούμε με .squeeze(-1)
        m_next = jax.scipy.linalg.solve(A_sys, m_curr[..., None]).squeeze(-1)
        m_next = jax.nn.relu(m_next)
        m_next = m_next / (jnp.sum(m_next, axis=1, keepdims=True) * cfg.dx + 1e-8)
        return m_next, m_next

    _, m_trajectory = jax.lax.scan(fp_step, m0_batch, drift[:, :-1, :].transpose(1, 0, 2))
    m_trajectory = m_trajectory.transpose(1, 0, 2)
    m = jnp.concatenate([m0_batch[:, None, :], m_trajectory], axis=1)
    m_bar = jnp.sum(m * S, axis=2) * cfg.dx

    # --- 5. Υπολογισμός Κόστους (Reward) ---
    running_reward = (
        (cfg.alpha_term * m_bar[:, :, None])
        - (0.5 * cfg.beta_term * (m_bar**2)[:, :, None])
        - (cfg.epsilon / 2) * (S[None, None, :] - m_bar[:, :, None])**2
    )
    L = running_reward - 0.5 * (drift**2)
    J = jnp.sum(jnp.sum(m * L, axis=2)[:, :-1], axis=1) * cfg.dx * cfg.dt
    J0 = m_bar[:, -1]

    # --- 6. HJB Solver ---
    M1 = jnp.mean(m_bar, axis=0)
    M2 = jnp.mean(m_bar**2, axis=0)

    F_cost = (cfg.alpha_term + cfg.epsilon * S[None, :]) * M1[:, None] \
             - 0.5 * (cfg.beta_term + cfg.epsilon) * M2[:, None] \
             - 0.5 * cfg.epsilon * (S**2)[None, :]

    V_init = jnp.zeros(Ns)
    nu = 0.5 * cfg.sigma**2
    dt_sub = cfg.dt / 5.0

    def hjb_step(V, F_t):
        def inner_loop(i, V_inner):
            V_x = (V_inner[2:] - V_inner[:-2]) / (2 * cfg.dx)
            V_xx = (V_inner[2:] - 2 * V_inner[1:-1] + V_inner[:-2]) / (cfg.dx**2)
            a_opt = jnp.clip(V_x, -3.0, 3.0)
            H = a_opt * V_x - 0.5 * (a_opt**2)
            V_new = V_inner[1:-1] + dt_sub * (nu * V_xx + H + F_t[1:-1])
            return jnp.pad(V_new, (1, 1), mode='edge')
        V_next = jax.lax.fori_loop(0, 5, inner_loop, V)
        return V_next, None

    V_final, _ = jax.lax.scan(hjb_step, V_init, F_cost[:-1], reverse=True)
    V0 = jnp.sum(V_final * m0) * cfg.dx

    # --- 7. Υπολογισμός τελικής Loss ---
    J_mean = jnp.mean(J)
    reg = V0 - J_mean
    J0_mean = jnp.mean(J0)

    prox_term = (1.0 / (2.0 * tau_n)) * (reg - prev_reg)**2
    loss = (-J0_mean) + (lambda_val * reg) + (theta_eff * jax.nn.relu(reg)**2) + prox_term

    return loss, (J_mean, J0_mean, reg)
