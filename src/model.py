class ActorNet(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, tx, xi_val):
        # tx: [batch, 2], xi_val: [batch, 1]
        if xi_val.shape[0] != tx.shape[0]:
            xi_val = jnp.broadcast_to(xi_val, (tx.shape[0], 1))

        inp = jnp.concatenate([tx, xi_val], axis=-1)
        x = nn.Dense(64)(inp)
        x = nn.LayerNorm()(x)
        x = nn.silu(x)
        x = nn.Dense(64)(x)
        x = nn.LayerNorm()(x)
        x = nn.silu(x)
        mu = 3.0 * jnp.tanh(nn.Dense(1)(x) / 3.0)

        # Παράμετροι της κατανομής της δράσης
        log_theta = self.param('log_theta', lambda rng: jnp.array([jnp.log(0.4)]))
        theta = jnp.exp(log_theta) + 1e-4

        actions = jnp.linspace(-3.0, 3.0, self.action_dim)
        sq_diff = (actions[None, :] - mu) ** 2
        exponent = -sq_diff / (2 * theta ** 2)

        # Softmax over actions
        return jax.nn.softmax(exponent, axis=-1)

# Παράμετροι για το σήμα (Coordinator)
class CoordinatorParams(nn.Module):
    @nn.compact
    def __call__(self):
        xi_mu = self.param('xi_mu', lambda rng: jnp.array([0.0]))
        xi_log_sigma = self.param('xi_log_sigma', lambda rng: jnp.array([jnp.log(0.2)]))
        return xi_mu, xi_log_sigma
