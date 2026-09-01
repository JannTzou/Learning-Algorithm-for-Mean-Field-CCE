"""Smoke test for the external-regret objective."""

import jax
import jax.numpy as jnp

from mfcce.config import MFGConfig, MFGEngineStatic
from mfcce.networks import ActorNet, CoordinatorParams
from mfcce.objectives import compute_loss


def test_compute_loss_is_finite() -> None:
    """Run the objective on a small grid and check that outputs are finite."""
    cfg = MFGConfig(
        T_max=0.02,
        dt=0.01,
        dx=0.5,
        s_min=-1.0,
        s_max=1.0,
    )
    engine = MFGEngineStatic(cfg)

    actor = ActorNet(action_dim=len(engine.A))
    coordinator = CoordinatorParams()

    key_actor, key_coord, key_loss = jax.random.split(
        jax.random.PRNGKey(0), 3
    )

    actor_variables = actor.init(
        key_actor,
        jnp.zeros((1, 2)),
        jnp.zeros((1, 1)),
    )
    coordinator_variables = coordinator.init(key_coord)

    params = {
        "actor": actor_variables["params"],
        "coord": coordinator_variables["params"],
    }

    loss, metrics = compute_loss(
        params=params,
        actor_def=actor,
        coord_def=coordinator,
        cfg=cfg,
        T=engine.T,
        S=engine.S,
        A=engine.A,
        Nt=engine.Nt,
        Ns=engine.Ns,
        mc_samples=2,
        rng=key_loss,
        lambda_val=1.0,
        prev_reg=0.0,
        tau_n=1.0,
        theta_eff=1.0,
    )

    values = jnp.asarray((loss, *metrics))
    assert bool(jnp.all(jnp.isfinite(values)))

    print("loss and metrics:", values)


if __name__ == "__main__":
    test_compute_loss_is_finite()
