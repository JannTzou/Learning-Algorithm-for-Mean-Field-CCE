"""Smoke test for the JAX training loop."""

import jax.numpy as jnp

from mfcce.config import MFGConfig
from mfcce.training import train_primal_dual


def test_two_training_steps_are_finite() -> None:
    """Run two small training steps and check the output."""
    cfg = MFGConfig(
        T_max=0.02,
        dt=0.01,
        dx=0.5,
        s_min=-1.0,
        s_max=1.0,
    )

    _, history = train_primal_dual(
        cfg,
        epochs=2,
        mc_samples=2,
        seed=0,
        show_progress=False,
    )

    assert len(history["loss"]) == 2
    assert bool(
        jnp.all(
            jnp.isfinite(
                jnp.asarray(history["loss"])
            )
        )
    )


if __name__ == "__main__":
    test_two_training_steps_are_finite()
    print("Training smoke test passed")