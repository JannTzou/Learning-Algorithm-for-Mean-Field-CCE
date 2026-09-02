"""Smoke tests for the JAX training loop."""

from pathlib import Path
from tempfile import TemporaryDirectory

import jax.numpy as jnp

from mfcce.checkpointing import load_checkpoint
from mfcce.config import MFGConfig
from mfcce.training import train_primal_dual


def _small_config() -> MFGConfig:
    """Create a small configuration for fast tests."""
    return MFGConfig(
        T_max=0.02,
        dt=0.01,
        dx=0.5,
        s_min=-1.0,
        s_max=1.0,
    )


def test_two_training_steps_are_finite() -> None:
    """Run two small training steps and check the output."""
    _, history = train_primal_dual(
        _small_config(),
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


def test_training_can_resume_from_checkpoint(
    tmp_path: Path,
) -> None:
    """Save one epoch and resume training for a second epoch."""
    checkpoint_path = tmp_path / "training-checkpoint.pkl"

    _, first_history = train_primal_dual(
        _small_config(),
        epochs=1,
        mc_samples=2,
        dual_step_size=0.1,
        seed=0,
        show_progress=False,
        checkpoint_path=checkpoint_path,
    )

    first_checkpoint = load_checkpoint(checkpoint_path)

    assert first_checkpoint["epoch"] == 1
    assert len(first_history["loss"]) == 1

    _, resumed_history = train_primal_dual(
        _small_config(),
        epochs=2,
        mc_samples=2,
        dual_step_size=0.1,
        show_progress=False,
        checkpoint_path=checkpoint_path,
        resume_from=checkpoint_path,
    )

    final_checkpoint = load_checkpoint(checkpoint_path)

    assert final_checkpoint["epoch"] == 2
    assert len(resumed_history["loss"]) == 2
    assert resumed_history["loss"][0] == first_history["loss"][0]


if __name__ == "__main__":
    test_two_training_steps_are_finite()

    with TemporaryDirectory() as directory:
        test_training_can_resume_from_checkpoint(
            Path(directory)
        )

    print("Training and checkpoint tests passed")