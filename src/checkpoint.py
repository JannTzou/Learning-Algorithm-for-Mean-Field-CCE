"""Utilities for saving and loading training checkpoints."""

from pathlib import Path
from typing import Any

import pickle

import flax
import jax
import jax.numpy as jnp
import numpy as np


def _to_numpy_tree(tree: Any) -> Any:
    """Convert all leaves of a JAX pytree to NumPy arrays."""
    return jax.tree_util.tree_map(np.array, tree)


def _to_jax_tree(tree: Any) -> Any:
    """Convert all leaves of a pytree to JAX arrays."""
    return jax.tree_util.tree_map(jnp.array, tree)


def save_checkpoint(
    filepath: str | Path,
    epoch: int,
    params: Any,
    opt_state: Any,
    hist: dict[str, Any],
    current_lambda: float,
    prev_reg: float,
    sum_tau: float,
    sum_tau_reward: float,
    sum_tau_regret: float,
    sum_tau_J0: float,
) -> None:
    """Save a training checkpoint."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    params_cpu = _to_numpy_tree(params)
    opt_state_cpu = _to_numpy_tree(opt_state)

    try:
        params_cpu = flax.core.unfreeze(params_cpu)
    except Exception:
        pass

    checkpoint_data = {
        "epoch": epoch,
        "params": params_cpu,
        "opt_state": opt_state_cpu,
        "hist": hist,
        "current_lambda": current_lambda,
        "prev_reg": prev_reg,
        "sum_tau": sum_tau,
        "sum_tau_reward": sum_tau_reward,
        "sum_tau_regret": sum_tau_regret,
        "sum_tau_J0": sum_tau_J0,
    }

    with filepath.open("wb") as f:
        pickle.dump(checkpoint_data, f)


def load_checkpoint(filepath: str | Path) -> dict[str, Any]:
    """Load a training checkpoint."""
    filepath = Path(filepath)

    with filepath.open("rb") as f:
        checkpoint_data = pickle.load(f)

    checkpoint_data["params"] = _to_jax_tree(checkpoint_data["params"])
    checkpoint_data["opt_state"] = _to_jax_tree(checkpoint_data["opt_state"])

    return checkpoint_data
