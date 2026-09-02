"""Utilities for saving and loading resumable training checkpoints."""

from pathlib import Path
from typing import Any

import pickle

import jax
import jax.numpy as jnp
import numpy as np


_REQUIRED_KEYS = {
    "version",
    "epoch",
    "params",
    "opt_state",
    "history",
    "lambda_value",
    "previous_regret",
    "rng",
}


def _to_numpy_tree(tree: Any) -> Any:
    """Move every JAX array in a pytree to a NumPy array."""
    return jax.tree_util.tree_map(np.asarray, tree)


def _to_jax_tree(tree: Any) -> Any:
    """Convert every NumPy array in a pytree to a JAX array."""
    return jax.tree_util.tree_map(jnp.asarray, tree)


def save_checkpoint(
    filepath: str | Path,
    *,
    epoch: int,
    params: Any,
    opt_state: Any,
    history: dict[str, list[float]],
    lambda_value: Any,
    previous_regret: Any,
    rng: Any,
) -> None:
    """Save all state required to resume training."""
    if epoch < 0:
        raise ValueError("epoch must be non-negative")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_data = {
        "version": 1,
        "epoch": int(epoch),
        "params": _to_numpy_tree(params),
        "opt_state": _to_numpy_tree(opt_state),
        "history": history,
        "lambda_value": np.asarray(lambda_value),
        "previous_regret": np.asarray(previous_regret),
        "rng": np.asarray(rng),
    }

    temporary_path = filepath.with_name(f"{filepath.name}.tmp")

    with temporary_path.open("wb") as file:
        pickle.dump(
            checkpoint_data,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    temporary_path.replace(filepath)


def load_checkpoint(filepath: str | Path) -> dict[str, Any]:
    """Load and validate a resumable training checkpoint."""
    filepath = Path(filepath)

    if not filepath.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    with filepath.open("rb") as file:
        checkpoint_data = pickle.load(file)

    missing_keys = _REQUIRED_KEYS.difference(checkpoint_data)

    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Invalid checkpoint; missing entries: {missing}")

    checkpoint_data["params"] = _to_jax_tree(
        checkpoint_data["params"]
    )
    checkpoint_data["opt_state"] = _to_jax_tree(
        checkpoint_data["opt_state"]
    )
    checkpoint_data["lambda_value"] = jnp.asarray(
        checkpoint_data["lambda_value"]
    )
    checkpoint_data["previous_regret"] = jnp.asarray(
        checkpoint_data["previous_regret"]
    )
    checkpoint_data["rng"] = jnp.asarray(checkpoint_data["rng"])

    return checkpoint_data