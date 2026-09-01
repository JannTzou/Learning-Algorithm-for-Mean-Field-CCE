"""JAX training loop for the primal-dual mean-field CCE algorithm."""

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import optax
from tqdm.auto import trange

from .config import MFGConfig, MFGEngineStatic
from .networks import ActorNet, CoordinatorParams
from .objectives import compute_loss


History = dict[str, list[float]]
Parameters = Mapping[str, Any]


def initialize_parameters(
    actor: ActorNet,
    coordinator: CoordinatorParams,
    rng: jax.Array,
) -> dict[str, Any]:
    """Initialize the actor and correlation-device parameters."""
    actor_key, coordinator_key = jax.random.split(rng)

    actor_variables = actor.init(
        actor_key,
        jnp.zeros((1, 2)),
        jnp.zeros((1, 1)),
    )
    coordinator_variables = coordinator.init(coordinator_key)

    return {
        "actor": actor_variables["params"],
        "coord": coordinator_variables["params"],
    }


def train_primal_dual(
    cfg: MFGConfig,
    *,
    epochs: int = 1_000,
    mc_samples: int = 20,
    learning_rate: float = 1e-3,
    initial_lambda: float = 1.0,
    dual_step_size: float | None = None,
    proximal_step_size: float = 1.0,
    regret_penalty: float = 100.0,
    seed: int = 0,
    show_progress: bool = True,
) -> tuple[dict[str, Any], History]:
    """Train the recommendation policy with primal-dual updates.

    The actor and correlation-device parameters are updated with Adam.
    The non-negative Lagrange multiplier is updated by projected
    gradient ascent.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if mc_samples < 1:
        raise ValueError("mc_samples must be positive")
    if proximal_step_size <= 0:
        raise ValueError("proximal_step_size must be positive")

    engine = MFGEngineStatic(cfg)
    actor = ActorNet(action_dim=len(engine.A))
    coordinator = CoordinatorParams()

    rng = jax.random.PRNGKey(seed)
    init_key, rng = jax.random.split(rng)
    params = initialize_parameters(actor, coordinator, init_key)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    lambda_value = jnp.asarray(initial_lambda)
    previous_regret = jnp.asarray(0.0)

    sigma = (
        dual_step_size
        if dual_step_size is not None
        else epochs ** -0.5
    )

    history: History = {
        "loss": [],
        "reward": [],
        "moderator_objective": [],
        "regret": [],
        "lambda": [],
    }

    def loss_fn(
        current_params: Parameters,
        step_key: jax.Array,
        current_lambda: jax.Array,
        previous_regret_value: jax.Array,
    ):
        return compute_loss(
            params=current_params,
            actor_def=actor,
            coord_def=coordinator,
            cfg=cfg,
            T=engine.T,
            S=engine.S,
            A=engine.A,
            Nt=engine.Nt,
            Ns=engine.Ns,
            mc_samples=mc_samples,
            rng=step_key,
            lambda_val=current_lambda,
            prev_reg=previous_regret_value,
            tau_n=proximal_step_size,
            theta_eff=regret_penalty,
        )

    value_and_grad = jax.value_and_grad(
        loss_fn,
        has_aux=True,
    )

    progress = trange(
        epochs,
        disable=not show_progress,
        desc="Training",
    )

    for _ in progress:
        rng, step_key = jax.random.split(rng)

        (loss, metrics), grads = value_and_grad(
            params,
            step_key,
            lambda_value,
            previous_regret,
        )

        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            params,
        )
        params = optax.apply_updates(params, updates)

        reward, moderator_objective, regret = metrics

        lambda_value = jnp.maximum(
            0.0,
            lambda_value + sigma * regret,
        )
        previous_regret = jax.lax.stop_gradient(regret)

        history["loss"].append(float(loss))
        history["reward"].append(float(reward))
        history["moderator_objective"].append(
            float(moderator_objective)
        )
        history["regret"].append(float(regret))
        history["lambda"].append(float(lambda_value))

        progress.set_postfix(
            loss=f"{history['loss'][-1]:.4f}",
            regret=f"{history['regret'][-1]:.4f}",
        )

    return params, history