import matplotlib.pyplot as plt
import numpy as np


def calculate_baselines(eng):
    if "Emissions Abatement Game" in eng.cfg.name:
        return -1.2, 5.8, 4.7

    raise ValueError(
        f"No reference baselines are available for {eng.cfg.name!r}."
    )


def _running_average(values):
    values = np.asarray(values, dtype=float)
    return np.cumsum(values) / np.arange(1, len(values) + 1)


def plot_enhanced_results(
    hist,
    j_n,
    j_s,
    j_explicit_CCE,
    example_name,
    show=True,
):
    required_keys = {
        "loss",
        "reward",
        "moderator_objective",
        "regret",
        "lambda",
    }
    missing_keys = required_keys.difference(hist)

    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise KeyError(f"Missing history entries: {missing}")

    reward = np.asarray(hist["reward"], dtype=float)
    regret = np.asarray(hist["regret"], dtype=float)
    loss = np.asarray(hist["loss"], dtype=float)
    moderator = np.asarray(hist["moderator_objective"], dtype=float)
    lambda_values = np.asarray(hist["lambda"], dtype=float)

    epochs = np.arange(1, len(reward) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(example_name, fontsize=15, fontweight="bold")

    axes[0, 0].plot(epochs, reward, color="gray", alpha=0.4, label="Reward")
    axes[0, 0].plot(
        epochs,
        _running_average(reward),
        color="blue",
        linewidth=2.5,
        label="Running average",
    )
    axes[0, 0].axhline(j_n, color="red", linestyle="--", label="Nash")
    axes[0, 0].axhline(j_s, color="green", linestyle="--", label="MFC")
    axes[0, 0].axhline(
        j_explicit_CCE,
        color="orange",
        linestyle="--",
        label="Optimal CCE",
    )
    axes[0, 0].set_title("Reward")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, regret, color="gray", alpha=0.4, label="Regret")
    axes[0, 1].plot(
        epochs,
        _running_average(regret),
        color="purple",
        linewidth=2.5,
        label="Running average",
    )
    axes[0, 1].axhline(0.0, color="black", linestyle="--")
    axes[0, 1].set_title("External regret")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, loss, label="Training loss")
    axes[1, 0].plot(epochs, moderator, label="Moderator objective")
    axes[1, 0].set_title("Training objectives")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, lambda_values, color="teal", label="Lambda")
    axes[1, 1].set_title("Dual variable")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)

    fig.tight_layout()

    if show:
        plt.show()

    return fig