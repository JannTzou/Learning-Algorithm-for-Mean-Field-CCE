"""Run the emissions-abatement mean-field CCE experiment."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from mfcce.config import MFGConfig, MFGEngineStatic
from mfcce.plotting import calculate_baselines, plot_enhanced_results
from mfcce.training import train_primal_dual


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the mean-field CCE emissions experiment."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small two-epoch smoke experiment.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=None,
        help="Number of Monte Carlo samples per epoch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="JAX random seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory used for generated outputs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the checkpoint in the output directory.",
    )
    return parser.parse_args()


def create_config(quick: bool) -> MFGConfig:
    """Create either the full or the small test configuration."""
    if quick:
        return MFGConfig(
            name="Emissions Abatement Game",
            T_max=0.02,
            dt=0.01,
            dx=0.5,
            s_min=-1.0,
            s_max=1.0,
        )

    return MFGConfig(name="Emissions Abatement Game")


def main() -> None:
    """Train the model and save its diagnostics."""
    args = parse_arguments()

    epochs = args.epochs
    if epochs is None:
        epochs = 2 if args.quick else 1_000

    mc_samples = args.mc_samples
    if mc_samples is None:
        mc_samples = 2 if args.quick else 20

    output_dir = args.output_dir
    if output_dir is None:
        run_name = "quick" if args.quick else "full"
        output_dir = (
            Path("outputs")
            / "emissions_abatement"
            / run_name
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "checkpoint.pkl"
    history_path = output_dir / "history.json"
    figure_path = output_dir / "training_diagnostics.png"

    if args.resume and not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Cannot resume; checkpoint not found: {checkpoint_path}"
        )

    config = create_config(args.quick)

    _, history = train_primal_dual(
        config,
        epochs=epochs,
        mc_samples=mc_samples,
        seed=args.seed,
        show_progress=True,
        checkpoint_path=checkpoint_path,
        checkpoint_every=max(1, epochs // 10),
        resume_from=checkpoint_path if args.resume else None,
    )

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    engine = MFGEngineStatic(config)
    baselines = calculate_baselines(engine)

    figure = plot_enhanced_results(
        history,
        *baselines,
        example_name=config.name,
        show=False,
    )
    figure.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    print("\nExperiment completed")
    print(f"Epochs: {len(history['loss'])}")
    print(f"Final reward: {history['reward'][-1]:.6f}")
    print(f"Final regret: {history['regret'][-1]:.6f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()