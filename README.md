# No-Regret Learning Algorithm for Optimal Mean-Field Coarse-Correlated Equilibrium

This file includes code for the project 'Optimal Mean-Field Coarse-Correlated Equilibrium: Linear Programming and No-Regret Learning' (coming soon to arXiv).

## Goal

The code provides a computational framework for learning **optimal coarse correlated equilibria (CCE)** in continuous mean-field games through **primal-dual** and **no-regret optimization schemes**.

The implementation is designed to connect the mathematical formulation of mean-field coarse correlated equilibria with reproducible numerical experiments. It contains experimental JAX implementations for

- simulating mean-field population dynamics under learned recommendation policies;
- estimating deviation values and external regret;
- training neural recommendation policies with primal-dual updates;
- monitoring ergodic objectives, regret estimates, and dual variables;
- visualizing convergence behavior and external-regret diagnostics.

Some examples are intended as validated benchmarks, while others are included as diagnostic experiments for testing modeling and numerical assumptions.


## Repository Structure

```text
src/
└── ├── __init__.py
    ├── configs.py
    ├── grids.py
    ├── networks.py
    ├── checkpointing.py
    ├── fp.py
    ├── hjb.py
    ├── training.py
    ├── models/
    │   ├── __init__.py
    │   ├── emission_abatement.py
    │   └── flocking.py
    └── visualization/
        ├── __init__.py
        ├── emission_plots.py
        └── flocking_plots.py

experiments/
├──  run_flocking.py
├──  run_emission_abatement.py
└──  run_emission_abatement_terminal.py 

results/
├── checkpoints/
│   └── .gitkeep
├── data/
│   └── .gitkeep
└── figures/
    └── .gitkeep
```


## Documentation roadmap

The documentation will be expanded with

- a detailed derivation of the primal-dual scheme;
- the connection between the linear-programming formulation and occupation measures;
- the numerical solution of the HJB equation;
- the construction of the best-deviation oracle;
- implementation details for population-aware policies;
- reproducible experiment tables;
- interpretation of the numerical results.


# Requirements

Installation
First, install the basic requirements:
pip install -r requirements.txt

Note: By default, this installs the CPU version of JAX. If you have an NVIDIA GPU and want to use hardware acceleration, please install JAX with CUDA support by following the official documentation:
pip install -U "jax[cuda12]"

The project requires the following Python packages:

- `jax>=0.4`
- `jaxlib>=0.4`
- `flax>=0.8`
- `optax>=0.2`
- `numpy>=1.24`
- `matplotlib>=3.7`
- `tqdm>=4.65`




## Citation

If you use this code in academic work, please cite the associated manuscript.




**The citation information will be updated once the manuscript metadata is finalized.**

---

## License

The license for this repository is specified in the `LICENSE` file.

For academic and research use, common choices include the MIT License and the BSD 3-Clause License. The final choice should be made before the repository is publicly released.

---

## Contact

For questions about the implementation or the mathematical formulation, please open a GitHub issue or contact the authors through the information provided in the associated manuscript.
