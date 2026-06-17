# No-Regret Learning Algorithm for Optimal Mean-Field Coarse-Correlated Equilibrium

This file includes code for the project 'Optimal Mean-Field Coarse-Correlated Equilibrium: Linear Programming and No-Regret Learning' (coming soon to arXiv).

#Goal

The code provides a computational framework for learning **optimal coarse correlated equilibria (CCE)** in continuous mean-field games through **primal-dual** and **no-regret optimization schemes**.

The implementation is designed to connect the mathematical formulation of the problem with reproducible numerical experiments. In particular, it includes tools for

- solving the representative agent's control problem;
- computing best unilateral deviations;
- evaluating external regret;
- updating correlation devices and dual variables;
- comparing population-aware and population-unaware policies;
- visualizing convergence, population dynamics, and learned controls.
# Structure

# Overview

A mean-field coarse correlated equilibrium is a randomized recommendation mechanism under which a representative player has no incentive to commit, before observing the recommendation, to a different admissible strategy.

The optimization problem considered in this repository combines two objectives:

1. **Equilibrium:** the recommendation must satisfy the coarse correlated equilibrium constraints;
2. **Optimality:** among approximately incentive-compatible recommendations, the algorithm seeks one minimizing a prescribed social or planner objective.

The numerical method is based on a primal-dual formulation. At each iteration, the algorithm alternates between

- a primal update of the distribution over recommended policies;
- the computation of a best-deviation response;
- the evaluation of the corresponding external regret;
- a dual update penalizing violations of the equilibrium constraint.

The resulting procedure can be interpreted as a no-regret learning scheme over a space of probability measures.

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
