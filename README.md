# GCN robustness certification under heterogeneous node attack costs

Research code accompanying **Linear Programming-Based Robustness Certification of Multilayer Graph Convolutional Networks under Heterogeneous Node Attack Costs**.

This code supports exact certification of one-layer linear GCNs and affine linear-programming outer relaxations for two- and three-layer sigmoid GCNs under binary node-feature flips. Graph structure and trained parameters are fixed during certification. The heterogeneous-cost threat model retains a local flip-count limit and uses a global weighted cost budget.

## Contents

- `VRP/real_experiments.py`: one-layer experiments.
- `VRP/deep_experiments.py`: two- and three-layer experiments.
- `VRP/deep_verifier.py`, `VRP/sigmoid_envelope.py`: LP verification and activation bounds.
- `VRP/weighted_bounds.py`, `VRP/node_importance.py`, `VRP/graph_semantics.py`: cost-aware bounds and graph processing.
- `VRP/test_heterogeneous.py`, `VRP/test_deep_verifier.py`: regression and small-graph correctness tests.
- `scripts/`: experiment and analysis workflows.

The older `VRP/LP.py`, `VRP/active_bound.py`, and `VRP/compute.py` are legacy components, not the multilayer verifier entry point.

## Environment

The supplied dependency snapshot targets Python 3.12 and CPU PyTorch. Platform-specific package availability may require a compatible PyTorch installation. A fresh installation on every platform has not been verified.

```bash
python3.12 -m venv .venv-risk
.venv-risk/bin/python -m pip install -r requirements-lock.txt
.venv-risk/bin/python -m unittest discover -v
```

## Reproduction

```bash
# One-layer Cora/CiteSeer experiments and analysis
bash scripts/reproduce_cora_citeseer.sh

# Two-/three-layer experiments, sample-count ablations and analysis
bash scripts/reproduce_deep_cora_citeseer.sh
```

The scripts use five seeds (0–4), Legacy and heterogeneous-cost threat modes, and dataset-specific global budget grids. One-layer certification evaluates the standard 1,000-node test mask; multilayer certification evaluates fixed 50-target test cohorts per trained model. Local limits are 14 for Cora and 37 for CiteSeer; global budgets are {3,7,14,28} and {9,18,37,74}, respectively.

Scripts create their output and model-cache directories, including default paths under `.pi-tasks/`. These directories do not need to exist in advance. `PY`, `ARTIFACTS`, `CACHE`, and `MODEL_CACHE` can be overridden through environment variables. Full runs can be computationally expensive; the deep script defaults to six parallel workers (`PARALLEL`).

Cora and CiteSeer are obtained through the dataset loader. Downloaded datasets, trained checkpoints, raw experiment results, manuscript files, and private development history are not included in this code release. This repository is a code release, not a deposit of the original experimental data.

## Interpretation

Comparing Legacy with heterogeneous costs changes the admissible attack set; it does not demonstrate an improvement in the trained model's intrinsic robustness. Baseline versus Dual comparisons hold the model, target, and attack constraints fixed. Dual retains baseline constraints and adds full-interval-calibrated affine constraints. The method does not require strictly improved certification rates.

No software reuse license has been added to this release. Public access alone does not grant an open-source license; contact the authors regarding reuse permissions.
