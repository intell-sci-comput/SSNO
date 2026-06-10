# Stable Spectral Neural Operator for Learning Stiff PDE Systems From Limited Data

Official implementation of **Stable Spectral Neural Operator (SSNO)** for learning stiff PDE dynamics from limited data.

<p align="center">
  <img src="https://github.com/user-attachments/assets/cffa7cc4-2383-4f32-81d7-ace8b50b9610" width="720" alt="SSNO overview" />
</p>

## Overview

This repository contains data-generation and neural-operator training code for stiff PDE benchmarks. The implementation focuses on stable long-horizon prediction when only a small number of trajectories are available for training.

Supported benchmark systems:

| Folder | System | Dimension |
| --- | --- | --- |
| `kuramoto_sivashinsky/` | Kuramoto-Sivashinsky equation | 1D space + time |
| `kuramoto_sivashinsky3d/` | 3D Kuramoto-Sivashinsky-style benchmark | 2D space + time |
| `navier_stokes/` | Navier-Stokes equation | 2D space + time |
| `swift_hohenberg/` | Swift-Hohenberg equation | 1D space + time |
| `swift_hohenberg3d/` | 3D Swift-Hohenberg-style benchmark | 2D space + time |

## Repository Structure

Each benchmark folder follows the same layout:

```text
<benchmark>/
├── data/
│   ├── main.py          # generate train/val/test trajectories
│   └── *.py             # PDE solver and data utilities
└── neural_operator/
    ├── main.py          # training entry point
    ├── train.py         # training loop
    ├── tools.py         # utilities
    └── model/           # SSNO and baseline neural operators
```

Implemented model families include SSNO, FFNO, CNext, FactFormer, DPOT, and PeRCNN, depending on the benchmark folder.

## Installation

Create a Python environment and install the core dependencies:

```bash
conda create -n ssno python=3.10 -y
conda activate ssno

# Install the PyTorch build that matches your CUDA or CPU environment first.
pip install torch numpy matplotlib einops tqdm thop
```

The experiments are written for PyTorch and can run on CUDA devices when available. Use `--device cpu` for CPU-only runs.

## Data Generation

Data are generated separately for each benchmark. For example:

```bash
cd kuramoto_sivashinsky/data
python main.py --path ../../data/kuramoto_sivashinsky --device cuda:0
```

The data scripts generate train, validation, and test splits. The default paths in the scripts are machine-specific, so pass `--path` explicitly on a new machine.

Common data-generation arguments:

- `--path`: output directory for generated data
- `--device`: compute device, such as `cuda:0` or `cpu`
- `--s`: spatial resolution
- `--N`: number of trajectories
- `--T`: final simulation time
- `--dt`: solver time step
- `--record_ratio`: temporal subsampling ratio

## Training

After generating data, train a model from the corresponding `neural_operator` folder:

```bash
cd kuramoto_sivashinsky/neural_operator
python main.py \
  --model SSNO \
  --data_path ../../data/kuramoto_sivashinsky \
  --device cuda:0
```

Common training arguments:

- `--model`: model name, for example `SSNO`, `FFNO`, `CNext`, `FactFormer`, `DPOT`, or `PeRCNN`
- `--data_path`: directory containing generated train/val/test data
- `--num_train`: number of training trajectories
- `--batch_size`: batch size
- `--num_iterations`: number of optimization iterations
- `--lr`: learning rate
- `--num_step1`, `--num_step2`: rollout/training horizon controls
- `--sparse_coeff`: sparse regularization coefficient

Training logs, checkpoints, histories, and results are written under the selected `--data_path` directory.

## Notes

- Generated datasets, checkpoints, logs, and Python cache files are intentionally not tracked.
- Override the hard-coded default paths in scripts with `--path` and `--data_path` when running on a new machine.

## Citation

If this code is useful for your research, please cite the accompanying paper:

```bibtex
@article{ssno,
  title   = {Stable Spectral Neural Operator for Learning Stiff PDE Systems From Limited Data},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```
