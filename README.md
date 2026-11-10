# NCRF

Neumann-aware Control-Relaxation Flux Forecaster (NCRF) is a dual-scale convolutional recurrent model for predicting controlled phase-separation evolution. This repository contains the main model, comparison models, and training and evaluation code.

## Environment

Python 3.10 or later and PyTorch are required. The recorded experiment used Python 3.11.13, PyTorch 2.8.0 with CUDA 12.8, and NumPy 2.3.2. Core versions are listed in `requirements/paper-server-constraints.txt`.

## Project structure

- `src/ncrf/`: models, data processing, training, and evaluation.
- `configs/`: experiment configurations.
- `tools/`: command-line entry point and solver-timing helpers.
- `tests/`: checks using synthetic data.
- `third_party/`: third-party code and licenses.

Training and evaluation require the corresponding dataset and metadata.
