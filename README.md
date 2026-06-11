# CausalSR

This repository contains the implementation of **CausalSR**, a scale-resolved causal discovery workflow for multivariate time-series data.

The method uses time-frequency representation, scale-specific causal relation estimation, indirect-effect screening, and joint graph refinement to recover causal structures under heterogeneous temporal response scales.

The repository currently includes scripts for three types of experiments:

- `CausalSR_synthetic.py`: experiments on synthetic multi-scale time-series data.
- `CausalSR_fMRI.py`: experiments on fMRI simulation data stored in `.mat` format.
- `CausalSR_causalrivers.py`: experiments on the CausalRivers benchmark.

## External Dependencies

The code uses the following external Python packages:

```text
numpy
pandas
matplotlib
scipy
scikit-learn
torch
pycwt
hydra-core
omegaconf
tigramite
```

### Package Usage

| Package | Usage |
|---|---|
| `numpy` | Numerical computation, matrix operations, FFT, and synthetic data generation |
| `pandas` | DataFrame storage and tabular result processing |
| `matplotlib` | Visualization and plotting |
| `scipy` | Loading `.mat` files and scientific computing utilities |
| `scikit-learn` | Evaluation metrics such as F1, confusion matrix, MSE, and AUROC |
| `torch` | Optimization variables and gradient-based computation |
| `pycwt` | Continuous wavelet transform, inverse wavelet transform, and wavelet coherence-related computation |
| `hydra-core` | Configuration management for the CausalRivers experiment |
| `omegaconf` | Configuration object support used together with Hydra |
| `tigramite` | Time-series causal discovery algorithms, such as PCMCI and PCMCI+ |
