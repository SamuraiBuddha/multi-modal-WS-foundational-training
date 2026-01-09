# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an interactive 40-hour educational program teaching neural architecture design, culminating in a novel **Segmented Watts-Strogatz Multi-Modal Architecture**. The program teaches ML fundamentals through hands-on Jupyter notebooks and web-based visualizations.

**Current Status**: Planning/documentation phase. The PRD and technical specification exist in `docs/`, but the actual implementation (notebooks, source code, web components) has not been built yet.

## Target Architecture

The capstone involves building a multi-modal neural network where:
- Heterogeneous modules handle different modalities (visual/audio/text)
- Modules interconnect via Watts-Strogatz small-world topology
- Rewiring probability (beta) is learnable during training
- Dynamic connectivity evolves using SET/DEEP R sparse training principles

## Development Commands

```powershell
# Environment setup (once implemented)
python scripts/setup_environment.py

# Launch JupyterLab
jupyter lab

# Run tests
pytest tests/

# Type checking
mypy src/

# Linting
flake8 src/
black src/ --check
isort src/ --check-only
```

## Technology Stack

**Primary:**
- Python 3.11+
- PyTorch 2.1+
- JupyterLab 4.x
- NetworkX (graph algorithms)
- React + D3.js + Plotly (web visualizations)
- ipywidgets (notebook interactivity)

**Secondary (for comparison examples):**
- TensorFlow/Keras
- JAX/Flax
- Pure NumPy (early modules)

## Repository Structure (Planned)

```
notebooks/           # Jupyter notebooks - main educational content
  00_setup/         # Environment verification
  01_foundations/   # Neural network basics
  02_supervised/    # Loss, gradients, backprop
  03_graphs/        # Graph theory fundamentals
  04_topology/      # WS, BA, ER network models
  05_sparse/        # Sparse neural networks
  06_unsupervised/  # Autoencoders
  07_dynamic_sparse/# SET, DEEP R algorithms
  08_modular/       # Mixture of Experts
  09_multimodal/    # Multi-modal fusion
  10_capstone/      # Build the full architecture

src/                # Python source code
  core/             # Utilities
  models/           # Neural network implementations
  topology/         # Graph/topology algorithms
  training/         # Training loops
  visualization/    # Python viz helpers
  data/             # Data loading

web/                # React web application for visualizations
tests/              # pytest test suites
configs/            # YAML configuration files
```

## Key Algorithms to Implement

1. **Watts-Strogatz graph generator** - Small-world network creation with tunable rewiring
2. **SET (Sparse Evolutionary Training)** - Dynamic sparse training via evolutionary principles
3. **DEEP R** - Deep Rewiring algorithm for connectivity evolution
4. **Bipartite Small-World (BSW)** - SW topology applied to neural layers
5. **Learnable beta rewiring** - Differentiable rewiring probability

## Key Research Papers

- Watts & Strogatz 1998 (Nature) - Small-world networks
- Mocanu et al. 2018 (Nature Comms) - SET algorithm
- Bellec et al. 2017 - DEEP R algorithm
- Zhang et al. 2023 - BSW/CHT

## Platform Requirements

- Primary platform: Windows (native PowerShell)
- GPU: NVIDIA with 8GB+ VRAM recommended (CPU fallback required)
- All notebook cells should execute within 30 seconds on single GPU

## Output Formatting

Do not use Unicode emoji characters. Use ASCII alternatives:
- `[OK]` instead of checkmarks
- `[X]` or `[FAIL]` instead of x-marks
- `[WARN]` or `[!]` instead of warning symbols
- `[->]` instead of arrows
