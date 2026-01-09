# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An interactive 40-hour educational program teaching neural architecture design, culminating in a **Segmented Watts-Strogatz Multi-Modal Architecture**. The capstone network uses sparse small-world connectivity between modality-specific encoders, with learnable rewiring probability (beta) that evolves during training via SET/DEEP R algorithms.

## Development Commands

```powershell
# Environment setup
python scripts/setup_environment.py

# Install dependencies
pip install -e ".[dev]"

# Launch JupyterLab
jupyter lab

# Run all tests
pytest tests/

# Run single test file
pytest tests/test_topology.py -v

# Run single test
pytest tests/test_topology.py::test_watts_strogatz_basic -v

# Type checking
mypy src/

# Linting
flake8 src/
black src/ --check
isort src/ --check-only

# Web visualizations (in web/ directory)
npm install
npm run dev
```

## Architecture Overview

The codebase implements a modular multi-modal architecture with sparse small-world connectivity:

```
src/
  topology/          # Graph generation and sparse connectivity
    graphs.py        -> WS, ER, BA graph generators, adjacency conversions
    sparse_masks.py  -> Sparse mask generation (random, WS-based)
    rewiring.py      -> SET and DEEP R rewiring algorithms
    metrics.py       -> Clustering, path length, small-world coefficient

  models/            # Neural network components
    layers.py        -> SparseLinear, WSLinear, MixtureOfExperts
    encoders.py      -> VisualEncoder, TextEncoder, AudioEncoder
    connectors.py    -> WSConnector, LearnableWSConnector
    fusion.py        -> MultiModalFusion, AttentionFusion
    multimodal.py    -> MultiModalWSNetwork, SegmentedWSArchitecture

  training/          # Training infrastructure
    trainer.py       -> Base Trainer class with callbacks
    sparse_training.py -> SparseTrainer with SET/DEEP R rewiring
    callbacks.py     -> Logging, checkpointing, early stopping

  data/              # Data loading
    datasets.py      -> Multi-modal dataset classes
    transforms.py    -> Modality-specific transforms
    loaders.py       -> DataLoader factories

  visualization/     # Plotting utilities
    graph_viz.py     -> Network topology visualization
    training_viz.py  -> Loss curves, metrics
    topology_viz.py  -> Sparsity evolution, rewiring dynamics
```

## Key Patterns

**Sparse Layers**: `SparseLinear` multiplies weights by a binary mask. The mask can use random or WS topology and supports rewiring:
```python
from src.models.layers import SparseLinear
layer = SparseLinear(in_features=100, out_features=50, density=0.3, mask_type="ws")
layer.rewire(method="set", prune_rate=0.3)  # Dynamically rewire during training
```

**Graph Generation**: All graphs return numpy adjacency matrices:
```python
from src.topology.graphs import watts_strogatz_graph
adj = watts_strogatz_graph(n=100, k=4, beta=0.3, seed=42)
```

**Multi-Modal Forward Pass**: Models accept a dictionary of modality tensors:
```python
from src.models.multimodal import MultiModalWSNetwork
model = MultiModalWSNetwork(modalities=["visual", "text", "audio"])
output = model({"visual": img_tensor, "text": text_tensor, "audio": audio_tensor})
```

**Learnable Beta**: The `SegmentedWSArchitecture` stores beta as logits and uses sigmoid:
```python
model.betas  # Returns list of current beta values per layer
```

## Testing Fixtures

Key fixtures in `tests/conftest.py`:
- `device` - CUDA if available, else CPU
- `seed` - Sets torch/numpy seeds to 42
- `small_ws_graph` - Pre-generated 20-node WS graph
- `dummy_batch` - Multi-modal batch dict with visual/text/audio/label tensors
- `simple_model` - Basic 784->128->10 network
- `sparse_mask` - 100x50 random sparse mask at 30% density

## Configuration

`configs/default.yaml` contains all hyperparameters organized by category:
- `topology.watts_strogatz` - n_nodes, k_neighbors, beta
- `model.ws_connector` - learnable_beta, connections_per_module
- `sparse.set` - prune_rate, regrow_rate, update_frequency
- `training` - batch_size, learning_rate, epochs

Load config with:
```python
from src.utils.config import load_config
config = load_config("configs/default.yaml")
```

## Platform Requirements

- Windows (native PowerShell)
- GPU: NVIDIA 8GB+ VRAM recommended, CPU fallback required
- Notebook cells must execute within 30 seconds on single GPU

## Output Formatting

Do not use Unicode emoji. Use ASCII alternatives:
- `[OK]` instead of checkmarks
- `[X]` or `[FAIL]` instead of x-marks
- `[WARN]` or `[!]` instead of warning symbols
- `[->]` instead of arrows
