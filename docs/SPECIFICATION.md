# Technical Specification Document
## Multi-Modal Watts-Strogatz Foundational Training Program

**Version**: 1.0  
**Date**: 2025-01-09  
**Companion Document**: PRD.md  
**Target Implementer**: Claude Code  

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Development Environment Setup](#2-development-environment-setup)
3. [Module Specifications](#3-module-specifications)
4. [Visualization Components](#4-visualization-components)
5. [Core Algorithm Implementations](#5-core-algorithm-implementations)
6. [Data Pipeline](#6-data-pipeline)
7. [Testing Strategy](#7-testing-strategy)
8. [File Structure](#8-file-structure)
9. [Implementation Order](#9-implementation-order)
10. [Research Content Guidelines](#10-research-content-guidelines)
11. [Capstone Architecture Specification](#11-capstone-architecture-specification)
12. [API Reference](#12-api-reference)

---

## 1. Architecture Overview

### 1.1 System Components

```
multi-modal-WS-foundational-training/
│
├── notebooks/                    # JupyterLab notebooks (main content)
│   ├── 00_setup/                # Environment verification
│   ├── 01_foundations/          # Neural network basics
│   ├── 02_supervised/           # Supervised learning deep dive
│   ├── 03_graphs/               # Graph theory fundamentals
│   ├── 04_topology/             # Network topology models
│   ├── 05_sparse/               # Sparse neural networks
│   ├── 06_unsupervised/         # Unsupervised learning
│   ├── 07_dynamic_sparse/       # SET, DEEP R algorithms
│   ├── 08_modular/              # Modular architectures
│   ├── 09_multimodal/           # Multi-modal learning
│   ├── 10_capstone/             # Segmented WS architecture
│   └── exercises/               # Standalone exercises
│
├── src/                         # Python source code
│   ├── core/                    # Core utilities
│   ├── models/                  # Neural network implementations
│   ├── topology/                # Graph/topology algorithms
│   ├── training/                # Training loops and utilities
│   ├── visualization/           # Python-side viz helpers
│   └── data/                    # Data loading utilities
│
├── web/                         # React web application
│   ├── src/
│   │   ├── components/          # React components
│   │   ├── visualizers/         # D3/Plotly visualizations
│   │   ├── sandbox/             # Interactive sandbox
│   │   └── api/                 # Communication with Python backend
│   └── public/
│
├── tests/                       # Test suites
├── data/                        # Datasets (gitignored, downloaded)
├── docs/                        # Documentation
├── configs/                     # Configuration files
└── scripts/                     # Utility scripts
```

### 1.2 Technology Stack Details

#### Primary Stack (Implement in Full Detail)

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Language | Python | 3.11+ | Core implementation |
| ML Framework | PyTorch | 2.1+ | Model training |
| Notebooks | JupyterLab | 4.x | Content delivery |
| Web Framework | React | 18.x | Interactive UI |
| Graph Viz | D3.js | 7.x | Network visualizations |
| Charts | Plotly | 5.x | Training metrics |
| Widgets | ipywidgets | 8.x | Notebook interactivity |

#### Secondary Stack (Conceptual + Examples)

| Technology | Coverage | Example Count |
|------------|----------|---------------|
| TensorFlow/Keras | 1 full example per major concept | 5-6 total |
| JAX/Flax | Functional paradigm comparison | 2-3 total |
| Pure NumPy | Early fundamentals | Modules 1-2 |
| Three.js | Optional 3D viz | 1-2 demos |

### 1.3 Communication Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        JupyterLab                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Notebook Cells                                              ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       ││
│  │  │ Markdown │ │  Code    │ │ Widget   │ │  Output  │       ││
│  │  │ Content  │ │  Cell    │ │  Cell    │ │  Cell    │       ││
│  │  └──────────┘ └────┬─────┘ └────┬─────┘ └──────────┘       ││
│  └────────────────────┼────────────┼────────────────────────────┘│
│                       │            │                             │
│                       ▼            ▼                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              IPython Kernel (Python Backend)                │ │
│  │                                                             │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │ │
│  │  │   PyTorch   │  │  NetworkX   │  │   Plotly    │        │ │
│  │  │   Models    │  │   Graphs    │  │   Figures   │        │ │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │ │
│  └─────────┼────────────────┼────────────────┼─────────────────┘ │
│            │                │                │                   │
│            ▼                ▼                ▼                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    ipywidgets Bridge                         ││
│  │         (Bidirectional Python ↔ JavaScript)                 ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Web Visualization Components                    ││
│  │                                                              ││
│  │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   ││
│  │   │   D3.js  │  │  Plotly  │  │  React   │  │ Three.js │   ││
│  │   │  Graphs  │  │  Charts  │  │   UI     │  │   3D     │   ││
│  │   └──────────┘  └──────────┘  └──────────┘  └──────────┘   ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Development Environment Setup

### 2.1 Prerequisites Installation Script

Create `scripts/setup_environment.py`:

```python
#!/usr/bin/env python3
"""
Environment setup script for Multi-Modal WS Training Program.
Run: python scripts/setup_environment.py
"""

import subprocess
import sys
import platform
from pathlib import Path

def check_python_version():
    """Ensure Python 3.11+"""
    if sys.version_info < (3, 11):
        print("❌ Python 3.11+ required")
        print(f"   Current: {sys.version}")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")

def check_gpu():
    """Check for CUDA-capable GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU: {gpu_name} ({vram:.1f} GB)")
            return True
        else:
            print("⚠ No CUDA GPU detected - CPU mode enabled")
            return False
    except ImportError:
        print("⚠ PyTorch not installed yet")
        return None

def install_requirements():
    """Install Python dependencies"""
    requirements = Path(__file__).parent.parent / "requirements.txt"
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", str(requirements)
    ])

def setup_jupyter_extensions():
    """Install and enable JupyterLab extensions"""
    extensions = [
        "jupyterlab-plotly",
        "@jupyter-widgets/jupyterlab-manager",
    ]
    for ext in extensions:
        subprocess.run([
            sys.executable, "-m", "jupyter", "labextension", "install", ext
        ], capture_output=True)

def verify_installation():
    """Verify all components are working"""
    checks = [
        ("PyTorch", "import torch; print(torch.__version__)"),
        ("NetworkX", "import networkx; print(networkx.__version__)"),
        ("Plotly", "import plotly; print(plotly.__version__)"),
        ("ipywidgets", "import ipywidgets; print(ipywidgets.__version__)"),
    ]
    
    for name, cmd in checks:
        try:
            result = subprocess.run(
                [sys.executable, "-c", cmd],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"✓ {name}: {result.stdout.strip()}")
            else:
                print(f"❌ {name}: {result.stderr}")
        except Exception as e:
            print(f"❌ {name}: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("Multi-Modal WS Training - Environment Setup")
    print("=" * 50)
    
    check_python_version()
    check_gpu()
    
    print("\nInstalling dependencies...")
    install_requirements()
    
    print("\nSetting up Jupyter extensions...")
    setup_jupyter_extensions()
    
    print("\nVerifying installation...")
    verify_installation()
    
    print("\n" + "=" * 50)
    print("Setup complete! Run: jupyter lab")
    print("=" * 50)
```

### 2.2 Requirements File

Create `requirements.txt`:

```
# Core ML
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0

# Graph and Network
networkx>=3.2
torch-geometric>=2.4.0
scipy>=1.11.0

# Visualization
matplotlib>=3.8.0
plotly>=5.18.0
bokeh>=3.3.0
seaborn>=0.13.0

# Jupyter
jupyterlab>=4.0.0
ipywidgets>=8.1.0
voila>=0.5.0
notebook>=7.0.0

# Utilities
numpy>=1.26.0
pandas>=2.1.0
tqdm>=4.66.0
einops>=0.7.0
rich>=13.7.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Type checking
mypy>=1.7.0

# Documentation
mkdocs>=1.5.0
mkdocs-material>=9.5.0

# Development
black>=23.12.0
isort>=5.13.0
flake8>=6.1.0

# Optional: Alternative frameworks for comparison
# tensorflow>=2.15.0  # Uncomment for TF examples
# jax>=0.4.23         # Uncomment for JAX examples
# flax>=0.8.0         # Uncomment for Flax examples
```

### 2.3 Configuration Files

Create `configs/default.yaml`:

```yaml
# Multi-Modal WS Training Configuration

environment:
  device: "auto"  # auto, cuda, cpu
  seed: 42
  deterministic: true

training:
  default_epochs: 10
  default_batch_size: 64
  checkpoint_dir: "./checkpoints"
  log_dir: "./logs"

visualization:
  theme: "plotly_white"
  default_colors:
    - "#1f77b4"  # Blue
    - "#ff7f0e"  # Orange
    - "#2ca02c"  # Green
    - "#d62728"  # Red
    - "#9467bd"  # Purple
  animation_fps: 30
  max_nodes_display: 500

topology:
  default_n_nodes: 100
  default_k_neighbors: 4
  default_rewiring_prob: 0.1

sparse_training:
  initial_sparsity: 0.9
  final_sparsity: 0.95
  pruning_frequency: 100  # steps

progress:
  save_file: "./progress.json"
  auto_save: true
```

---

## 3. Module Specifications

### 3.1 Module 00: Setup and Orientation (1 hour)

**Files:**
- `notebooks/00_setup/00_welcome.ipynb`
- `notebooks/00_setup/01_environment_check.ipynb`
- `notebooks/00_setup/02_learning_path.ipynb`

**Content:**

```markdown
# 00_welcome.ipynb

## Learning Objectives
- Understand the program structure
- Verify environment setup
- Preview the capstone project

## Sections
1. Welcome and Introduction
   - What you'll learn
   - Why this matters (efficiency, multi-modal AI, novel architectures)
   - The journey ahead

2. Your Capstone Preview
   - Show the final architecture diagram
   - Explain: "By the end, you'll build this from scratch"
   - Interactive demo of a pre-built version

3. Navigation Guide
   - How notebooks are structured
   - How to use interactive elements
   - Where to find help
```

**Implementation Notes:**
- Include a working demo of the capstone architecture
- Add a "run all checks" button that validates environment
- Store initial timestamp for progress tracking

---

### 3.2 Module 01: Neural Network Foundations (3 hours)

**Files:**
- `notebooks/01_foundations/01_what_is_nn.ipynb`
- `notebooks/01_foundations/02_forward_pass.ipynb`
- `notebooks/01_foundations/03_activation_functions.ipynb`
- `notebooks/01_foundations/04_from_scratch.ipynb`
- `notebooks/01_foundations/quiz_01.ipynb`

**Content Outline:**

```markdown
# 01_what_is_nn.ipynb

## Learning Objectives
- Visualize neurons as mathematical functions
- Understand layers as transformations
- See how networks compose functions

## Sections

### 1. The Neuron as a Function
- Interactive: Single neuron with adjustable weights
- Visualization: Input → Weighted Sum → Output
- Analogy: Neuron as a "voting machine"

### 2. Stacking Neurons into Layers
- Interactive: 2-layer network visualization
- Show information flow through layers
- Introduce: Dense (fully connected) concept

### 3. Networks as Function Composition
- Mathematical view: f(g(h(x)))
- Visual: Data transformation through layers
- Preview: Why topology matters (connections determine capabilities)
```

**Key Visualizations:**
1. **Single Neuron Playground**: Adjustable weights, bias, activation
2. **Layer Flow Diagram**: Animated data flow through network
3. **Function Composition**: Side-by-side math and visual

**Exercise (04_from_scratch.ipynb):**
```python
# Exercise: Implement a 2-layer neural network using only NumPy

import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        TODO: Initialize weights and biases
        - self.W1: shape (input_size, hidden_size)
        - self.b1: shape (hidden_size,)
        - self.W2: shape (hidden_size, output_size)
        - self.b2: shape (output_size,)
        
        Hint: Use np.random.randn() scaled by 0.01
        """
        # YOUR CODE HERE
        pass
    
    def relu(self, x):
        """
        TODO: Implement ReLU activation
        ReLU(x) = max(0, x)
        """
        # YOUR CODE HERE
        pass
    
    def forward(self, x):
        """
        TODO: Implement forward pass
        1. Linear transform: z1 = x @ W1 + b1
        2. Activation: a1 = relu(z1)
        3. Linear transform: z2 = a1 @ W2 + b2
        4. Return z2
        """
        # YOUR CODE HERE
        pass

# Test your implementation
def test_network():
    net = SimpleNeuralNetwork(10, 20, 5)
    x = np.random.randn(32, 10)  # Batch of 32 samples
    output = net.forward(x)
    assert output.shape == (32, 5), f"Expected (32, 5), got {output.shape}"
    print("✓ Forward pass works!")

test_network()
```

---

### 3.3 Module 02: Supervised Learning Deep Dive (4 hours)

**Files:**
- `notebooks/02_supervised/01_loss_functions.ipynb`
- `notebooks/02_supervised/02_gradient_descent.ipynb`
- `notebooks/02_supervised/03_backprop_intuition.ipynb`
- `notebooks/02_supervised/04_pytorch_intro.ipynb`
- `notebooks/02_supervised/05_training_loop.ipynb`
- `notebooks/02_supervised/06_overfitting.ipynb`
- `notebooks/02_supervised/lab_mnist.ipynb`
- `notebooks/02_supervised/quiz_02.ipynb`

**Key Concepts:**

```markdown
# 02_gradient_descent.ipynb

## Visualization: The Loss Landscape

### Interactive Element: 3D Loss Surface
- Plotly 3D surface plot of loss landscape
- Animated ball rolling down gradient
- Controls:
  - Learning rate slider (see overshooting)
  - Starting point selector
  - Momentum toggle

### Code Implementation:
```python
# Interactive gradient descent visualization
from src.visualization import LossLandscapeViz

viz = LossLandscapeViz(
    loss_function="rosenbrock",  # or "quadratic", "saddle"
    dimensions=2
)

# Widget controls
learning_rate = widgets.FloatSlider(min=0.001, max=0.5, value=0.01)
momentum = widgets.FloatSlider(min=0.0, max=0.99, value=0.0)

viz.animate_optimization(
    optimizer="sgd",
    learning_rate=learning_rate,
    momentum=momentum,
    steps=100
)
```

### Key Insight Boxes:
- "Why learning rate matters" (too high = overshoot, too low = slow)
- "Local minima vs global minima" (preview of topology relevance)
- "The role of randomness" (stochastic gradient descent)
```

**Lab: MNIST Classification**
```python
# lab_mnist.ipynb - Full implementation with scaffolding

"""
Lab: Train a neural network on MNIST

You will:
1. Load and visualize the MNIST dataset
2. Build a simple feedforward network
3. Implement training loop
4. Evaluate and visualize results
5. Experiment with hyperparameters

Estimated time: 45 minutes
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from src.training import TrainingDashboard

# Step 1: Data Loading (PROVIDED)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, transform=transform)

# Step 2: Define Your Network
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        """
        TODO: Define layers
        - Flatten: 28x28 → 784
        - Linear: 784 → 256
        - ReLU
        - Linear: 256 → 128
        - ReLU
        - Linear: 128 → 10
        """
        # YOUR CODE HERE
        
    def forward(self, x):
        """
        TODO: Implement forward pass
        """
        # YOUR CODE HERE
        pass

# Step 3: Training Loop
def train_epoch(model, dataloader, optimizer, criterion):
    """
    TODO: Implement one training epoch
    
    For each batch:
    1. Zero gradients
    2. Forward pass
    3. Compute loss
    4. Backward pass
    5. Update weights
    
    Return: average loss for epoch
    """
    # YOUR CODE HERE
    pass

# Step 4: Evaluation
def evaluate(model, dataloader):
    """
    TODO: Compute accuracy on test set
    """
    # YOUR CODE HERE
    pass

# Step 5: Training Dashboard (PROVIDED)
dashboard = TrainingDashboard()
model = MNISTClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    test_acc = evaluate(model, test_loader)
    dashboard.update(epoch, train_loss, test_acc)
```

---

### 3.4 Module 03: Graph Theory Fundamentals (3 hours)

**Files:**
- `notebooks/03_graphs/01_what_is_graph.ipynb`
- `notebooks/03_graphs/02_representations.ipynb`
- `notebooks/03_graphs/03_properties.ipynb`
- `notebooks/03_graphs/04_networkx_intro.ipynb`
- `notebooks/03_graphs/05_neural_nets_as_graphs.ipynb`
- `notebooks/03_graphs/quiz_03.ipynb`

**Core Visualizations:**

```python
# src/visualization/graph_basics.py

class GraphExplorer:
    """Interactive graph exploration widget"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.fig = go.FigureWidget()
        self._setup_controls()
    
    def add_node_interactive(self, x, y):
        """Add node at click position"""
        node_id = len(self.graph.nodes)
        self.graph.add_node(node_id, pos=(x, y))
        self._update_visualization()
    
    def add_edge_interactive(self, node1, node2):
        """Add edge between selected nodes"""
        self.graph.add_edge(node1, node2)
        self._update_visualization()
    
    def show_adjacency_matrix(self):
        """Display adjacency matrix alongside graph"""
        adj = nx.adjacency_matrix(self.graph).todense()
        # Show heatmap next to graph
        
    def compute_properties(self):
        """Show graph properties in real-time"""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "avg_clustering": nx.average_clustering(self.graph),
            "avg_path_length": self._safe_avg_path_length(),
        }
```

**Key Content: Neural Networks as Graphs**

```markdown
# 05_neural_nets_as_graphs.ipynb

## The Big Insight

Every neural network IS a graph:
- Neurons → Nodes
- Weights → Edges
- Layer structure → Graph topology

### Interactive: See Your Network as a Graph

```python
from src.topology import NetworkToGraph

# Create a simple network
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 5)
)

# Convert to graph
graph = NetworkToGraph(model)
graph.visualize()  # Shows network as bipartite graph
graph.show_properties()  # Clustering, path lengths, etc.
```

### Question to Ponder
"If neural networks are graphs, can we use graph theory to design BETTER networks?"

This is the central question of this entire program. The answer is YES.
```

---

### 3.5 Module 04: Network Topology Models (4 hours)

**Files:**
- `notebooks/04_topology/01_random_graphs.ipynb`
- `notebooks/04_topology/02_watts_strogatz.ipynb`
- `notebooks/04_topology/03_scale_free.ipynb`
- `notebooks/04_topology/04_comparing_topologies.ipynb`
- `notebooks/04_topology/05_small_world_properties.ipynb`
- `notebooks/04_topology/lab_topology_explorer.ipynb`
- `notebooks/04_topology/quiz_04.ipynb`

**Critical Implementation: Watts-Strogatz Visualizer**

```python
# src/visualization/watts_strogatz.py

class WattsStrogatzVisualizer:
    """
    Interactive Watts-Strogatz model exploration.
    
    This is a CORE visualization for the entire program.
    Must support:
    - Real-time rewiring animation
    - Property computation (clustering, path length)
    - Comparison with random graphs
    - Export to PyTorch sparse tensor
    """
    
    def __init__(self, n_nodes=100, k_neighbors=4, rewiring_prob=0.1):
        self.n = n_nodes
        self.k = k_neighbors
        self.p = rewiring_prob
        self.graph = self._generate_ws_graph()
        self.history = []  # Track rewiring history
        
    def _generate_ws_graph(self):
        """Generate Watts-Strogatz graph"""
        return nx.watts_strogatz_graph(self.n, self.k, self.p)
    
    def animate_rewiring(self, target_p, steps=50):
        """
        Animate the rewiring process from p=0 to target_p.
        
        Shows:
        - Graph visualization updating
        - Clustering coefficient decreasing
        - Path length decreasing
        - Small-world coefficient (σ) changing
        """
        p_values = np.linspace(0, target_p, steps)
        
        for p in p_values:
            self.p = p
            self.graph = self._generate_ws_graph()
            
            metrics = {
                'p': p,
                'clustering': nx.average_clustering(self.graph),
                'path_length': nx.average_shortest_path_length(self.graph),
                'sw_coefficient': self._compute_sw_coefficient()
            }
            self.history.append(metrics)
            
            yield self._render_frame(metrics)
    
    def _compute_sw_coefficient(self):
        """
        Compute small-world coefficient σ = (C/C_rand) / (L/L_rand)
        
        σ > 1 indicates small-world properties
        Optimal is around 4.8 per research
        """
        # Generate equivalent random graph for comparison
        random_graph = nx.erdos_renyi_graph(self.n, nx.density(self.graph))
        
        C = nx.average_clustering(self.graph)
        C_rand = nx.average_clustering(random_graph)
        
        L = nx.average_shortest_path_length(self.graph)
        L_rand = nx.average_shortest_path_length(random_graph)
        
        if C_rand == 0 or L_rand == 0:
            return float('inf')
        
        return (C / C_rand) / (L / L_rand)
    
    def show_rewiring_effect(self):
        """
        Side-by-side comparison:
        - Regular lattice (p=0)
        - Small-world (p=0.1)
        - Random (p=1)
        """
        configs = [
            ("Regular Lattice", 0.0),
            ("Small World", 0.1),
            ("Random", 1.0)
        ]
        
        fig = make_subplots(rows=1, cols=3, subplot_titles=[c[0] for c in configs])
        
        for i, (name, p) in enumerate(configs):
            g = nx.watts_strogatz_graph(self.n, self.k, p)
            # Add graph visualization to subplot
            
        return fig
    
    def to_sparse_tensor(self):
        """Convert current graph to PyTorch sparse tensor"""
        adj = nx.adjacency_matrix(self.graph)
        indices = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
        values = torch.ones(indices.shape[1])
        return torch.sparse_coo_tensor(indices, values, (self.n, self.n))
```

**Content: The Small-World Sweet Spot**

```markdown
# 05_small_world_properties.ipynb

## The Magic of β ≈ 0.1

### Key Discovery (Watts & Strogatz, 1998)

At low rewiring probability (β ≈ 0.1):
- Clustering coefficient remains HIGH (like regular lattice)
- Path length becomes LOW (like random graph)
- Best of both worlds!

### Interactive Demonstration

```python
viz = WattsStrogatzVisualizer(n_nodes=100, k_neighbors=4)

# Widget: Beta slider
beta_slider = widgets.FloatSlider(
    min=0, max=1, step=0.01, value=0.1,
    description='β (rewiring):'
)

# Live metrics display
@beta_slider.observe
def on_beta_change(change):
    viz.p = change['new']
    viz.update()
    metrics_display.value = f"""
    Clustering: {viz.clustering:.3f}
    Path Length: {viz.path_length:.2f}
    SW Coefficient: {viz.sw_coefficient:.2f}
    """
```

### Why This Matters for Neural Networks

1. **High Clustering** → Related neurons stay connected
   - Enables specialized local processing
   - Like "modules" that work on specific features

2. **Short Paths** → Any neuron can influence any other quickly
   - Enables global information integration
   - Like "highways" between modules

3. **Efficiency** → More capability with fewer connections
   - Sparse networks with dense-network performance
   - Lower memory, faster inference
```

---

### 3.6 Module 05: Sparse Neural Networks (4 hours)

**Files:**
- `notebooks/05_sparse/01_why_sparse.ipynb`
- `notebooks/05_sparse/02_pruning_basics.ipynb`
- `notebooks/05_sparse/03_sparse_representations.ipynb`
- `notebooks/05_sparse/04_pytorch_sparse.ipynb`
- `notebooks/05_sparse/05_sparse_vs_dense.ipynb`
- `notebooks/05_sparse/lab_pruning.ipynb`
- `notebooks/05_sparse/quiz_05.ipynb`

**Key Implementation: Sparse Layer**

```python
# src/models/sparse_layers.py

class SparseLinear(nn.Module):
    """
    Linear layer with sparse connectivity.
    
    This is foundational for all later modules.
    Must support:
    - Initialization from adjacency matrix
    - Efficient forward pass using sparse ops
    - Mask-based gradient updates
    - Visualization of connectivity pattern
    """
    
    def __init__(self, in_features, out_features, sparsity=0.9, 
                 topology='random'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        
        # Initialize sparse mask
        self.register_buffer('mask', self._create_mask(topology))
        
        # Initialize weights (only where mask=1)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        self._init_weights()
    
    def _create_mask(self, topology):
        """
        Create connectivity mask based on topology type.
        
        Args:
            topology: 'random', 'watts_strogatz', 'scale_free', 
                     'bipartite_sw', or custom adjacency matrix
        """
        total_connections = self.in_features * self.out_features
        n_connections = int(total_connections * (1 - self.sparsity))
        
        if topology == 'random':
            # Erdős–Rényi random
            mask = torch.zeros(self.out_features, self.in_features)
            indices = torch.randperm(total_connections)[:n_connections]
            mask.view(-1)[indices] = 1
            
        elif topology == 'watts_strogatz':
            # Create WS graph and extract adjacency
            mask = self._ws_bipartite_mask()
            
        elif topology == 'scale_free':
            # Barabási-Albert
            mask = self._ba_bipartite_mask()
            
        elif isinstance(topology, torch.Tensor):
            # Custom adjacency matrix
            mask = topology
            
        return mask
    
    def _ws_bipartite_mask(self, k=4, p=0.1):
        """
        Create Watts-Strogatz-inspired bipartite connectivity.
        
        Based on BSW (Bipartite Small-World) from Zhang et al. 2023:
        - Each output node connects to k nearest input nodes
        - With probability p, rewire to random input node
        """
        mask = torch.zeros(self.out_features, self.in_features)
        
        for i in range(self.out_features):
            # Connect to k nearest neighbors (circular)
            center = int(i * self.in_features / self.out_features)
            for j in range(-k//2, k//2 + 1):
                idx = (center + j) % self.in_features
                
                # Rewire with probability p
                if torch.rand(1) < p:
                    idx = torch.randint(0, self.in_features, (1,)).item()
                
                mask[i, idx] = 1
        
        return mask
    
    def forward(self, x):
        """Forward pass with masked weights"""
        # Apply mask to weights
        masked_weight = self.weight * self.mask
        return F.linear(x, masked_weight, self.bias)
    
    def visualize_connectivity(self):
        """Show connectivity pattern as bipartite graph"""
        from src.visualization import BipartiteGraphViz
        return BipartiteGraphViz(self.mask).render()
    
    @property
    def actual_sparsity(self):
        """Compute actual sparsity of current mask"""
        return 1 - (self.mask.sum() / self.mask.numel()).item()
```

**Lab: Pruning Experiment**

```python
# lab_pruning.ipynb

"""
Lab: Compare Dense vs Pruned Networks

You will:
1. Train a dense network on CIFAR-10
2. Prune it to various sparsity levels
3. Fine-tune pruned networks
4. Compare accuracy vs parameters vs inference time

Key insight: Networks can be 90%+ sparse with minimal accuracy loss!
"""

from src.models import DenseNet, PrunedNet
from src.training import train, evaluate
from src.visualization import PruningDashboard

# Step 1: Train dense baseline
dense_model = DenseNet(num_classes=10)
train(dense_model, train_loader, epochs=20)
dense_accuracy = evaluate(dense_model, test_loader)

# Step 2: Prune to different sparsity levels
sparsities = [0.5, 0.7, 0.9, 0.95, 0.99]
results = []

for sparsity in sparsities:
    # TODO: Implement magnitude pruning
    pruned_model = prune_by_magnitude(dense_model, sparsity)
    
    # TODO: Fine-tune for 5 epochs
    train(pruned_model, train_loader, epochs=5)
    
    # Evaluate
    acc = evaluate(pruned_model, test_loader)
    params = count_nonzero_params(pruned_model)
    
    results.append({
        'sparsity': sparsity,
        'accuracy': acc,
        'parameters': params
    })

# Step 3: Visualize results
dashboard = PruningDashboard(results)
dashboard.show_accuracy_vs_sparsity()
dashboard.show_parameter_reduction()
```

---

### 3.7 Module 06: Unsupervised Learning (3 hours)

**Files:**
- `notebooks/06_unsupervised/01_why_unsupervised.ipynb`
- `notebooks/06_unsupervised/02_autoencoders.ipynb`
- `notebooks/06_unsupervised/03_latent_spaces.ipynb`
- `notebooks/06_unsupervised/04_contrastive_learning.ipynb`
- `notebooks/06_unsupervised/05_representation_quality.ipynb`
- `notebooks/06_unsupervised/lab_autoencoder.ipynb`
- `notebooks/06_unsupervised/quiz_06.ipynb`

**Key Visualization: Latent Space Explorer**

```python
# src/visualization/latent_explorer.py

class LatentSpaceExplorer:
    """
    Interactive exploration of learned latent spaces.
    
    Features:
    - 2D/3D visualization of encodings (t-SNE, UMAP, PCA)
    - Interpolation between points
    - Decoding from arbitrary latent points
    - Clustering visualization
    """
    
    def __init__(self, autoencoder, data_loader):
        self.model = autoencoder
        self.data = data_loader
        self.encodings = None
        self.labels = None
        
    def encode_dataset(self):
        """Encode all data points"""
        encodings = []
        labels = []
        
        with torch.no_grad():
            for x, y in self.data:
                z = self.model.encode(x)
                encodings.append(z)
                labels.append(y)
        
        self.encodings = torch.cat(encodings)
        self.labels = torch.cat(labels)
    
    def visualize_2d(self, method='tsne'):
        """Project to 2D and visualize"""
        if method == 'tsne':
            from sklearn.manifold import TSNE
            projected = TSNE(n_components=2).fit_transform(
                self.encodings.numpy()
            )
        elif method == 'umap':
            import umap
            projected = umap.UMAP(n_components=2).fit_transform(
                self.encodings.numpy()
            )
        
        # Interactive plotly scatter
        fig = px.scatter(
            x=projected[:, 0], y=projected[:, 1],
            color=self.labels.numpy(),
            hover_data=['label'],
            title=f'Latent Space ({method.upper()})'
        )
        return fig
    
    def interpolate(self, idx1, idx2, steps=10):
        """
        Interpolate between two latent points and decode.
        Shows how representation changes smoothly.
        """
        z1 = self.encodings[idx1]
        z2 = self.encodings[idx2]
        
        interpolated = []
        for alpha in np.linspace(0, 1, steps):
            z = (1 - alpha) * z1 + alpha * z2
            decoded = self.model.decode(z.unsqueeze(0))
            interpolated.append(decoded)
        
        return torch.stack(interpolated)
    
    def interactive_decoder(self):
        """
        Widget to explore decoding from arbitrary latent points.
        Click in 2D projection to decode that point.
        """
        # Returns ipywidget with click-to-decode functionality
        pass
```

**Connection to Architecture Design:**

```markdown
# 05_representation_quality.ipynb

## Why Unsupervised Matters for Architecture Design

### Key Insight

Unsupervised learning reveals how networks SELF-ORGANIZE their internal 
representations. This is directly relevant to:

1. **Sparse Training**: Which connections naturally become important?
2. **Modular Design**: Do representations cluster into modules?
3. **Multi-Modal**: How do different modalities share representation space?

### Experiment: Representation Quality vs Topology

```python
# Compare representation quality for different network topologies

topologies = ['dense', 'random_sparse', 'watts_strogatz', 'scale_free']
results = {}

for topology in topologies:
    # Create autoencoder with this topology
    ae = SparseAutoencoder(topology=topology, sparsity=0.9)
    
    # Train unsupervised
    train_autoencoder(ae, unlabeled_data, epochs=50)
    
    # Evaluate representation quality
    results[topology] = {
        'reconstruction_loss': evaluate_reconstruction(ae, test_data),
        'linear_probe_accuracy': linear_probe(ae.encoder, labeled_data),
        'clustering_nmi': clustering_quality(ae, test_data)
    }

# Visualization: Bar chart comparing topologies
```

### Preview: The Small-World Advantage

Small-world topology tends to produce:
- Better clustering in latent space (high local clustering)
- Faster training convergence (short paths)
- More robust representations (redundant pathways)

We'll explore this in depth in Module 07.
```

---

### 3.8 Module 07: Dynamic Sparse Training (5 hours)

**Files:**
- `notebooks/07_dynamic_sparse/01_static_vs_dynamic.ipynb`
- `notebooks/07_dynamic_sparse/02_set_algorithm.ipynb`
- `notebooks/07_dynamic_sparse/03_set_implementation.ipynb`
- `notebooks/07_dynamic_sparse/04_deep_r.ipynb`
- `notebooks/07_dynamic_sparse/05_deep_r_implementation.ipynb`
- `notebooks/07_dynamic_sparse/06_bsw_init.ipynb`
- `notebooks/07_dynamic_sparse/07_topology_evolution.ipynb`
- `notebooks/07_dynamic_sparse/lab_dynamic_training.ipynb`
- `notebooks/07_dynamic_sparse/quiz_07.ipynb`

**Core Algorithm: SET (Sparse Evolutionary Training)**

```python
# src/training/set_algorithm.py

class SETTrainer:
    """
    Sparse Evolutionary Training (Mocanu et al., 2018)
    
    Key idea: Start sparse, stay sparse, but EVOLVE the connections.
    
    Algorithm:
    1. Initialize with random sparse connectivity
    2. Train for T steps
    3. Remove fraction of smallest-magnitude weights
    4. Regrow same number of connections randomly
    5. Repeat from step 2
    
    Result: Network discovers optimal sparse topology through evolution!
    """
    
    def __init__(self, model, sparsity=0.9, prune_rate=0.3, 
                 regrow_rate=0.3, update_frequency=100):
        """
        Args:
            model: Neural network with SparseLinear layers
            sparsity: Target sparsity level
            prune_rate: Fraction of existing connections to prune
            regrow_rate: Fraction of pruned connections to regrow
            update_frequency: Steps between topology updates
        """
        self.model = model
        self.sparsity = sparsity
        self.prune_rate = prune_rate
        self.regrow_rate = regrow_rate
        self.update_frequency = update_frequency
        self.step_count = 0
        
        # Tracking for visualization
        self.topology_history = []
        
    def step(self):
        """Called after each training step"""
        self.step_count += 1
        
        if self.step_count % self.update_frequency == 0:
            self._update_topology()
    
    def _update_topology(self):
        """Prune and regrow connections"""
        for name, module in self.model.named_modules():
            if isinstance(module, SparseLinear):
                self._update_layer(module, name)
    
    def _update_layer(self, layer, name):
        """Update topology for a single layer"""
        mask = layer.mask
        weight = layer.weight.data
        
        # Get current active connections
        active_mask = mask.bool()
        active_weights = weight[active_mask]
        
        # Determine number to prune
        n_active = active_mask.sum().item()
        n_prune = int(n_active * self.prune_rate)
        
        # Find smallest magnitude weights
        magnitudes = active_weights.abs()
        threshold = torch.kthvalue(magnitudes, n_prune).values
        
        # Prune: set mask to 0 where weight magnitude is below threshold
        prune_mask = (weight.abs() < threshold) & active_mask
        mask[prune_mask] = 0
        
        # Regrow: add new random connections
        inactive_mask = ~mask.bool()
        n_inactive = inactive_mask.sum().item()
        n_regrow = min(n_prune, n_inactive)  # Can't regrow more than available
        
        if n_regrow > 0:
            # Random selection from inactive positions
            inactive_indices = inactive_mask.nonzero()
            regrow_indices = inactive_indices[
                torch.randperm(len(inactive_indices))[:n_regrow]
            ]
            
            for idx in regrow_indices:
                mask[idx[0], idx[1]] = 1
                # Initialize new weight
                weight[idx[0], idx[1]] = torch.randn(1) * 0.01
        
        # Record for visualization
        self.topology_history.append({
            'step': self.step_count,
            'layer': name,
            'n_pruned': n_prune,
            'n_regrown': n_regrow,
            'sparsity': 1 - mask.sum().item() / mask.numel()
        })
    
    def visualize_evolution(self, layer_name=None):
        """
        Show how topology evolved during training.
        
        Displays:
        - Sparsity over time
        - Connection churn (pruned + regrown per step)
        - Degree distribution changes
        - Clustering coefficient changes
        """
        from src.visualization import TopologyEvolutionViz
        return TopologyEvolutionViz(self.topology_history, layer_name)
```

**Core Algorithm: DEEP R (Deep Rewiring)**

```python
# src/training/deep_r.py

class DeepRewiring:
    """
    Deep Rewiring (Bellec et al., 2017)
    
    Key idea: Treat connections as random variables, sample network
    configurations from a posterior distribution.
    
    Differences from SET:
    - Stochastic: connections have probability of being active
    - Gradient-informed: uses gradient information for rewiring
    - Sign-based: connections can be dormant (0), positive, or negative
    
    Algorithm:
    1. Each connection has a 'temperature' parameter θ
    2. Connection is active with probability σ(|θ|)
    3. Sign of active connection is sign(θ)
    4. Gradients flow through active connections only
    5. θ is updated based on gradient magnitude
    """
    
    def __init__(self, model, target_sparsity=0.9, 
                 dormant_threshold=0.01, rewire_temperature=1.0):
        self.model = model
        self.target_sparsity = target_sparsity
        self.dormant_threshold = dormant_threshold
        self.temperature = rewire_temperature
        
        # Initialize connection parameters
        self._init_connection_params()
    
    def _init_connection_params(self):
        """Initialize θ parameters for each connection"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Add θ parameter with same shape as weight
                theta = torch.randn_like(module.weight) * 0.1
                module.register_parameter('theta', nn.Parameter(theta))
    
    def sample_topology(self):
        """
        Sample active topology from current θ parameters.
        
        Returns masks for each layer.
        """
        masks = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'theta'):
                # Probability of being active
                prob = torch.sigmoid(module.theta.abs() / self.temperature)
                
                # Sample mask (Bernoulli)
                mask = torch.bernoulli(prob)
                
                # Enforce sparsity constraint
                mask = self._enforce_sparsity(mask, prob)
                
                masks[name] = mask
        
        return masks
    
    def _enforce_sparsity(self, mask, prob):
        """Ensure mask has exactly (1-sparsity) active connections"""
        n_total = mask.numel()
        n_target = int(n_total * (1 - self.target_sparsity))
        n_current = mask.sum().item()
        
        if n_current > n_target:
            # Too many active: remove lowest probability ones
            active_probs = prob[mask.bool()]
            threshold = torch.kthvalue(active_probs, 
                                       int(n_current - n_target)).values
            mask[mask.bool() & (prob < threshold)] = 0
            
        elif n_current < n_target:
            # Too few active: add highest probability inactive ones
            inactive_probs = prob[~mask.bool()]
            threshold = torch.kthvalue(inactive_probs, 
                                       int(inactive_probs.numel() - (n_target - n_current))).values
            mask[~mask.bool() & (prob >= threshold)] = 1
        
        return mask
    
    def forward_with_mask(self, x, masks):
        """
        Forward pass using sampled masks.
        
        Note: This requires custom forward to apply masks.
        """
        # Implementation depends on model structure
        pass
    
    def update_theta(self, gradients):
        """
        Update θ based on gradient magnitude.
        
        High gradient magnitude → increase |θ| → more likely to be active
        Low gradient magnitude → decrease |θ| → may become dormant
        """
        for name, module in self.model.named_modules():
            if hasattr(module, 'theta'):
                # Get gradient for this layer's weight
                grad = gradients[name]
                
                # Update θ based on gradient magnitude
                # Connections with high gradient become more active
                delta_theta = self.temperature * grad.abs()
                
                # Preserve sign of θ
                module.theta.data += delta_theta * module.theta.sign()
```

**Visualization: Topology Evolution**

```python
# src/visualization/topology_evolution.py

class TopologyEvolutionViz:
    """
    Visualize how network topology evolves during training.
    
    This is CRUCIAL for understanding dynamic sparse training.
    Shows learner that networks discover their own optimal structure!
    """
    
    def __init__(self, model, trainer):
        self.model = model
        self.trainer = trainer
        self.snapshots = []  # Topology snapshots over training
    
    def capture_snapshot(self, step):
        """Capture current topology state"""
        snapshot = {
            'step': step,
            'layers': {}
        }
        
        for name, module in self.model.named_modules():
            if isinstance(module, SparseLinear):
                mask = module.mask.clone()
                weight = module.weight.data.clone()
                
                # Compute graph metrics
                adj = mask.cpu().numpy()
                
                snapshot['layers'][name] = {
                    'mask': mask,
                    'sparsity': 1 - mask.sum().item() / mask.numel(),
                    'weight_magnitude_mean': weight[mask.bool()].abs().mean().item(),
                    'degree_distribution': adj.sum(axis=1).tolist(),
                }
        
        self.snapshots.append(snapshot)
    
    def animate_evolution(self, layer_name):
        """
        Create animation showing topology changes over training.
        
        Shows:
        - Bipartite graph of connections
        - Connections being pruned (red flash)
        - Connections being added (green flash)
        - Clustering coefficient evolving
        """
        frames = []
        
        for i, snapshot in enumerate(self.snapshots):
            layer_data = snapshot['layers'][layer_name]
            
            # Create frame
            frame = self._render_bipartite_graph(
                layer_data['mask'],
                highlight_changes=i > 0,
                prev_mask=self.snapshots[i-1]['layers'][layer_name]['mask'] if i > 0 else None
            )
            frames.append(frame)
        
        return self._create_animation(frames)
    
    def plot_metrics_over_training(self):
        """
        Plot how topology metrics change during training.
        
        Metrics:
        - Sparsity (should stay constant with SET/DEEP R)
        - Average degree
        - Degree variance (does it become scale-free?)
        - Clustering coefficient (does it become small-world?)
        """
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=['Sparsity', 'Average Degree',
                                          'Degree Variance', 'Clustering'])
        
        # Extract metrics from snapshots
        steps = [s['step'] for s in self.snapshots]
        
        for layer_name in self.snapshots[0]['layers'].keys():
            # ... plot each metric
            pass
        
        return fig
    
    def compare_initial_vs_final(self, layer_name):
        """
        Side-by-side comparison of topology before and after training.
        
        Shows how network "learned" its structure!
        """
        initial = self.snapshots[0]['layers'][layer_name]
        final = self.snapshots[-1]['layers'][layer_name]
        
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=['Initial (Random)', 'Final (Learned)'])
        
        # Add visualizations
        # ...
        
        return fig
```

---

### 3.9 Module 08: Modular Architectures (4 hours)

**Files:**
- `notebooks/08_modular/01_why_modular.ipynb`
- `notebooks/08_modular/02_mixture_of_experts.ipynb`
- `notebooks/08_modular/03_gating_mechanisms.ipynb`
- `notebooks/08_modular/04_sparse_moe.ipynb`
- `notebooks/08_modular/05_module_specialization.ipynb`
- `notebooks/08_modular/06_inter_module_communication.ipynb`
- `notebooks/08_modular/lab_moe.ipynb`
- `notebooks/08_modular/quiz_08.ipynb`

**Key Implementation: Mixture of Experts**

```python
# src/models/mixture_of_experts.py

class Expert(nn.Module):
    """Single expert network"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class GatingNetwork(nn.Module):
    """Gating network that routes inputs to experts"""
    
    def __init__(self, input_dim, num_experts, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        # Compute expert scores
        scores = self.gate(x)  # (batch, num_experts)
        
        # Select top-k experts
        top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=-1)
        
        # Softmax over selected experts only
        top_k_weights = F.softmax(top_k_scores, dim=-1)
        
        return top_k_weights, top_k_indices


class SparseMixtureOfExperts(nn.Module):
    """
    Sparse Mixture of Experts layer.
    
    Key insight: Only activate top-k experts per input.
    This is a form of CONDITIONAL COMPUTATION - dynamic sparsity!
    
    Connection to small-world:
    - Experts are like "modules" with specialized function
    - Gating creates "short paths" to relevant experts
    - Overall sparse but powerful
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 num_experts=8, top_k=2):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim)
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = GatingNetwork(input_dim, num_experts, top_k)
        
        # Load balancing loss (prevents expert collapse)
        self.load_balance_weight = 0.01
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Get gating decisions
        weights, indices = self.gate(x)  # (batch, top_k), (batch, top_k)
        
        # Compute expert outputs (only for selected experts)
        expert_outputs = torch.zeros(batch_size, self.experts[0].net[-1].out_features,
                                    device=x.device)
        
        for i in range(self.top_k):
            expert_idx = indices[:, i]  # (batch,)
            expert_weight = weights[:, i:i+1]  # (batch, 1)
            
            # Group inputs by expert
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_out = self.experts[e](x[mask])
                    expert_outputs[mask] += expert_weight[mask] * expert_out
        
        # Compute load balancing loss
        load_balance_loss = self._compute_load_balance_loss(indices)
        
        return expert_outputs, load_balance_loss
    
    def _compute_load_balance_loss(self, indices):
        """
        Encourage uniform expert usage.
        
        Without this, the network often collapses to using only 1-2 experts.
        """
        # Count how many times each expert was selected
        expert_counts = torch.zeros(self.num_experts, device=indices.device)
        for e in range(self.num_experts):
            expert_counts[e] = (indices == e).sum()
        
        # Normalize to get usage distribution
        usage_dist = expert_counts / expert_counts.sum()
        
        # Uniform distribution
        uniform_dist = torch.ones_like(usage_dist) / self.num_experts
        
        # KL divergence from uniform
        loss = F.kl_div(usage_dist.log(), uniform_dist, reduction='sum')
        
        return loss * self.load_balance_weight
    
    def visualize_routing(self, x, labels=None):
        """
        Visualize which experts are selected for which inputs.
        
        Helps understand expert specialization!
        """
        from src.visualization import ExpertRoutingViz
        
        with torch.no_grad():
            weights, indices = self.gate(x)
        
        return ExpertRoutingViz(indices, labels).render()
```

**Key Content: Inter-Module Communication**

```markdown
# 06_inter_module_communication.ipynb

## The Challenge: How Do Modules Talk?

So far, modules are independent experts selected by a gate.
But what if we want modules to COMMUNICATE directly?

### Options for Inter-Module Connectivity

1. **Dense All-to-All**: Every module talks to every other
   - Simple but expensive O(n²)
   - Doesn't scale

2. **Fixed Topology**: Predetermined connections
   - Hand-designed patterns
   - May not match task structure

3. **Learned Sparse**: Learn which connections matter
   - Use attention or gating
   - Dynamic but costly to learn

4. **Small-World Topology**: Best of both worlds! ⭐
   - High clustering: nearby modules communicate densely
   - Short paths: distant modules reachable quickly
   - Efficient: sparse but effective

### Preview: Watts-Strogatz for Inter-Module Wiring

```python
# This is exactly what we'll build in the capstone!

class WSModularNetwork(nn.Module):
    def __init__(self, num_modules, k_neighbors, rewiring_prob):
        # Create specialized modules
        self.modules = nn.ModuleList([...])
        
        # Create WS topology for inter-module connections
        self.topology = nx.watts_strogatz_graph(
            num_modules, k_neighbors, rewiring_prob
        )
        
        # Learnable inter-module weights
        self.inter_module_weights = nn.ParameterDict()
        for edge in self.topology.edges():
            self.inter_module_weights[f"{edge[0]}_{edge[1]}"] = nn.Parameter(...)
```

This is THE KEY INNOVATION we're building toward!
```

---

### 3.10 Module 09: Multi-Modal Learning (4 hours)

**Files:**
- `notebooks/09_multimodal/01_what_is_multimodal.ipynb`
- `notebooks/09_multimodal/02_modality_encoders.ipynb`
- `notebooks/09_multimodal/03_fusion_strategies.ipynb`
- `notebooks/09_multimodal/04_cross_modal_attention.ipynb`
- `notebooks/09_multimodal/05_alignment_binding.ipynb`
- `notebooks/09_multimodal/06_multimodal_sparse.ipynb`
- `notebooks/09_multimodal/lab_vision_language.ipynb`
- `notebooks/09_multimodal/quiz_09.ipynb`

**Key Implementations:**

```python
# src/models/multimodal.py

class ModalityEncoder(nn.Module):
    """Base class for modality-specific encoders"""
    
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
    
    def forward(self, x):
        raise NotImplementedError


class ImageEncoder(ModalityEncoder):
    """CNN-based image encoder"""
    
    def __init__(self, output_dim=256):
        super().__init__(output_dim)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.backbone(x)


class TextEncoder(ModalityEncoder):
    """Simple transformer-based text encoder"""
    
    def __init__(self, vocab_size, output_dim=256, max_len=128):
        super().__init__(output_dim)
        self.embedding = nn.Embedding(vocab_size, output_dim)
        self.positional = nn.Parameter(torch.randn(max_len, output_dim) * 0.01)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(output_dim, nhead=4, batch_first=True),
            num_layers=2
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        # x: (batch, seq_len) token ids
        embedded = self.embedding(x) + self.positional[:x.shape[1]]
        encoded = self.transformer(embedded)
        pooled = self.pool(encoded.transpose(1, 2)).squeeze(-1)
        return pooled


class AudioEncoder(ModalityEncoder):
    """Simple audio encoder (mel spectrogram → CNN)"""
    
    def __init__(self, output_dim=256):
        super().__init__(output_dim)
        # Treat mel spectrogram as 1-channel image
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        # x: (batch, freq_bins, time_steps)
        x = x.unsqueeze(1)  # Add channel dim
        return self.backbone(x)


class MultiModalFusion(nn.Module):
    """
    Fusion strategies for combining modality representations.
    
    This demonstrates different approaches before we apply WS topology.
    """
    
    def __init__(self, modality_dim, output_dim, fusion_type='concat'):
        super().__init__()
        self.fusion_type = fusion_type
        self.modality_dim = modality_dim
        
        if fusion_type == 'concat':
            # Simple concatenation
            self.fusion = nn.Linear(modality_dim * 3, output_dim)
            
        elif fusion_type == 'attention':
            # Cross-modal attention
            self.cross_attn = nn.MultiheadAttention(modality_dim, num_heads=4)
            self.fusion = nn.Linear(modality_dim, output_dim)
            
        elif fusion_type == 'gated':
            # Gated fusion
            self.gates = nn.ModuleDict({
                'image': nn.Linear(modality_dim, 1),
                'text': nn.Linear(modality_dim, 1),
                'audio': nn.Linear(modality_dim, 1)
            })
            self.fusion = nn.Linear(modality_dim, output_dim)
    
    def forward(self, image_repr, text_repr, audio_repr):
        if self.fusion_type == 'concat':
            combined = torch.cat([image_repr, text_repr, audio_repr], dim=-1)
            return self.fusion(combined)
            
        elif self.fusion_type == 'attention':
            # Stack modalities as sequence
            modalities = torch.stack([image_repr, text_repr, audio_repr], dim=1)
            attended, _ = self.cross_attn(modalities, modalities, modalities)
            return self.fusion(attended.mean(dim=1))
            
        elif self.fusion_type == 'gated':
            # Compute gates
            g_image = torch.sigmoid(self.gates['image'](image_repr))
            g_text = torch.sigmoid(self.gates['text'](text_repr))
            g_audio = torch.sigmoid(self.gates['audio'](audio_repr))
            
            # Weighted combination
            combined = g_image * image_repr + g_text * text_repr + g_audio * audio_repr
            return self.fusion(combined)
```

**Content: Why Topology Matters for Multi-Modal**

```markdown
# 06_multimodal_sparse.ipynb

## The Multi-Modal Binding Problem

When fusing multiple modalities, we face challenges:

1. **Scale mismatch**: Images have millions of pixels, text has hundreds of tokens
2. **Semantic alignment**: Which image region corresponds to which word?
3. **Computational cost**: Dense cross-modal attention is O(n²)

### How Sparse Topology Helps

1. **Selective Attention**: Only connect related elements
   - Short paths between semantically related features
   - Ignore unrelated pairs

2. **Hierarchical Binding**: 
   - Local clusters process single modality
   - Cross-modal "shortcut" connections for binding

3. **Efficiency**:
   - Sparse attention: O(n) instead of O(n²)
   - Small-world: maintain global information flow

### Experiment: Dense vs Sparse Multi-Modal Fusion

```python
# Compare fusion approaches on image-text matching

dense_model = MultiModalFusion(fusion_type='attention')
sparse_model = SparseMultiModalFusion(topology='watts_strogatz')

# Train both on same data
results_dense = train_and_evaluate(dense_model, data)
results_sparse = train_and_evaluate(sparse_model, data)

# Compare
print(f"Dense accuracy: {results_dense['accuracy']:.2%}")
print(f"Sparse accuracy: {results_sparse['accuracy']:.2%}")
print(f"Dense params: {results_dense['params']:,}")
print(f"Sparse params: {results_sparse['params']:,}")  # Much fewer!
print(f"Sparse is {results_dense['params']/results_sparse['params']:.1f}x smaller")
```

### Key Insight

Small-world topology is particularly powerful for multi-modal:
- High clustering → modality-specific processing
- Short paths → rapid cross-modal information flow
- Sparse → computational efficiency

This is exactly what we'll build in the capstone!
```

---

### 3.11 Module 10: Capstone Project (5 hours)

**Files:**
- `notebooks/10_capstone/01_architecture_design.ipynb`
- `notebooks/10_capstone/02_module_implementation.ipynb`
- `notebooks/10_capstone/03_ws_connector.ipynb`
- `notebooks/10_capstone/04_learnable_rewiring.ipynb`
- `notebooks/10_capstone/05_training_integration.ipynb`
- `notebooks/10_capstone/06_evaluation.ipynb`
- `notebooks/10_capstone/07_ablation_study.ipynb`
- `notebooks/10_capstone/08_next_steps.ipynb`

**Complete Architecture Implementation:**

```python
# src/models/segmented_ws_architecture.py

"""
Segmented Watts-Strogatz Multi-Modal Architecture

This is the culmination of all learned concepts:
- Modular design (Module 08)
- Sparse connectivity (Module 05)
- Dynamic rewiring (Module 07)
- Multi-modal fusion (Module 09)
- Small-world topology (Module 04)

Architecture Overview:
┌─────────────────────────────────────────────────────────────────┐
│  Input: Image + Text + Audio                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                    │
│  │  Visual  │   │   Text   │   │  Audio   │                    │
│  │  Module  │   │  Module  │   │  Module  │   ← Modality       │
│  │  (CNN)   │   │ (Trans.) │   │  (RNN)   │     Encoders       │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘                    │
│       │              │              │                           │
│       └──────────────┼──────────────┘                           │
│                      │                                          │
│              ┌───────▼───────┐                                  │
│              │   WS Inter-   │  ← Learnable Small-World         │
│              │   Module      │    Connector with Dynamic        │
│              │   Connector   │    Rewiring                      │
│              └───────┬───────┘                                  │
│                      │                                          │
│              ┌───────▼───────┐                                  │
│              │    Fusion     │  ← Cross-Modal Integration       │
│              │    Module     │                                  │
│              └───────┬───────┘                                  │
│                      │                                          │
│              ┌───────▼───────┐                                  │
│              │    Output     │                                  │
│              │    Head       │                                  │
│              └───────────────┘                                  │
└─────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from typing import Dict, Optional, Tuple


class WSInterModuleConnector(nn.Module):
    """
    Watts-Strogatz Inter-Module Connector
    
    The KEY INNOVATION of this architecture.
    
    Connects module outputs using small-world topology:
    - High clustering: nearby modules (same/similar modality) connected densely
    - Short paths: any module can quickly communicate with any other
    - Learnable β: rewiring probability adapts during training
    - Dynamic: connections can evolve using SET-inspired updates
    """
    
    def __init__(
        self,
        num_modules: int,
        module_dim: int,
        k_neighbors: int = 2,
        initial_beta: float = 0.1,
        learnable_beta: bool = True,
        dynamic_rewiring: bool = True,
        rewiring_frequency: int = 100
    ):
        super().__init__()
        
        self.num_modules = num_modules
        self.module_dim = module_dim
        self.k = k_neighbors
        self.dynamic_rewiring = dynamic_rewiring
        self.rewiring_frequency = rewiring_frequency
        self.step_count = 0
        
        # Learnable rewiring probability
        if learnable_beta:
            # Use sigmoid to constrain to [0, 1]
            self._beta_logit = nn.Parameter(
                torch.tensor(self._inverse_sigmoid(initial_beta))
            )
        else:
            self.register_buffer('_beta_logit', 
                                torch.tensor(self._inverse_sigmoid(initial_beta)))
        
        # Initialize topology
        self._init_topology()
        
        # Connection weights
        self.connection_weights = nn.ParameterDict()
        self._init_connection_weights()
        
        # Optional: connection-specific projections
        self.use_projections = True
        if self.use_projections:
            self.projections = nn.ModuleDict()
            self._init_projections()
    
    @property
    def beta(self):
        """Current rewiring probability"""
        return torch.sigmoid(self._beta_logit)
    
    def _inverse_sigmoid(self, x):
        """Inverse of sigmoid for initialization"""
        return torch.log(torch.tensor(x / (1 - x)))
    
    def _init_topology(self):
        """Initialize Watts-Strogatz topology"""
        # Create WS graph
        self.graph = nx.watts_strogatz_graph(
            self.num_modules, self.k, self.beta.item()
        )
        
        # Store adjacency as buffer (non-learnable)
        adj = nx.adjacency_matrix(self.graph).todense()
        self.register_buffer('adjacency', torch.tensor(adj, dtype=torch.float))
        
        # Track which edges exist
        self.edge_list = list(self.graph.edges())
    
    def _init_connection_weights(self):
        """Initialize learnable weights for each connection"""
        for i, j in self.edge_list:
            key = f"{i}_{j}"
            # Scalar weight for this connection
            self.connection_weights[key] = nn.Parameter(torch.ones(1))
    
    def _init_projections(self):
        """Initialize projection matrices for each connection"""
        for i, j in self.edge_list:
            key = f"{i}_{j}"
            self.projections[key] = nn.Linear(self.module_dim, self.module_dim, bias=False)
    
    def forward(
        self, 
        module_outputs: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        """
        Forward pass through inter-module connections.
        
        Args:
            module_outputs: Dict mapping module index to output tensor
                           Each tensor has shape (batch, module_dim)
        
        Returns:
            Dict mapping module index to aggregated input from other modules
        """
        batch_size = next(iter(module_outputs.values())).shape[0]
        device = next(iter(module_outputs.values())).device
        
        # Initialize aggregated inputs
        aggregated = {
            i: torch.zeros(batch_size, self.module_dim, device=device)
            for i in range(self.num_modules)
        }
        
        # For each connection, send information
        for i, j in self.edge_list:
            key = f"{i}_{j}"
            weight = self.connection_weights[key]
            
            # Get source output
            source_out = module_outputs[i]
            
            # Apply projection if enabled
            if self.use_projections:
                projected = self.projections[key](source_out)
            else:
                projected = source_out
            
            # Add weighted contribution to target
            aggregated[j] = aggregated[j] + weight * projected
            
            # Bidirectional (undirected graph)
            aggregated[i] = aggregated[i] + weight * self.projections[key](module_outputs[j])
        
        return aggregated
    
    def update_topology(self):
        """
        Update topology using current beta value.
        
        Called during training if dynamic_rewiring is True.
        Implements SET-inspired evolution: prune weak, regrow random.
        """
        if not self.dynamic_rewiring:
            return
        
        self.step_count += 1
        if self.step_count % self.rewiring_frequency != 0:
            return
        
        # Regenerate WS graph with current (possibly updated) beta
        new_graph = nx.watts_strogatz_graph(
            self.num_modules, self.k, self.beta.item()
        )
        
        # Update edge list
        old_edges = set(self.edge_list)
        new_edges = set(new_graph.edges())
        
        # Edges to remove
        removed = old_edges - new_edges
        for i, j in removed:
            key = f"{i}_{j}"
            del self.connection_weights[key]
            if self.use_projections:
                del self.projections[key]
        
        # Edges to add
        added = new_edges - old_edges
        for i, j in added:
            key = f"{i}_{j}"
            self.connection_weights[key] = nn.Parameter(torch.ones(1))
            if self.use_projections:
                self.projections[key] = nn.Linear(
                    self.module_dim, self.module_dim, bias=False
                )
        
        self.graph = new_graph
        self.edge_list = list(new_graph.edges())
        
        # Update adjacency buffer
        adj = nx.adjacency_matrix(self.graph).todense()
        self.adjacency = torch.tensor(adj, dtype=torch.float, device=self.adjacency.device)
    
    def get_topology_metrics(self):
        """Return current topology metrics"""
        return {
            'beta': self.beta.item(),
            'num_edges': len(self.edge_list),
            'clustering': nx.average_clustering(self.graph),
            'avg_path_length': nx.average_shortest_path_length(self.graph) 
                              if nx.is_connected(self.graph) else float('inf'),
            'small_world_coefficient': self._compute_sw_coefficient()
        }
    
    def _compute_sw_coefficient(self):
        """Compute small-world coefficient σ"""
        random_graph = nx.erdos_renyi_graph(
            self.num_modules, 
            len(self.edge_list) / (self.num_modules * (self.num_modules - 1) / 2)
        )
        
        C = nx.average_clustering(self.graph)
        L = nx.average_shortest_path_length(self.graph) if nx.is_connected(self.graph) else float('inf')
        
        C_rand = nx.average_clustering(random_graph)
        L_rand = nx.average_shortest_path_length(random_graph) if nx.is_connected(random_graph) else float('inf')
        
        if C_rand == 0 or L_rand == 0 or L == float('inf') or L_rand == float('inf'):
            return float('nan')
        
        return (C / C_rand) / (L / L_rand)


class SegmentedWSMultiModalNetwork(nn.Module):
    """
    Complete Segmented Watts-Strogatz Multi-Modal Network.
    
    This is the CAPSTONE architecture combining all concepts.
    """
    
    def __init__(
        self,
        # Modality encoder configs
        image_encoder_config: Optional[Dict] = None,
        text_encoder_config: Optional[Dict] = None,
        audio_encoder_config: Optional[Dict] = None,
        # Module configs
        module_dim: int = 256,
        num_extra_modules: int = 2,  # Additional processing modules
        # WS connector config
        k_neighbors: int = 2,
        initial_beta: float = 0.1,
        learnable_beta: bool = True,
        dynamic_rewiring: bool = True,
        # Output config
        num_classes: int = 10,
        # Sparsity config
        internal_sparsity: float = 0.9,
        use_sparse_modules: bool = True
    ):
        super().__init__()
        
        self.module_dim = module_dim
        
        # Create modality encoders
        self.modality_encoders = nn.ModuleDict()
        
        if image_encoder_config is not None:
            if use_sparse_modules:
                self.modality_encoders['image'] = SparseImageEncoder(
                    output_dim=module_dim, sparsity=internal_sparsity
                )
            else:
                self.modality_encoders['image'] = ImageEncoder(output_dim=module_dim)
        
        if text_encoder_config is not None:
            if use_sparse_modules:
                self.modality_encoders['text'] = SparseTextEncoder(
                    vocab_size=text_encoder_config.get('vocab_size', 10000),
                    output_dim=module_dim, sparsity=internal_sparsity
                )
            else:
                self.modality_encoders['text'] = TextEncoder(
                    vocab_size=text_encoder_config.get('vocab_size', 10000),
                    output_dim=module_dim
                )
        
        if audio_encoder_config is not None:
            if use_sparse_modules:
                self.modality_encoders['audio'] = SparseAudioEncoder(
                    output_dim=module_dim, sparsity=internal_sparsity
                )
            else:
                self.modality_encoders['audio'] = AudioEncoder(output_dim=module_dim)
        
        # Additional processing modules (for fusion/reasoning)
        self.processing_modules = nn.ModuleList([
            SparseModule(module_dim, module_dim, sparsity=internal_sparsity)
            if use_sparse_modules else nn.Linear(module_dim, module_dim)
            for _ in range(num_extra_modules)
        ])
        
        # Total number of modules
        self.num_modules = len(self.modality_encoders) + num_extra_modules
        
        # WS Inter-Module Connector
        self.ws_connector = WSInterModuleConnector(
            num_modules=self.num_modules,
            module_dim=module_dim,
            k_neighbors=k_neighbors,
            initial_beta=initial_beta,
            learnable_beta=learnable_beta,
            dynamic_rewiring=dynamic_rewiring
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(module_dim * self.num_modules, module_dim),
            nn.ReLU(),
            nn.Linear(module_dim, num_classes)
        )
        
        # Module names for indexing
        self.module_names = list(self.modality_encoders.keys()) + \
                          [f'process_{i}' for i in range(num_extra_modules)]
    
    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        num_message_passes: int = 2
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through segmented WS architecture.
        
        Args:
            image: Image tensor (batch, channels, height, width)
            text: Text token ids (batch, seq_len)
            audio: Audio mel spectrogram (batch, freq_bins, time_steps)
            num_message_passes: Number of inter-module communication rounds
        
        Returns:
            output: Classification logits (batch, num_classes)
            metadata: Dict with topology metrics and intermediate activations
        """
        # Step 1: Encode each modality
        module_outputs = {}
        module_idx = 0
        
        if image is not None and 'image' in self.modality_encoders:
            module_outputs[module_idx] = self.modality_encoders['image'](image)
            module_idx += 1
        
        if text is not None and 'text' in self.modality_encoders:
            module_outputs[module_idx] = self.modality_encoders['text'](text)
            module_idx += 1
        
        if audio is not None and 'audio' in self.modality_encoders:
            module_outputs[module_idx] = self.modality_encoders['audio'](audio)
            module_idx += 1
        
        # Initialize processing modules with zeros (will receive input from connector)
        batch_size = next(iter(module_outputs.values())).shape[0]
        device = next(iter(module_outputs.values())).device
        
        for i, module in enumerate(self.processing_modules):
            module_outputs[module_idx + i] = torch.zeros(
                batch_size, self.module_dim, device=device
            )
        
        # Step 2: Inter-module communication via WS connector
        for pass_idx in range(num_message_passes):
            # Get messages from other modules
            messages = self.ws_connector(module_outputs)
            
            # Update each module's state
            new_outputs = {}
            for idx in range(self.num_modules):
                # Combine current state with incoming messages
                combined = module_outputs[idx] + messages[idx]
                
                # Apply non-linearity
                new_outputs[idx] = F.relu(combined)
            
            module_outputs = new_outputs
            
            # Dynamic topology update (if enabled)
            self.ws_connector.update_topology()
        
        # Step 3: Aggregate all module outputs
        all_outputs = torch.cat([module_outputs[i] for i in range(self.num_modules)], dim=-1)
        
        # Step 4: Output head
        output = self.output_head(all_outputs)
        
        # Metadata for visualization
        metadata = {
            'topology_metrics': self.ws_connector.get_topology_metrics(),
            'module_activations': {
                self.module_names[i]: module_outputs[i].detach()
                for i in range(self.num_modules)
            }
        }
        
        return output, metadata
    
    def get_num_parameters(self):
        """Return parameter count breakdown"""
        counts = {
            'modality_encoders': sum(
                p.numel() for p in self.modality_encoders.parameters()
            ),
            'processing_modules': sum(
                p.numel() for p in self.processing_modules.parameters()
            ),
            'ws_connector': sum(
                p.numel() for p in self.ws_connector.parameters()
            ),
            'output_head': sum(
                p.numel() for p in self.output_head.parameters()
            )
        }
        counts['total'] = sum(counts.values())
        return counts
    
    def visualize_architecture(self):
        """Create visualization of the full architecture"""
        from src.visualization import ArchitectureViz
        return ArchitectureViz(self).render()
```

---

## 4. Visualization Components

### 4.1 Required Visualizations

| Component | Technology | Priority | Module |
|-----------|------------|----------|--------|
| Single Neuron Playground | D3.js + ipywidgets | High | 01 |
| Gradient Descent Animation | Plotly 3D | High | 02 |
| Graph Builder | D3.js drag-drop | High | 03 |
| WS Rewiring Animation | D3.js + React | Critical | 04 |
| Sparse Connectivity Viewer | D3.js bipartite | High | 05 |
| Latent Space Explorer | Plotly scatter + UMAP | Medium | 06 |
| Topology Evolution Tracker | Plotly + D3.js | Critical | 07 |
| Expert Routing Visualizer | D3.js Sankey | Medium | 08 |
| Cross-Modal Attention Heatmap | Plotly heatmap | Medium | 09 |
| Full Architecture Diagram | D3.js + Three.js | Critical | 10 |

### 4.2 Visualization Implementation Guidelines

```python
# src/visualization/base.py

class InteractiveVisualization:
    """Base class for all visualizations"""
    
    def __init__(self):
        self.fig = None
        self.widgets = {}
        self._setup_callbacks()
    
    def render(self):
        """Render in Jupyter notebook"""
        raise NotImplementedError
    
    def to_html(self):
        """Export as standalone HTML"""
        raise NotImplementedError
    
    def _setup_callbacks(self):
        """Setup widget callbacks for interactivity"""
        pass
    
    def update(self, **kwargs):
        """Update visualization with new data"""
        raise NotImplementedError
```

### 4.3 Web Component Structure

```
web/
├── src/
│   ├── components/
│   │   ├── NetworkGraph.jsx          # D3 force-directed graph
│   │   ├── BipartiteGraph.jsx        # Bipartite visualization
│   │   ├── WattsStrogatzSlider.jsx   # Beta control widget
│   │   ├── MetricsPanel.jsx          # Real-time metrics display
│   │   ├── TrainingDashboard.jsx     # Training progress viz
│   │   └── ArchitectureDiagram.jsx   # Full architecture view
│   │
│   ├── visualizers/
│   │   ├── TopologyVisualizer.js     # Core topology rendering
│   │   ├── ActivationVisualizer.js   # Neural activation viz
│   │   ├── AttentionVisualizer.js    # Attention heatmaps
│   │   └── LossLandscape.js          # 3D loss surface
│   │
│   ├── sandbox/
│   │   ├── GraphBuilder.jsx          # Drag-drop graph creation
│   │   ├── ModuleDesigner.jsx        # Visual module design
│   │   └── ExperimentRunner.jsx      # Configurable experiments
│   │
│   └── api/
│       ├── jupyterBridge.js          # Communication with kernel
│       └── dataTransforms.js         # Data format conversions
```

---

## 5. Core Algorithm Implementations

See Module 07 specification for SET and DEEP R implementations.

Additional algorithms to implement:

### 5.1 BSW (Bipartite Small-World) Initialization

```python
# src/topology/bsw.py

def bipartite_small_world(n_input, n_output, k, p):
    """
    Generate Bipartite Small-World connectivity pattern.
    
    Based on Zhang et al. 2023.
    
    Args:
        n_input: Number of input nodes
        n_output: Number of output nodes
        k: Number of neighbors per output node
        p: Rewiring probability
    
    Returns:
        torch.Tensor: Adjacency matrix (n_output, n_input)
    """
    adj = torch.zeros(n_output, n_input)
    
    for i in range(n_output):
        # Map output node to input space
        center = int(i * n_input / n_output)
        
        # Connect to k nearest neighbors
        for offset in range(-k//2, k//2 + 1):
            j = (center + offset) % n_input
            
            # Rewire with probability p
            if torch.rand(1) < p:
                j = torch.randint(0, n_input, (1,)).item()
            
            adj[i, j] = 1
    
    return adj
```

### 5.2 Cannistraci-Hebb Regrowth

```python
# src/training/cannistraci_hebb.py

def cannistraci_hebb_regrow(mask, activations, n_regrow):
    """
    Gradient-free regrowth based on Cannistraci-Hebb training.
    
    Key idea: Regrow connections where co-activation is highest.
    Brain-inspired: Neurons that fire together, wire together.
    
    Args:
        mask: Current connectivity mask (n_out, n_in)
        activations: Tuple of (pre_activations, post_activations)
        n_regrow: Number of connections to add
    
    Returns:
        Updated mask
    """
    pre_act, post_act = activations
    
    # Compute co-activation scores
    # (outer product of activation magnitudes)
    co_activation = torch.abs(post_act.T @ pre_act)  # (n_out, n_in)
    
    # Zero out existing connections
    co_activation[mask.bool()] = 0
    
    # Find top-k candidates
    flat_scores = co_activation.flatten()
    _, top_indices = torch.topk(flat_scores, n_regrow)
    
    # Add new connections
    new_mask = mask.clone()
    for idx in top_indices:
        i = idx // mask.shape[1]
        j = idx % mask.shape[1]
        new_mask[i, j] = 1
    
    return new_mask
```

---

## 6. Data Pipeline

### 6.1 Dataset Configurations

```python
# src/data/datasets.py

DATASET_CONFIGS = {
    'mnist': {
        'class': 'torchvision.datasets.MNIST',
        'modalities': ['image'],
        'num_classes': 10,
        'image_size': (28, 28),
        'channels': 1,
        'transform': 'mnist_standard'
    },
    'fashion_mnist': {
        'class': 'torchvision.datasets.FashionMNIST',
        'modalities': ['image'],
        'num_classes': 10,
        'image_size': (28, 28),
        'channels': 1,
        'transform': 'mnist_standard'
    },
    'cifar10': {
        'class': 'torchvision.datasets.CIFAR10',
        'modalities': ['image'],
        'num_classes': 10,
        'image_size': (32, 32),
        'channels': 3,
        'transform': 'cifar_standard'
    },
    'audio_mnist': {
        'class': 'src.data.AudioMNIST',
        'modalities': ['audio'],
        'num_classes': 10,
        'sample_rate': 16000,
        'transform': 'audio_mel'
    },
    'flickr8k': {
        'class': 'src.data.Flickr8k',
        'modalities': ['image', 'text'],
        'num_classes': None,  # Retrieval task
        'image_size': (224, 224),
        'max_text_len': 128,
        'transform': 'multimodal_standard'
    }
}
```

### 6.2 Multi-Modal Data Loader

```python
# src/data/multimodal_loader.py

class MultiModalDataLoader:
    """
    Unified data loader for multi-modal experiments.
    
    Handles:
    - Aligned multi-modal samples
    - Missing modalities (returns None)
    - Modality-specific transforms
    """
    
    def __init__(self, dataset_name, batch_size, modalities=None):
        config = DATASET_CONFIGS[dataset_name]
        self.modalities = modalities or config['modalities']
        
        # Load dataset
        self.dataset = self._load_dataset(config)
        
        # Create loader
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._multimodal_collate
        )
    
    def _multimodal_collate(self, batch):
        """Custom collation for multi-modal data"""
        result = {}
        
        for modality in self.modalities:
            modality_data = [sample.get(modality) for sample in batch]
            
            if all(d is not None for d in modality_data):
                result[modality] = torch.stack(modality_data)
            else:
                result[modality] = None
        
        result['labels'] = torch.tensor([sample['label'] for sample in batch])
        
        return result
```

---

## 7. Testing Strategy

### 7.1 Test Categories

```
tests/
├── unit/
│   ├── test_sparse_layers.py
│   ├── test_topology_generators.py
│   ├── test_set_algorithm.py
│   ├── test_deep_r.py
│   └── test_ws_connector.py
│
├── integration/
│   ├── test_training_loop.py
│   ├── test_visualization_generation.py
│   └── test_multimodal_pipeline.py
│
├── exercises/
│   ├── test_exercise_01_solutions.py
│   ├── test_exercise_02_solutions.py
│   └── ...
│
└── performance/
    ├── test_sparse_vs_dense_speed.py
    └── test_memory_usage.py
```

### 7.2 Exercise Validation Framework

```python
# src/testing/exercise_validator.py

class ExerciseValidator:
    """
    Validate learner exercise submissions.
    
    Provides:
    - Immediate feedback
    - Hint progression
    - Solution comparison
    """
    
    def __init__(self, exercise_id):
        self.exercise_id = exercise_id
        self.tests = self._load_tests()
        self.hints = self._load_hints()
        self.attempts = 0
    
    def validate(self, submission):
        """Run tests against submission"""
        results = {
            'passed': [],
            'failed': [],
            'errors': []
        }
        
        for test in self.tests:
            try:
                test.run(submission)
                results['passed'].append(test.name)
            except AssertionError as e:
                results['failed'].append({
                    'test': test.name,
                    'message': str(e)
                })
            except Exception as e:
                results['errors'].append({
                    'test': test.name,
                    'error': str(e)
                })
        
        self.attempts += 1
        
        return results
    
    def get_hint(self, level=None):
        """Get progressive hint"""
        if level is None:
            level = min(self.attempts, len(self.hints) - 1)
        return self.hints[level]
    
    def show_solution(self):
        """Reveal reference solution"""
        return self._load_solution()
```

---

## 8. File Structure

Complete file structure for the project:

```
multi-modal-WS-foundational-training/
│
├── README.md                           # Project overview and quick start
├── LICENSE                             # MIT License
├── requirements.txt                    # Python dependencies
├── pyproject.toml                      # Project configuration
├── setup.py                            # Package installation
│
├── docs/
│   ├── PRD.md                          # Product Requirements Document
│   ├── SPECIFICATION.md                # This document
│   ├── CONTRIBUTING.md                 # Contribution guidelines
│   ├── CHANGELOG.md                    # Version history
│   └── api/                            # API documentation
│
├── configs/
│   ├── default.yaml                    # Default configuration
│   ├── experiments/                    # Experiment configs
│   └── visualization/                  # Viz configs
│
├── notebooks/
│   ├── 00_setup/
│   │   ├── 00_welcome.ipynb
│   │   ├── 01_environment_check.ipynb
│   │   └── 02_learning_path.ipynb
│   │
│   ├── 01_foundations/
│   │   ├── 01_what_is_nn.ipynb
│   │   ├── 02_forward_pass.ipynb
│   │   ├── 03_activation_functions.ipynb
│   │   ├── 04_from_scratch.ipynb
│   │   └── quiz_01.ipynb
│   │
│   ├── 02_supervised/
│   │   ├── 01_loss_functions.ipynb
│   │   ├── 02_gradient_descent.ipynb
│   │   ├── 03_backprop_intuition.ipynb
│   │   ├── 04_pytorch_intro.ipynb
│   │   ├── 05_training_loop.ipynb
│   │   ├── 06_overfitting.ipynb
│   │   ├── lab_mnist.ipynb
│   │   └── quiz_02.ipynb
│   │
│   ├── 03_graphs/
│   │   ├── 01_what_is_graph.ipynb
│   │   ├── 02_representations.ipynb
│   │   ├── 03_properties.ipynb
│   │   ├── 04_networkx_intro.ipynb
│   │   ├── 05_neural_nets_as_graphs.ipynb
│   │   └── quiz_03.ipynb
│   │
│   ├── 04_topology/
│   │   ├── 01_random_graphs.ipynb
│   │   ├── 02_watts_strogatz.ipynb
│   │   ├── 03_scale_free.ipynb
│   │   ├── 04_comparing_topologies.ipynb
│   │   ├── 05_small_world_properties.ipynb
│   │   ├── lab_topology_explorer.ipynb
│   │   └── quiz_04.ipynb
│   │
│   ├── 05_sparse/
│   │   ├── 01_why_sparse.ipynb
│   │   ├── 02_pruning_basics.ipynb
│   │   ├── 03_sparse_representations.ipynb
│   │   ├── 04_pytorch_sparse.ipynb
│   │   ├── 05_sparse_vs_dense.ipynb
│   │   ├── lab_pruning.ipynb
│   │   └── quiz_05.ipynb
│   │
│   ├── 06_unsupervised/
│   │   ├── 01_why_unsupervised.ipynb
│   │   ├── 02_autoencoders.ipynb
│   │   ├── 03_latent_spaces.ipynb
│   │   ├── 04_contrastive_learning.ipynb
│   │   ├── 05_representation_quality.ipynb
│   │   ├── lab_autoencoder.ipynb
│   │   └── quiz_06.ipynb
│   │
│   ├── 07_dynamic_sparse/
│   │   ├── 01_static_vs_dynamic.ipynb
│   │   ├── 02_set_algorithm.ipynb
│   │   ├── 03_set_implementation.ipynb
│   │   ├── 04_deep_r.ipynb
│   │   ├── 05_deep_r_implementation.ipynb
│   │   ├── 06_bsw_init.ipynb
│   │   ├── 07_topology_evolution.ipynb
│   │   ├── lab_dynamic_training.ipynb
│   │   └── quiz_07.ipynb
│   │
│   ├── 08_modular/
│   │   ├── 01_why_modular.ipynb
│   │   ├── 02_mixture_of_experts.ipynb
│   │   ├── 03_gating_mechanisms.ipynb
│   │   ├── 04_sparse_moe.ipynb
│   │   ├── 05_module_specialization.ipynb
│   │   ├── 06_inter_module_communication.ipynb
│   │   ├── lab_moe.ipynb
│   │   └── quiz_08.ipynb
│   │
│   ├── 09_multimodal/
│   │   ├── 01_what_is_multimodal.ipynb
│   │   ├── 02_modality_encoders.ipynb
│   │   ├── 03_fusion_strategies.ipynb
│   │   ├── 04_cross_modal_attention.ipynb
│   │   ├── 05_alignment_binding.ipynb
│   │   ├── 06_multimodal_sparse.ipynb
│   │   ├── lab_vision_language.ipynb
│   │   └── quiz_09.ipynb
│   │
│   ├── 10_capstone/
│   │   ├── 01_architecture_design.ipynb
│   │   ├── 02_module_implementation.ipynb
│   │   ├── 03_ws_connector.ipynb
│   │   ├── 04_learnable_rewiring.ipynb
│   │   ├── 05_training_integration.ipynb
│   │   ├── 06_evaluation.ipynb
│   │   ├── 07_ablation_study.ipynb
│   │   └── 08_next_steps.ipynb
│   │
│   └── exercises/
│       ├── solutions/
│       └── templates/
│
├── src/
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── device.py
│   │   └── logging.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── sparse_layers.py
│   │   ├── sparse_encoders.py
│   │   ├── mixture_of_experts.py
│   │   ├── multimodal.py
│   │   └── segmented_ws_architecture.py
│   │
│   ├── topology/
│   │   ├── __init__.py
│   │   ├── generators.py
│   │   ├── bsw.py
│   │   ├── metrics.py
│   │   └── network_to_graph.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── set_algorithm.py
│   │   ├── deep_r.py
│   │   ├── cannistraci_hebb.py
│   │   └── callbacks.py
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── graph_basics.py
│   │   ├── watts_strogatz.py
│   │   ├── topology_evolution.py
│   │   ├── latent_explorer.py
│   │   ├── training_dashboard.py
│   │   └── architecture_viz.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   ├── transforms.py
│   │   └── multimodal_loader.py
│   │
│   └── testing/
│       ├── __init__.py
│       └── exercise_validator.py
│
├── web/
│   ├── package.json
│   ├── webpack.config.js
│   │
│   ├── src/
│   │   ├── index.js
│   │   ├── App.jsx
│   │   │
│   │   ├── components/
│   │   │   ├── NetworkGraph.jsx
│   │   │   ├── BipartiteGraph.jsx
│   │   │   ├── WattsStrogatzSlider.jsx
│   │   │   ├── MetricsPanel.jsx
│   │   │   ├── TrainingDashboard.jsx
│   │   │   └── ArchitectureDiagram.jsx
│   │   │
│   │   ├── visualizers/
│   │   │   ├── TopologyVisualizer.js
│   │   │   ├── ActivationVisualizer.js
│   │   │   ├── AttentionVisualizer.js
│   │   │   └── LossLandscape.js
│   │   │
│   │   ├── sandbox/
│   │   │   ├── GraphBuilder.jsx
│   │   │   ├── ModuleDesigner.jsx
│   │   │   └── ExperimentRunner.jsx
│   │   │
│   │   └── api/
│   │       ├── jupyterBridge.js
│   │       └── dataTransforms.js
│   │
│   └── public/
│       └── index.html
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   │
│   ├── unit/
│   │   ├── test_sparse_layers.py
│   │   ├── test_topology_generators.py
│   │   ├── test_set_algorithm.py
│   │   ├── test_deep_r.py
│   │   └── test_ws_connector.py
│   │
│   ├── integration/
│   │   ├── test_training_loop.py
│   │   ├── test_visualization_generation.py
│   │   └── test_multimodal_pipeline.py
│   │
│   ├── exercises/
│   │   └── test_exercise_solutions.py
│   │
│   └── performance/
│       ├── test_sparse_vs_dense_speed.py
│       └── test_memory_usage.py
│
├── scripts/
│   ├── setup_environment.py
│   ├── download_datasets.py
│   ├── run_all_tests.py
│   └── build_docs.py
│
└── data/                               # .gitignored, downloaded at runtime
    ├── mnist/
    ├── cifar10/
    └── ...
```

---

## 9. Implementation Order

### Phase 1: Foundation (Weeks 1-2)

**Week 1:**
1. Project scaffolding (file structure, configs)
2. `src/core/` utilities
3. `notebooks/00_setup/` all notebooks
4. `notebooks/01_foundations/` all notebooks
5. Basic test infrastructure

**Week 2:**
1. `notebooks/02_supervised/` all notebooks
2. `src/visualization/base.py` and training dashboard
3. Exercise validation framework
4. MNIST lab completion

### Phase 2: Graph & Topology (Weeks 3-4)

**Week 3:**
1. `notebooks/03_graphs/` all notebooks
2. `src/visualization/graph_basics.py`
3. NetworkX integration
4. Graph builder sandbox

**Week 4:**
1. `notebooks/04_topology/` all notebooks
2. `src/topology/` all modules
3. **CRITICAL: Watts-Strogatz visualizer**
4. Topology explorer lab

### Phase 3: Sparse Networks (Weeks 5-6)

**Week 5:**
1. `src/models/sparse_layers.py`
2. `notebooks/05_sparse/` all notebooks
3. Pruning lab
4. Sparse vs dense comparison visualization

**Week 6:**
1. `notebooks/06_unsupervised/` all notebooks
2. `src/visualization/latent_explorer.py`
3. Autoencoder lab
4. Representation quality experiments

### Phase 4: Dynamic Training (Weeks 7-8)

**Week 7:**
1. `src/training/set_algorithm.py`
2. `src/training/deep_r.py`
3. `notebooks/07_dynamic_sparse/01-04` (SET)

**Week 8:**
1. `notebooks/07_dynamic_sparse/05-07` (DEEP R, BSW)
2. **CRITICAL: Topology evolution visualizer**
3. Dynamic training lab
4. BSW initialization

### Phase 5: Modular & Multi-Modal (Weeks 9-10)

**Week 9:**
1. `src/models/mixture_of_experts.py`
2. `notebooks/08_modular/` all notebooks
3. Expert routing visualizer
4. MoE lab

**Week 10:**
1. `src/models/multimodal.py`
2. `src/data/multimodal_loader.py`
3. `notebooks/09_multimodal/` all notebooks
4. Vision-language lab

### Phase 6: Capstone (Weeks 11-12)

**Week 11:**
1. `src/models/segmented_ws_architecture.py`
2. `notebooks/10_capstone/01-04`
3. WS Connector implementation
4. Learnable rewiring

**Week 12:**
1. `notebooks/10_capstone/05-08`
2. Full architecture visualization
3. Ablation study framework
4. Final documentation and polish

---

## 10. Research Content Guidelines

### 10.1 Paper Integration Strategy

For each referenced paper, provide:

1. **One-paragraph summary** (intuition, no jargon)
2. **Key figure reproduction** (recreate main visualization)
3. **Simplified implementation** (core algorithm only)
4. **Connection to architecture** (how it fits our design)

### 10.2 Content Depth Levels

| Level | Audience | Content |
|-------|----------|---------|
| Intuitive | Everyone | Analogies, visualizations, no math |
| Practical | Implementers | Code, experiments, minimal math |
| Rigorous | Researchers | Full equations, proofs (collapsible) |

### 10.3 Key Papers by Module

**Module 04 (Topology):**
- Watts & Strogatz 1998 → Full algorithm implementation
- Newman 2003 → Conceptual overview of network science

**Module 07 (Dynamic Sparse):**
- Mocanu et al. 2018 (SET) → Full implementation + experiments
- Bellec et al. 2017 (DEEP R) → Full implementation
- Zhang et al. 2023 (BSW/CHT) → Implementation + comparison

**Module 08 (Modular):**
- Shazeer et al. 2017 (Sparse MoE) → Simplified implementation
- Chen et al. 2024 (MacNet) → Conceptual + small-world connection

**Module 10 (Capstone):**
- Synthesize all above into novel architecture
- Reference DNM preprint for biological inspiration

---

## 11. Capstone Architecture Specification

### 11.1 Design Principles

1. **Modularity**: Each modality has dedicated processing module
2. **Sparse Connectivity**: Both within modules and between modules
3. **Small-World Topology**: Inter-module connections follow WS model
4. **Learnable Structure**: β parameter and connection weights are learned
5. **Dynamic Evolution**: Topology can rewire during training

### 11.2 Hyperparameter Recommendations

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| module_dim | 256 | 64-512 | Balance capacity vs compute |
| k_neighbors | 2 | 2-4 | Higher = more initial connections |
| initial_beta | 0.1 | 0.05-0.3 | Start with some randomness |
| internal_sparsity | 0.9 | 0.8-0.95 | Higher = more efficient |
| num_message_passes | 2 | 1-3 | More = better fusion, more compute |

### 11.3 Expected Results

Based on research synthesis, the capstone architecture should achieve:

- **Parameter efficiency**: 5-10x fewer parameters than dense equivalent
- **Training speed**: Comparable or faster (sparse ops)
- **Accuracy**: Within 1-2% of dense baseline
- **Interpretability**: Clear module specialization visible

### 11.4 Ablation Studies to Include

1. **β ablation**: Compare β=0 (lattice), β=0.1 (SW), β=1 (random)
2. **Sparsity ablation**: 70%, 80%, 90%, 95% internal sparsity
3. **Dynamic vs static**: With and without topology evolution
4. **Learnable β vs fixed**: Does learning β help?
5. **Module count**: 3, 5, 7 modules

---

## 12. API Reference

### 12.1 Core Classes

```python
# Quick reference for main classes

# Topology
from src.topology import WattsStrogatzGenerator, BipartiteSmallWorld
from src.topology import TopologyMetrics, NetworkToGraph

# Models
from src.models import SparseLinear, SparseConv2d
from src.models import SparseMixtureOfExperts
from src.models import WSInterModuleConnector
from src.models import SegmentedWSMultiModalNetwork

# Training
from src.training import SETTrainer, DeepRewiring
from src.training import MultiModalTrainer

# Visualization
from src.visualization import WattsStrogatzVisualizer
from src.visualization import TopologyEvolutionViz
from src.visualization import LatentSpaceExplorer
from src.visualization import ArchitectureViz

# Data
from src.data import MultiModalDataLoader
from src.data import get_dataset
```

### 12.2 Key Functions

```python
# Topology generation
ws_graph = generate_watts_strogatz(n=100, k=4, p=0.1)
bsw_mask = bipartite_small_world(n_in=784, n_out=256, k=4, p=0.1)

# Metrics
metrics = compute_topology_metrics(graph)  # clustering, path_length, sw_coef

# Training
trainer = SETTrainer(model, sparsity=0.9, prune_rate=0.3)
trainer.train(dataloader, epochs=100)

# Visualization
viz = WattsStrogatzVisualizer(n=100, k=4, p=0.1)
viz.animate_rewiring(target_p=0.5)
```

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| Watts-Strogatz (WS) | Small-world network model with tunable clustering and path length |
| Small-world coefficient (σ) | Ratio of normalized clustering to normalized path length; σ>1 indicates small-world |
| SET | Sparse Evolutionary Training - dynamic sparse training algorithm |
| DEEP R | Deep Rewiring - probabilistic sparse training algorithm |
| BSW | Bipartite Small-World - WS model adapted for neural network layers |
| MoE | Mixture of Experts - modular architecture with gated routing |
| β (beta) | Rewiring probability in WS model; controls randomness |
| k | Number of nearest neighbors in initial WS ring lattice |

---

## Appendix B: References

1. Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks. Nature, 393(6684), 440-442.

2. Mocanu, D. C., et al. (2018). Scalable training of artificial neural networks with adaptive sparse connectivity inspired by network science. Nature Communications, 9(1), 2383.

3. Bellec, G., et al. (2017). Deep rewiring: Training very sparse deep networks. arXiv:1711.05136.

4. Zhang, T., et al. (2023). Cannistraci-Hebb training for sparse neural networks. [Conference paper]

5. Chen, C., et al. (2024). Scaling Large-Language-Model-based Multi-Agent Collaboration. arXiv:2406.07155.

6. Papillon, M., et al. (2023). Architectures of Topological Deep Learning: A Survey. arXiv:2304.10031.

---

*End of Specification Document*

*This document should be used by Claude Code to implement the Multi-Modal Watts-Strogatz Foundational Training Program as specified in the companion PRD.md.*
