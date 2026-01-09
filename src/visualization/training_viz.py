"""
Training visualization utilities.

Provides real-time and post-training visualization of metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any


def plot_loss_curves(
    history: Dict[str, List[float]],
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 5),
    title: str = 'Training Progress',
) -> plt.Figure:
    """
    Plot training and validation loss curves.

    Args:
        history: Dictionary with 'train_loss', 'val_loss' keys
        ax: Matplotlib axes
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    epochs = range(1, len(history.get('train_loss', [])) + 1)

    if 'train_loss' in history:
        ax.plot(epochs, history['train_loss'], label='Train Loss', color='#1f77b4')

    if 'val_loss' in history and history['val_loss']:
        ax.plot(epochs, history['val_loss'], label='Val Loss', color='#ff7f0e')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_metrics(
    history: Dict[str, List[float]],
    metrics: List[str] = ['accuracy'],
    figsize: Tuple[int, int] = (12, 4),
) -> plt.Figure:
    """
    Plot multiple training metrics.

    Args:
        history: Training history dictionary
        metrics: List of metrics to plot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'

        epochs = range(1, len(history.get(train_key, history.get(metric, []))) + 1)

        if train_key in history:
            ax.plot(epochs, history[train_key], label=f'Train {metric}')
        elif metric in history:
            ax.plot(epochs, history[metric], label=metric)

        if val_key in history and history[val_key]:
            ax.plot(epochs, history[val_key], label=f'Val {metric}')

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_sparsity_evolution(
    sparsity_history: List[Dict[str, float]],
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Plot how sparsity changes during training.

    Args:
        sparsity_history: List of sparsity dictionaries from SparseTrainer
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if not sparsity_history:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No sparsity data available',
                ha='center', va='center', transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    # Extract data
    steps = [d.get('step', i) for i, d in enumerate(sparsity_history)]

    # Plot each layer's sparsity
    layer_keys = [k for k in sparsity_history[0].keys() if k.startswith('layer_')]

    for key in layer_keys:
        values = [d.get(key, 0) for d in sparsity_history]
        ax.plot(steps, values, label=key, alpha=0.7)

    # Plot overall sparsity if available
    if 'overall_sparsity' in sparsity_history[0]:
        overall = [d.get('overall_sparsity', 0) for d in sparsity_history]
        ax.plot(steps, overall, label='Overall', linewidth=2, color='black')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Sparsity')
    ax.set_title('Sparsity Evolution During Training')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_learning_rate(
    lr_history: List[float],
    figsize: Tuple[int, int] = (10, 4),
) -> plt.Figure:
    """
    Plot learning rate schedule.

    Args:
        lr_history: List of learning rates per epoch/step
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(lr_history, color='#1f77b4')
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    return fig


def create_training_dashboard(
    history: Dict[str, List[float]],
    sparsity_history: Optional[List[Dict]] = None,
    figsize: Tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Create a comprehensive training dashboard.

    Args:
        history: Training history
        sparsity_history: Optional sparsity history
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_rows = 2 if sparsity_history else 1
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)

    if n_rows == 1:
        axes = [axes]

    # Loss curves
    ax_loss = axes[0][0]
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    if 'train_loss' in history:
        ax_loss.plot(epochs, history['train_loss'], label='Train')
    if 'val_loss' in history and history['val_loss']:
        ax_loss.plot(epochs, history['val_loss'], label='Val')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Loss')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # Accuracy curves
    ax_acc = axes[0][1]
    if 'train_acc' in history:
        ax_acc.plot(epochs, history['train_acc'], label='Train')
    if 'val_acc' in history and history['val_acc']:
        ax_acc.plot(epochs, history['val_acc'], label='Val')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy (%)')
    ax_acc.set_title('Accuracy')
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    # Sparsity (if available)
    if sparsity_history and n_rows > 1:
        ax_sparse = axes[1][0]
        steps = [d.get('step', i) for i, d in enumerate(sparsity_history)]
        if 'overall_sparsity' in sparsity_history[0]:
            overall = [d.get('overall_sparsity', 0) for d in sparsity_history]
            ax_sparse.plot(steps, overall)
        ax_sparse.set_xlabel('Step')
        ax_sparse.set_ylabel('Sparsity')
        ax_sparse.set_title('Network Sparsity')
        ax_sparse.grid(True, alpha=0.3)

        # Compression ratio
        ax_comp = axes[1][1]
        if 'compression_ratio' in sparsity_history[0]:
            ratio = [d.get('compression_ratio', 1) for d in sparsity_history]
            ax_comp.plot(steps, ratio)
        ax_comp.set_xlabel('Step')
        ax_comp.set_ylabel('Compression Ratio')
        ax_comp.set_title('Parameter Compression')
        ax_comp.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 8),
    normalize: bool = True,
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for each class
        figsize: Figure size
        normalize: Normalize by row (true class)

    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    # Add text annotations
    thresh = cm.max() / 2
    for i in range(n_classes):
        for j in range(n_classes):
            value = f'{cm[i, j]:.2f}' if normalize else f'{cm[i, j]}'
            ax.text(j, i, value, ha='center', va='center',
                   color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    return fig
