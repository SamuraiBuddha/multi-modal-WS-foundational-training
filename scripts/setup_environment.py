#!/usr/bin/env python3
"""
Environment setup and verification script.

Checks that all required dependencies are installed and working correctly.
Run this before starting the notebooks to ensure everything is configured.

Usage:
    python scripts/setup_environment.py
"""

import sys
import subprocess
from typing import List, Tuple, Optional


def check_python_version() -> Tuple[bool, str]:
    """Check Python version is 3.10+."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor} (need 3.10+)"


def check_import(module: str, package: Optional[str] = None) -> Tuple[bool, str]:
    """Check if a module can be imported."""
    try:
        __import__(module)
        return True, package or module
    except ImportError as e:
        return False, f"{package or module}: {str(e)}"


def check_torch_cuda() -> Tuple[bool, str]:
    """Check PyTorch and CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return True, f"PyTorch {torch.__version__} with CUDA ({device_name})"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return True, f"PyTorch {torch.__version__} with MPS (Apple Silicon)"
        else:
            return True, f"PyTorch {torch.__version__} (CPU only)"
    except ImportError:
        return False, "PyTorch not installed"


def check_jupyter() -> Tuple[bool, str]:
    """Check JupyterLab installation."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "jupyter", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return True, "JupyterLab available"
        return False, "JupyterLab not configured"
    except Exception as e:
        return False, f"JupyterLab check failed: {e}"


def check_node_npm() -> Tuple[bool, str]:
    """Check Node.js and npm for web components."""
    try:
        node_result = subprocess.run(
            ["node", "--version"],
            capture_output=True, text=True, timeout=10
        )
        npm_result = subprocess.run(
            ["npm", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if node_result.returncode == 0 and npm_result.returncode == 0:
            node_ver = node_result.stdout.strip()
            npm_ver = npm_result.stdout.strip()
            return True, f"Node.js {node_ver}, npm {npm_ver}"
        return False, "Node.js or npm not found"
    except FileNotFoundError:
        return False, "Node.js not installed (optional for web components)"
    except Exception as e:
        return False, f"Node.js check failed: {e}"


def run_checks() -> List[Tuple[str, bool, str]]:
    """Run all environment checks."""
    checks = []

    # Python version
    ok, msg = check_python_version()
    checks.append(("Python Version", ok, msg))

    # Core ML libraries
    core_libs = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
    ]
    for module, name in core_libs:
        ok, msg = check_import(module, name)
        checks.append((name, ok, msg))

    # Optional audio library
    ok, msg = check_import("torchaudio", "TorchAudio")
    checks.append(("TorchAudio (optional)", ok, msg))

    # PyTorch CUDA/GPU
    ok, msg = check_torch_cuda()
    checks.append(("GPU Support", ok, msg))

    # Graph libraries
    graph_libs = [
        ("networkx", "NetworkX"),
        ("scipy", "SciPy"),
    ]
    for module, name in graph_libs:
        ok, msg = check_import(module, name)
        checks.append((name, ok, msg))

    # Visualization
    viz_libs = [
        ("matplotlib", "Matplotlib"),
        ("plotly", "Plotly"),
        ("ipywidgets", "ipywidgets"),
    ]
    for module, name in viz_libs:
        ok, msg = check_import(module, name)
        checks.append((name, ok, msg))

    # Utilities
    util_libs = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("tqdm", "tqdm"),
        ("sklearn", "scikit-learn"),
    ]
    for module, name in util_libs:
        ok, msg = check_import(module, name)
        checks.append((name, ok, msg))

    # Jupyter
    ok, msg = check_jupyter()
    checks.append(("JupyterLab", ok, msg))

    # Node.js (optional)
    ok, msg = check_node_npm()
    checks.append(("Node.js (optional)", ok, msg))

    # Check our own package
    ok, msg = check_import("src", "Local src package")
    checks.append(("Local Package", ok, msg))

    return checks


def print_results(checks: List[Tuple[str, bool, str]]) -> bool:
    """Print check results and return overall status."""
    print("\n" + "=" * 60)
    print("  Multi-Modal WS Training - Environment Check")
    print("=" * 60 + "\n")

    all_ok = True
    required_failed = []
    optional_failed = []

    for name, ok, msg in checks:
        status = "[OK]" if ok else "[X]"
        print(f"  {status:6} {name:20} {msg}")

        if not ok:
            if "optional" in name.lower():
                optional_failed.append(name)
            else:
                required_failed.append(name)
                all_ok = False

    print("\n" + "-" * 60)

    if all_ok:
        print("\n  [OK] All required dependencies are installed!")
        print("  You're ready to start the notebooks.\n")
        print("  Next steps:")
        print("    1. cd notebooks/00_setup")
        print("    2. jupyter lab")
        print("    3. Open 00_welcome.ipynb\n")
    else:
        print(f"\n  [!] Missing {len(required_failed)} required dependencies:")
        for name in required_failed:
            print(f"      - {name}")
        print("\n  Install missing packages with:")
        print("    pip install -r requirements.txt\n")

    if optional_failed:
        print(f"  [INFO] {len(optional_failed)} optional dependencies not found.")
        print("         Web components may have limited functionality.\n")

    return all_ok


def create_data_dirs():
    """Create necessary data directories."""
    import os
    from pathlib import Path

    base_dir = Path(__file__).parent.parent
    dirs = [
        base_dir / "data",
        base_dir / "data" / "raw",
        base_dir / "data" / "processed",
        base_dir / "checkpoints",
        base_dir / "logs",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    print("  [OK] Created data directories")


def main():
    """Main entry point."""
    checks = run_checks()
    all_ok = print_results(checks)

    if all_ok:
        create_data_dirs()

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
