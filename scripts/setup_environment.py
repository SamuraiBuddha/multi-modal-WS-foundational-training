#!/usr/bin/env python3
"""
Environment setup script for Multi-Modal WS Foundational Training.

Usage:
    python scripts/setup_environment.py
"""

import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Check Python version meets requirements."""
    print("[INFO] Checking Python version...")
    version = sys.version_info
    if version < (3, 11):
        print(f"[FAIL] Python 3.11+ required, found {version.major}.{version.minor}")
        return False
    print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_requirements():
    """Install required packages."""
    print("\n[INFO] Installing requirements...")

    requirements_path = Path(__file__).parent.parent / "requirements.txt"

    if not requirements_path.exists():
        print(f"[FAIL] requirements.txt not found at {requirements_path}")
        return False

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_path)
        ])
        print("[OK] Requirements installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Failed to install requirements: {e}")
        return False


def check_pytorch():
    """Check PyTorch installation and GPU availability."""
    print("\n[INFO] Checking PyTorch...")

    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("[OK] MPS (Apple Silicon) available")
        else:
            print("[INFO] No GPU detected, will use CPU")

        return True
    except ImportError:
        print("[FAIL] PyTorch not found")
        return False


def check_jupyter():
    """Check JupyterLab installation."""
    print("\n[INFO] Checking JupyterLab...")

    try:
        import jupyterlab
        print(f"[OK] JupyterLab {jupyterlab.__version__}")
        return True
    except ImportError:
        print("[WARN] JupyterLab not found")
        return False


def check_custom_modules():
    """Check custom modules can be imported."""
    print("\n[INFO] Checking custom modules...")

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    modules = [
        ("src.topology", "Topology module"),
        ("src.models", "Models module"),
        ("src.training", "Training module"),
        ("src.data", "Data module"),
        ("src.visualization", "Visualization module"),
    ]

    all_ok = True
    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"[OK] {description}")
        except ImportError as e:
            print(f"[FAIL] {description}: {e}")
            all_ok = False

    return all_ok


def create_data_directory():
    """Create data directory if it doesn't exist."""
    print("\n[INFO] Creating data directory...")

    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    print(f"[OK] Data directory: {data_dir}")


def main():
    """Run environment setup."""
    print("=" * 60)
    print("Multi-Modal WS Foundational Training - Environment Setup")
    print("=" * 60)

    success = True

    # Check Python version
    if not check_python_version():
        success = False

    # Install requirements
    if not install_requirements():
        success = False

    # Check PyTorch
    if not check_pytorch():
        success = False

    # Check JupyterLab
    check_jupyter()  # Warning only, not critical

    # Check custom modules
    if not check_custom_modules():
        success = False

    # Create data directory
    create_data_directory()

    # Summary
    print("\n" + "=" * 60)
    if success:
        print("[OK] Environment setup complete!")
        print("\nNext steps:")
        print("  1. Run: jupyter lab")
        print("  2. Open: notebooks/00_setup/00_welcome.ipynb")
    else:
        print("[FAIL] Environment setup encountered errors")
        print("Please fix the issues above and run this script again.")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
