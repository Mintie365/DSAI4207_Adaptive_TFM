"""Fail-fast environment validation for the fixed HWU64 experiment."""
from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path

# Make `python scripts/check_environment.py` work from any current directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MIN_PYTHON = (3, 10)
RECOMMENDED_PYTHON = (3, 12)

print(f"Operating system: {platform.platform()}")
print(f"Python: {platform.python_version()}")
if sys.version_info < MIN_PYTHON:
    raise SystemExit(
        "ERROR: TabPFN 9.2.0 requires Python 3.10+. Your current environment is "
        f"Python {platform.python_version()}. Create a fresh Python 3.12 virtual environment."
    )
if sys.version_info[:2] != RECOMMENDED_PYTHON:
    print("WARNING: Python 3.12 is the recommended project environment.")

packages = (
    "torch", "numpy", "pandas", "scikit-learn", "scipy", "datasets",
    "sentence-transformers", "transformers", "tabpfn", "huggingface-hub",
    "matplotlib", "reportlab", "PyYAML", "joblib",
)
missing = []
for package in packages:
    try:
        print(f"{package}: {importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        missing.append(package)
        print(f"{package}: NOT INSTALLED")
if missing:
    raise SystemExit("Install missing packages with: python -m pip install -r requirements.txt")

import torch
import reportlab
print(f"ReportLab import: {reportlab.Version}")
cuda_available = torch.cuda.is_available()
mps_available = torch.backends.mps.is_available()
print(f"CUDA available: {cuda_available}")
print(f"MPS available: {mps_available}")
if not cuda_available and not mps_available:
    raise SystemExit(
        "ERROR: this fixed TabPFN-v2 experiment has 1,791 meta-training rows and "
        "requires a supported accelerator (Apple MPS or NVIDIA CUDA). Do not start "
        "the full experiment on CPU; use an M-series Mac with MPS or a CUDA machine."
    )

from src.pipeline import read_config
cfg = read_config()
assert cfg["dataset"]["expected_classes"] == 64
assert cfg["selection"]["k"] == 8
assert cfg["retrieval"]["candidate_pool_size"] == 32
assert cfg["meta_selector"]["tabpfn_version"] == "v2"
assert cfg["meta_selector"]["tie_tolerance"] == 1e-6
print("Fixed HWU64 protocol configuration: OK")
