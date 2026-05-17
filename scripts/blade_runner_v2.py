#!/usr/bin/env python3
"""BLADE Inference Runner v2 - with proper mmcv compatibility"""
import sys, os, importlib

BLADE_ROOT = "/mnt/d/HERMES/study/lesserafim-thigh-paper/blade"
sys.path.insert(0, BLADE_ROOT)

os.environ["CUDA_HOME"] = os.path.expanduser("~/miniconda3/envs/blade_env")

# Force load mmcv from BLADE's local version first
mmcv_path = os.path.join(BLADE_ROOT, "mmcv")
sys.path.insert(0, mmcv_path)

# Now import and patch mmcv before anything else
spec = importlib.util.spec_from_file_location(
    "mmcv", 
    os.path.join(mmcv_path, "mmcv", "__init__.py"),
    submodule_search_locations=[os.path.join(mmcv_path, "mmcv")]
)
mmcv = importlib.util.module_from_spec(spec)

# Set attributes BEFORE the module is fully loaded
from mmcv.version import __version__ as mmcv_ver
mmcv.__version__ = mmcv_ver

# Now execute the module
spec.loader.exec_module(mmcv)

# Add print_log at top level
from mmcv.utils import print_log
mmcv.print_log = print_log

# Replace in sys.modules so all imports see our patched version
sys.modules['mmcv'] = mmcv

print(f"mmcv loaded from: {mmcv.__file__}")
print(f"mmcv version: {mmcv.__version__}")
print(f"print_log available: {hasattr(mmcv, 'print_log')}")

# Now try the BLADE imports
print("\nLoading BLADE...")
from blade.configs.base import root
print(f"Root: {root}")

from blade.configs.blade_inthewild import model as model_cfg
print("Config loaded")

from blade.models.architectures.builder import build_architecture
print("Architecture builder loaded")

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

model = build_architecture(model_cfg)

# Load checkpoint
ckpt_path = os.path.join(BLADE_ROOT, 'pretrained', 'epoch_2.pth')
if os.path.exists(ckpt_path):
    from mmcv.runner import load_checkpoint
    checkpoint = load_checkpoint(model, ckpt_path, map_location=device)
    print(f"Checkpoint loaded: {os.path.getsize(ckpt_path)/1024/1024:.0f} MB")
else:
    print(f"WARNING: Checkpoint not found at {ckpt_path}")

model = model.to(device)
model.eval()
print(f"\n✓ Model: {type(model).__name__}")
print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
print("\nBLADE is ready for inference!")
