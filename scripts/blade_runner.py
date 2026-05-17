#!/usr/bin/env python3
"""BLADE Inference Runner - wraps BLADE API with proper setup"""
import sys, os

# Add BLADE to path
BLADE_ROOT = "/mnt/d/HERMES/study/lesserafim-thigh-paper/blade"
sys.path.insert(0, BLADE_ROOT)

# Setup CUDA paths
os.environ["CUDA_HOME"] = os.path.expanduser("~/miniconda3/envs/blade_env")
os.environ["CPATH"] = os.path.expanduser("~/miniconda3/envs/blade_env/targets/x86_64-linux/include")

# Patch mmcv compatibility
import mmcv
import mmcv.runner
from mmcv.utils import print_log
mmcv.runner.print_log = print_log

from blade.configs.base import root
from blade.configs.blade_inthewild import model as model_cfg

from blade.models.architectures.builder import build_architecture
import torch

def setup_model(device='cuda'):
    """Build the BLADE model and load checkpoint."""
    model = build_architecture(model_cfg)
    
    # Load checkpoint
    ckpt_path = os.path.join(BLADE_ROOT, 'pretrained', 'epoch_2.pth')
    if os.path.exists(ckpt_path):
        from mmcv.runner import load_checkpoint
        checkpoint = load_checkpoint(model, ckpt_path, map_location=device)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"WARNING: Checkpoint not found at {ckpt_path}")
    
    model = model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    print("Building BLADE model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model = setup_model(device)
    print(f"Model: {type(model).__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n✓ BLADE model loaded successfully!")
