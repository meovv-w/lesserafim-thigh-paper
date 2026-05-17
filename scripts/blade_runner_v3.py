#!/usr/bin/env python3
"""BLADE Inference Runner v3 - with import monkey patches"""
import sys, os, types

BLADE_ROOT = "/mnt/d/HERMES/study/lesserafim-thigh-paper/blade"
sys.path.insert(0, BLADE_ROOT)
# Setup CUDA paths
CUDA_HOME = os.path.expanduser("~/cuda-11.8")
os.environ["CUDA_HOME"] = CUDA_HOME
os.environ["PATH"] = f"{CUDA_HOME}/bin:" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{CUDA_HOME}/lib64:" + os.path.expanduser("~/miniconda3/envs/blade_env/lib/python3.9/site-packages/torch/lib:~/miniconda3/envs/blade_env/lib")

# =============================================================
# Monkey-patch mmpose before anything else tries to import it
# BLADE uses mmpose 0.x API but pip installed mmpose is 1.x
# =============================================================
# Create a minimal mock for the specific function BLADE needs
class MockMmposeUtils:
    @staticmethod
    def adapt_mmdet_pipeline(cfg):
        return cfg
    
mock_mmpose = types.ModuleType('mmpose')
mock_mmpose.utils = MockMmposeUtils()
mock_mmpose.apis = types.ModuleType('mmpose.apis')
mock_mmpose.evaluation = types.ModuleType('mmpose.evaluation')
mock_mmpose.evaluation.functional = types.ModuleType('mmpose.evaluation.functional')

def mock_init_model(*args, **kwargs):
    return None

def mock_inference_topdown(*args, **kwargs):
    return []

def mock_nms(*args, **kwargs):
    return []

mock_mmpose.apis.init_model = mock_init_model
mock_mmpose.apis.inference_topdown = mock_inference_topdown
mock_mmpose.evaluation.functional.nms = mock_nms

sys.modules['mmpose'] = mock_mmpose
sys.modules['mmpose.utils'] = mock_mmpose.utils
sys.modules['mmpose.apis'] = mock_mmpose.apis
sys.modules['mmpose.evaluation'] = mock_mmpose.evaluation
sys.modules['mmpose.evaluation.functional'] = mock_mmpose.evaluation.functional

# Also mock mmseg
import types as _types
mock_mmseg = _types.ModuleType('mmseg')
mock_mmseg.apis = _types.ModuleType('mmseg.apis')
def mock_mmseg_init_model(*args, **kwargs): return None
def mock_mmseg_inference_model(*args, **kwargs): return None
mock_mmseg.apis.init_model = mock_mmseg_init_model
mock_mmseg.apis.inference_model = mock_mmseg_inference_model
sys.modules['mmseg'] = mock_mmseg
sys.modules['mmseg.apis'] = mock_mmseg.apis

# Now patch mmcv
mmcv_path = os.path.join(BLADE_ROOT, "mmcv")
sys.path.insert(0, mmcv_path)

import importlib.util
spec = importlib.util.spec_from_file_location(
    "mmcv", 
    os.path.join(mmcv_path, "mmcv", "__init__.py"),
    submodule_search_locations=[os.path.join(mmcv_path, "mmcv")]
)
mmcv = importlib.util.module_from_spec(spec)
from mmcv.version import __version__ as mmcv_ver
mmcv.__version__ = mmcv_ver
import mmcv.runner
from mmcv.utils import print_log
mmcv.runner.print_log = print_log

# Mock and register DepthEstimator - NO LONGER needed, build_sapiens catches error
print("Skip mock registration - build_sapiens has fallback")
sys.modules['mmcv'] = mmcv

print("mmcv patched OK")

# Now try BLADE
print("\nLoading BLADE...")
from blade.configs.base import root
print(f"Root: {root}")

from blade.configs.blade_inthewild import model as model_cfg
print("Config loaded")

from blade.models.architectures.builder import build_architecture
print("Architecture builder loaded!")

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = build_architecture(model_cfg)
ckpt_path = os.path.join(BLADE_ROOT, 'pretrained', 'epoch_2.pth')
if os.path.exists(ckpt_path):
    from mmcv.runner import load_checkpoint
    checkpoint = load_checkpoint(model, ckpt_path, map_location=device)
    print(f"Checkpoint loaded!")

model = model.to(device)
model.eval()
print(f"\n✓ BLADE ({type(model).__name__}) loaded on {device}")
print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
