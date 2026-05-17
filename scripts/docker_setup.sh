#!/bin/bash
set -e
echo "=== Install system deps ==="
apt-get update -qq && apt-get install -y -qq curl wget libgl1-mesa-glx libglib2.0-0 libgles2-mesa 2>&1 | tail -1

echo "=== Install Python deps ==="
pip install -q tqdm fvcore iopath numpy==1.24.4 wandb matplotlib==3.8.4 colorama requests huggingface-hub safetensors pillow six click chumpy scipy munkres cython fsspec yapf==0.40.1 packaging omegaconf ipdb ftfy regex json_tricks terminaltables modelindex prettytable albumentations smplx==0.1.28 debugpy numba yacs scikit-learn filterpy h5py trimesh scikit-image tensorboardx pyrender torchgeometry joblib boto3 easydict pycocotools colormap timm pyglet future tensorboard cdflib ftfy einops mediapipe mmengine open3d trimesh 2>&1 | tail -1

echo "=== Install MMCV ==="
cd /workspace/blade/mmcv && MMCV_WITH_OPS=1 pip install -e . -q 2>&1 | tail -1

echo "=== Install Sapiens ==="
for dir in engine pretrain pose det seg; do
  cd /workspace/blade/sapiens/$dir && pip install -e . -q 2>&1 | tail -1
done

echo "=== Compile AiOS ==="
cd /workspace/blade/aios_repo/models/aios/ops
python setup.py build_ext --inplace 2>&1 | tail -1
python setup.py install 2>&1 | tail -1

echo "=== Trust NCG ==="
cd /workspace/blade/torch-trust-ncg && python setup.py install 2>&1 | tail -1

echo "=== Verify ==="
cd /workspace
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo "=== Run BLADE ==="
python scripts/blade_runner_v3.py 2>&1
