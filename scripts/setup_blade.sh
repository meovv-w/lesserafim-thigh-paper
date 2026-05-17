#!/bin/bash
# BLADE Environment Setup Script (CUDA 13 compatible approach)
# Run from: /mnt/d/HERMES/study/lesserafim-thigh-paper/

set -e

BLADE_DIR="/mnt/d/HERMES/study/lesserafim-thigh-paper/blade"
cd "$BLADE_DIR"

# Source conda
source ~/miniconda3/etc/profile.d/conda.sh

echo "=== Step 1: Create conda environment ==="
conda create -y -n blade_env python=3.9.19
conda activate blade_env

echo "=== Step 2: Install CUDA 11.8 toolkit via conda ==="
conda install -y -c conda-forge gcc_linux-64=11 gxx_linux-64=11 sysroot_linux-64=2.17
conda install -y -c nvidia cuda-toolkit=11.8.0 cuda-nvcc=11.8.89 cuda-cudart-dev=11.8.89
conda install -y -c nvidia cuda-cusparse-dev=11.8.0 cuda-cusparse=11.8.0

# Set compiler environment
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export NVCCFLAGS="--compiler-bindir=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CUDA_HOME"

echo "=== Step 3: Install Thrust & CUB ==="
cd /tmp
git clone --recursive --branch 1.16.0 https://github.com/NVIDIA/thrust.git thrust-1.16 2>/dev/null || echo "thrust already cloned"
mkdir -p "$CONDA_PREFIX/include"
rsync -a thrust-1.16/thrust "$CONDA_PREFIX/include/" 2>/dev/null
rsync -a thrust-1.16/dependencies/cub/cub "$CONDA_PREFIX/include/" 2>/dev/null

export CPATH="$CONDA_PREFIX/include:$CUDA_HOME/include${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:$CUDA_HOME/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$CUDA_HOME/lib:$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "=== Step 4: Install PyTorch and core deps ==="
cd "$BLADE_DIR"
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install fvcore iopath numpy==1.24.4 wandb

echo "=== Step 5: Install pytorch3d ==="
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt201/download.html

echo "=== Step 6: Install misc deps ==="
pip install matplotlib==3.8.4 colorama requests huggingface-hub safetensors pillow six click openxlab
pip install chumpy scipy munkres tqdm cython fsspec yapf==0.40.1 packaging omegaconf ipdb ftfy regex
pip install json_tricks terminaltables modelindex prettytable albumentations smplx==0.1.28
pip install debugpy numba yacs scikit-learn filterpy h5py trimesh scikit-image tensorboardx pyrender
pip install torchgeometry joblib boto3 easydict pycocotools colormap pytorch-transformers pickle5 plyfile
pip install timm pyglet future tensorboard cdflib ftfy einops tqdm numpy==1.23.1 mediapipe

echo "=== Step 7: Install project dependencies ==="
cd "$BLADE_DIR"
# MMCV
cd mmcv && MMCV_WITH_OPS=1 pip install -e . -v 2>&1 | tail -5 && cd ..

# Sapiens
cd sapiens/engine && pip install -e . -v 2>&1 | tail -3 && cd ../pretrain && pip install -e . -v 2>&1 | tail -3
cd ../pose && pip install -e . -v 2>&1 | tail -3 && cd ../det && pip install -e . -v 2>&1 | tail -3
cd ../seg && pip install -e . -v 2>&1 | tail -3 && cd ../.. && pip install -e . -v 2>&1 | tail -3

pip install ffmpeg astropy easydev pandas rtree vedo codecov flake8 interrogate isort pytest surrogate xdoctest setuptools loguru open3d omegaconf

# Custom ops
cd aios_repo/models/aios/ops && python setup.py build install 2>&1 | tail -5 && cd ../../../..
cd torch-trust-ncg && python setup.py install 2>&1 | tail -5 && cd ..

pip install numpy==1.23.1

echo "=== Step 8: Setup activation scripts ==="
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/10_nvrtc11.sh" <<'EOS'
NVRTC_LIB_DIR="$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cuda_nvrtc/lib"
export _OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$NVRTC_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EOS

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/10_nvrtc11.sh" <<'EOS'
if [ -n "${_OLD_LD_LIBRARY_PATH+x}" ]; then
  export LD_LIBRARY_PATH="$_OLD_LD_LIBRARY_PATH"
  unset _OLD_LD_LIBRARY_PATH
else
  unset LD_LIBRARY_PATH
fi
EOS

cat > "$CONDA_PREFIX/etc/conda/activate.d/20_cuda11_toolchain.sh" <<'EOS'
if [ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" ]; then
  export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
  export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
  export CUDAHOSTCXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
  export NVCCFLAGS="--compiler-bindir=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
fi
EOS

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/20_cuda11_toolchain.sh" <<'EOS'
unset CC CXX CUDAHOSTCXX NVCCFLAGS
EOS

# EGL
export PYGLET_HEADLESS=True
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
[ -f "$CONDA_PREFIX/lib/libEGL.so" ] || ln -sf "$CONDA_PREFIX/lib/libEGL.so.1" "$CONDA_PREFIX/lib/libEGL.so" 2>/dev/null || true

echo ""
echo "=== BLADE environment setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. conda activate blade_env"
echo "  2. Download models (see download_models.sh)"
echo "  3. Download SMPL-X models manually from smpl-x.is.tue.mpg.de"
echo "  4. Run: MINI_BATCHSIZE=5 python api/test_api.py ./demo_images/"
