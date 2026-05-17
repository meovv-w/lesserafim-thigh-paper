FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget vim build-essential libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender-dev libgles2-mesa \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy BLADE repo
COPY blade/ ./blade/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Install BLADE dependencies
RUN cd blade && pip install --no-cache-dir \
    tqdm fvcore iopath numpy==1.24.4 wandb \
    matplotlib==3.8.4 colorama requests huggingface-hub safetensors \
    pillow six click chumpy scipy munkres cython fsspec yapf==0.40.1 \
    packaging omegaconf ipdb ftfy regex json_tricks terminaltables \
    modelindex prettytable albumentations smplx==0.1.28 debugpy numba \
    yacs scikit-learn filterpy h5py trimesh scikit-image tensorboardx \
    pyrender torchgeometry joblib boto3 easydict pycocotools colormap \
    timm pyglet future tensorboard cdflib ftfy einops mediapipe \
    mmengine open3d trimesh

# Install BLADE's custom mmcv
RUN cd blade/mmcv && MMCV_WITH_OPS=1 pip install --no-cache-dir -e . && cd ../..

# Install Sapiens
RUN cd blade/sapiens/engine && pip install -e . && cd ../../.. && \
    cd blade/sapiens/pretrain && pip install -e . && cd ../../.. && \
    cd blade/sapiens/pose && pip install -e . && cd ../../.. && \
    cd blade/sapiens/det && pip install -e . && cd ../../.. && \
    cd blade/sapiens/seg && pip install -e . && cd ../../..

# Install AiOS ops (with native CUDA 11.8)
RUN cd blade/aios_repo/models/aios/ops && \
    python setup.py build_ext --inplace && \
    python setup.py install && cd ../../../../..

# Install torch-trust-ncg
RUN cd blade/torch-trust-ncg && python setup.py install && cd ..

# Verify
RUN python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

CMD ["bash"]
