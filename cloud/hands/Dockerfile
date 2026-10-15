# EgoSmith hand toolkit for the RTX 5090: PyTorch 2.7 with CUDA 12.8 instead of
# the toolkit's 2.5.1/cu121, which has no Blackwell (sm_120) kernels.
# MANO models are licensed and mounted at run time, never baked in.
FROM python:3.10-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/egosmith
COPY egosmith_hand_mano_toolkit/ /opt/egosmith/
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps ./thirdparty/chumpy \
    && pip install --no-cache-dir -e . \
    && pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless || true \
    && pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless==4.10.0.84
RUN bash scripts/fetch_hawor.sh
RUN mkdir -p weights/hawor/checkpoints weights/external _DATA/data/mano _DATA/data_left/mano_left \
    && curl -fL --retry 3 -o weights/hawor/checkpoints/hawor.ckpt https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt \
    && curl -fL --retry 3 -o weights/hawor/model_config.yaml https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml \
    && curl -fL --retry 3 -o weights/external/detector.pt https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt \
    && cp .external/HaWoR/_DATA/data/mano_mean_params.npz _DATA/data/mano_mean_params.npz
