#!/usr/bin/env bash
# MuseTalk full environment + model setup
# Tested on Colab (Python 3.12, torch 2.10+cu128).
set -e

# --------------------------------------------------------------------------- #
# 1. Clone MuseTalk source
# --------------------------------------------------------------------------- #
if [ ! -d "MuseTalk" ]; then
    git clone https://github.com/TMElyralab/MuseTalk.git
fi

# --------------------------------------------------------------------------- #
# 2. torch/torchvision — use whatever Colab already has
# --------------------------------------------------------------------------- #
python -c "import torch" 2>/dev/null || \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

TORCH_VER=$(python -c "import torch; print(torch.__version__)")
echo "Using torch $TORCH_VER"

# --------------------------------------------------------------------------- #
# 3. Core Python deps (no version pins that fight Colab's environment)
# --------------------------------------------------------------------------- #
pip install -q \
    "transformers>=4.39.2" \
    diffusers \
    huggingface_hub \
    accelerate \
    safetensors \
    "peft==0.10.0" \
    "opencv-python>=4.9.0" \
    Pillow \
    moviepy \
    "imageio[ffmpeg]" \
    "albumentations==1.4.20" \
    openai-whisper \
    "numpy<2.0" \
    scipy \
    gradio \
    pyyaml \
    tqdm \
    gdown \
    requests

# --------------------------------------------------------------------------- #
# 4. OpenMMLab stack
#    openmim drags in ancient pinned deps; install it isolated then restore.
# --------------------------------------------------------------------------- #
echo "Installing OpenMMLab stack..."

# Install openmim but immediately restore the packages it downgrades
pip install -q openmim
pip install -q -U setuptools requests urllib3 tqdm rich filelock packaging

# Detect CUDA version from torch to pick the right mmcv wheel
CUDA_TAG=$(python -c "
import torch, re
v = torch.version.cuda or '11.8'
major, minor = v.split('.')[:2]
print(f'cu{major}{minor}')
")
TORCH_TAG=$(python -c "
import torch
v = torch.version.__version__ if hasattr(torch.version, '__version__') else torch.__version__
# strip +cu... suffix and take major.minor
import re
m = re.match(r'(\d+\.\d+)', v)
print('torch' + m.group(1) if m else 'torch2.0')
")
echo "Detected: $CUDA_TAG / $TORCH_TAG"

mim install mmengine -q

# mmcv 2.1.0 supports torch 2.x; fall back to pip if mim wheel is missing
mim install "mmcv==2.1.0" -q 2>/dev/null || \
    pip install -q "mmcv==2.1.0" \
        -f "https://download.openmmlab.com/mmcv/dist/${CUDA_TAG}/${TORCH_TAG}/index.html"

mim install "mmdet==3.1.0" -q
mim install "mmpose==1.1.0" -q

# Restore peft after mim may have touched it
pip install -q "peft==0.10.0"

# --------------------------------------------------------------------------- #
# 5. FFmpeg (use system ffmpeg on Colab — it's already installed)
# --------------------------------------------------------------------------- #
if command -v ffmpeg &>/dev/null; then
    FFMPEG_PATH=$(dirname "$(command -v ffmpeg)")
    echo "Using system ffmpeg at $FFMPEG_PATH"
else
    # Bare machine fallback: download static binary
    FFMPEG_DIR="$(pwd)/ffmpeg-static"
    if [ ! -d "$FFMPEG_DIR" ]; then
        wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -O ffmpeg.tar.xz
        mkdir -p "$FFMPEG_DIR"
        tar -xf ffmpeg.tar.xz --strip-components=1 -C "$FFMPEG_DIR"
        rm -f ffmpeg.tar.xz
    fi
    FFMPEG_PATH="$FFMPEG_DIR"
    export PATH="$FFMPEG_PATH:$PATH"
fi
echo "FFMPEG_PATH=$FFMPEG_PATH" > .env

# --------------------------------------------------------------------------- #
# 6. Patch diffusers if it still imports the removed cached_download symbol
# --------------------------------------------------------------------------- #
DYNAMIC_UTILS=$(python -c "
import diffusers, os
p = os.path.join(os.path.dirname(diffusers.__file__), 'utils', 'dynamic_modules_utils.py')
print(p if os.path.exists(p) else '')
")
if [ -n "$DYNAMIC_UTILS" ] && grep -q "cached_download" "$DYNAMIC_UTILS" 2>/dev/null; then
    sed -i 's/from huggingface_hub import cached_download, hf_hub_download, model_info/from huggingface_hub import hf_hub_download, model_info/' "$DYNAMIC_UTILS"
    echo "Patched $DYNAMIC_UTILS"
fi

# --------------------------------------------------------------------------- #
# 7. Download model weights
# --------------------------------------------------------------------------- #
MODELS_DIR="MuseTalk/models"
mkdir -p \
    "$MODELS_DIR/musetalk" \
    "$MODELS_DIR/dwpose" \
    "$MODELS_DIR/face-parse-bisent" \
    "$MODELS_DIR/sd-vae-ft-mse" \
    "$MODELS_DIR/whisper"

dl() {
    # dl <url> <dest>  — skip if file already exists and is non-empty
    local url="$1" dest="$2"
    if [ -s "$dest" ]; then
        echo "Already exists: $dest"
    else
        wget -q --show-progress "$url" -O "$dest"
    fi
}

echo "--- Downloading MuseTalk weights ---"
dl "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/musetalk.json" \
   "$MODELS_DIR/musetalk/musetalk.json"
dl "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/pytorch_model.bin" \
   "$MODELS_DIR/musetalk/pytorch_model.bin"

echo "--- Downloading DWPose ---"
dl "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth" \
   "$MODELS_DIR/dwpose/dw-ll_ucoco_384.pth"

echo "--- Downloading face-parse-bisent ---"
if [ ! -s "$MODELS_DIR/face-parse-bisent/79999_iter.pth" ]; then
    gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O "$MODELS_DIR/face-parse-bisent/79999_iter.pth"
fi
dl "https://download.pytorch.org/models/resnet18-5c106cde.pth" \
   "$MODELS_DIR/face-parse-bisent/resnet18-5c106cde.pth"

echo "--- Downloading Stable Diffusion VAE ---"
dl "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json" \
   "$MODELS_DIR/sd-vae-ft-mse/config.json"
dl "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin" \
   "$MODELS_DIR/sd-vae-ft-mse/diffusion_pytorch_model.bin"

echo "--- Downloading Whisper tiny ---"
dl "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt" \
   "$MODELS_DIR/whisper/tiny.pt"

echo ""
echo "Setup complete."
echo "  Inference : python inference.py --video data/video/your.mp4 --audio data/audio/your.wav"
echo "  Web UI    : python app.py --share"
