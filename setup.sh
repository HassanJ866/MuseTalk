#!/usr/bin/env bash
# MuseTalk full environment + model setup
# Run once on a fresh Linux/Colab instance with a CUDA-capable GPU.
set -e

# --------------------------------------------------------------------------- #
# 1. Clone MuseTalk source
# --------------------------------------------------------------------------- #
if [ ! -d "MuseTalk" ]; then
    git clone https://github.com/TMElyralab/MuseTalk.git
fi

# --------------------------------------------------------------------------- #
# 2. Python dependencies
# --------------------------------------------------------------------------- #
pip install -r requirements.txt

# OpenMMLab stack (exact versions validated in notebook)
pip install --no-cache-dir -U openmim
mim install mmengine
mim uninstall mmcv -y 2>/dev/null || true
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"
pip install "peft==0.10.0"

# --------------------------------------------------------------------------- #
# 3. FFmpeg static binary (Linux x86-64)
# --------------------------------------------------------------------------- #
FFMPEG_DIR="$(pwd)/ffmpeg-4.4-amd64-static"
if [ ! -d "$FFMPEG_DIR" ]; then
    wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -O ffmpeg.tar.xz
    tar -xf ffmpeg.tar.xz
    EXTRACTED=$(tar -tf ffmpeg.tar.xz | head -1 | cut -d/ -f1)
    mv "$EXTRACTED" ffmpeg-4.4-amd64-static 2>/dev/null || true
    rm -f ffmpeg.tar.xz
fi
export FFMPEG_PATH="$FFMPEG_DIR"
echo "FFMPEG_PATH=$FFMPEG_DIR" >> .env

# --------------------------------------------------------------------------- #
# 4. Fix huggingface_hub / diffusers incompatibility
#    (cached_download was removed in huggingface_hub >= 0.24)
# --------------------------------------------------------------------------- #
DYNAMIC_UTILS=$(python -c "import diffusers; import os; print(os.path.join(os.path.dirname(diffusers.__file__), 'utils', 'dynamic_modules_utils.py'))")
if grep -q "cached_download" "$DYNAMIC_UTILS" 2>/dev/null; then
    sed -i 's/from huggingface_hub import cached_download, hf_hub_download, model_info/from huggingface_hub import hf_hub_download, model_info/' "$DYNAMIC_UTILS"
    echo "Patched diffusers/utils/dynamic_modules_utils.py"
fi

# --------------------------------------------------------------------------- #
# 5. Download model weights
# --------------------------------------------------------------------------- #
MODELS_DIR="MuseTalk/models"
mkdir -p \
    "$MODELS_DIR/musetalk" \
    "$MODELS_DIR/dwpose" \
    "$MODELS_DIR/face-parse-bisent" \
    "$MODELS_DIR/sd-vae-ft-mse" \
    "$MODELS_DIR/whisper"

echo "Downloading MuseTalk weights..."
wget -q --show-progress -nc \
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/musetalk.json" \
    -O "$MODELS_DIR/musetalk/musetalk.json"

wget -q --show-progress -nc \
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/pytorch_model.bin" \
    -O "$MODELS_DIR/musetalk/pytorch_model.bin"

echo "Downloading DWPose..."
wget -q --show-progress -nc \
    "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth" \
    -O "$MODELS_DIR/dwpose/dw-ll_ucoco_384.pth"

echo "Downloading face-parse-bisent weights..."
# 79999_iter.pth is on Google Drive — use gdown
pip install -q gdown
gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O "$MODELS_DIR/face-parse-bisent/79999_iter.pth"
wget -q --show-progress -nc \
    "https://download.pytorch.org/models/resnet18-5c106cde.pth" \
    -O "$MODELS_DIR/face-parse-bisent/resnet18-5c106cde.pth"

echo "Downloading Stable Diffusion VAE..."
wget -q --show-progress -nc \
    "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json" \
    -O "$MODELS_DIR/sd-vae-ft-mse/config.json"
wget -q --show-progress -nc \
    "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin" \
    -O "$MODELS_DIR/sd-vae-ft-mse/diffusion_pytorch_model.bin"

echo "Downloading Whisper tiny..."
wget -q --show-progress -nc \
    "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt" \
    -O "$MODELS_DIR/whisper/tiny.pt"

echo ""
echo "Setup complete. Run inference with:"
echo "  python inference.py --video data/video/your.mp4 --audio data/audio/your.wav"
echo "Or launch the web UI with:"
echo "  python app.py"
