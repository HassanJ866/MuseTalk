#!/usr/bin/env bash
# MuseTalk full environment + model setup
# Tested on Colab (Python 3.12, torch 2.10+cu128).
# Does NOT use mim/openmim — broken on Python 3.12 due to pkg_resources bug.
# mmcv is built from source when no prebuilt wheel exists for the CUDA version.
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
# 3. Core Python deps
# --------------------------------------------------------------------------- #
pip install -q \
    "transformers>=4.39.2" \
    diffusers \
    huggingface_hub \
    accelerate \
    safetensors \
    "peft>=0.17.0" \
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
#    Python 3.12 removed pkgutil.ImpImporter which pkg_resources uses.
#    We patch mmengine's package_utils.py to use importlib instead.
# --------------------------------------------------------------------------- #
echo "Installing OpenMMLab stack..."

pip install -q "mmengine==0.10.5"

# Completely rewrite mmengine's package_utils.py to remove all pkg_resources
# usage. pkg_resources relies on pkgutil.ImpImporter, removed in Python 3.12.
# We write the replacement to a temp file to avoid any bash quoting issues.
python - << 'PYEOF'
import mmengine, os, ast

path = os.path.join(os.path.dirname(mmengine.__file__), 'utils', 'package_utils.py')

fixed = (
    "# Copyright (c) OpenMMLab. All rights reserved.\n"
    "import importlib.util\n"
    "import os.path as osp\n"
    "import subprocess\n"
    "from importlib.metadata import PackageNotFoundError as DistributionNotFound\n"
    "from importlib.metadata import distribution as _dist\n"
    "\n"
    "\n"
    "def _get_dist(package):\n"
    "    d = _dist(package)\n"
    "    class _D:\n"
    "        version = d.metadata['Version']\n"
    "        location = str(list(d.files)[0].parent.parent) if d.files else ''\n"
    "    return _D()\n"
    "\n"
    "\n"
    "def is_installed(package: str) -> bool:\n"
    "    try:\n"
    "        _get_dist(package)\n"
    "        return True\n"
    "    except DistributionNotFound:\n"
    "        spec = importlib.util.find_spec(package)\n"
    "        return spec is not None and spec.origin is not None\n"
    "\n"
    "\n"
    "def get_installed_path(package: str) -> str:\n"
    "    try:\n"
    "        pkg = _get_dist(package)\n"
    "    except DistributionNotFound as e:\n"
    "        spec = importlib.util.find_spec(package)\n"
    "        if spec is not None and spec.origin is not None:\n"
    "            return osp.dirname(spec.origin)\n"
    "        raise e\n"
    "    possible_path = osp.join(pkg.location, package)\n"
    "    if osp.exists(possible_path):\n"
    "        return possible_path\n"
    "    return osp.join(pkg.location, package2module(package))\n"
    "\n"
    "\n"
    "def package2module(package: str):\n"
    "    d = _dist(package)\n"
    "    txt = d.read_text('top_level.txt')\n"
    "    if txt:\n"
    "        return txt.split('\\n')[0]\n"
    "    raise ValueError(f'can not infer the module name of {package}')\n"
    "\n"
    "\n"
    "def call_command(cmd: list) -> None:\n"
    "    try:\n"
    "        subprocess.check_call(cmd)\n"
    "    except Exception as e:\n"
    "        raise e\n"
    "\n"
    "\n"
    "def install_package(package: str):\n"
    "    if not is_installed(package):\n"
    "        call_command(['python', '-m', 'pip', 'install', package])\n"
)

ast.parse(fixed)  # verify syntax before writing
open(path, 'w').write(fixed)
print("Rewrote:", path)
PYEOF

# Replace mmpose/mmdet/mmcv entirely with onnxruntime-based DWPose.
# mmcv has no Python 3.12 wheels and its setup.py fails on Python 3.12.
# onnxruntime runs the same DWPose ONNX weights with no compilation needed.
pip uninstall -y mmcv-lite mmcv mmdet mmpose 2>/dev/null || true
pip install -q onnxruntime-gpu

# Copy our onnxruntime-based preprocessing.py over MuseTalk's mmpose version
cp scripts/preprocessing.py MuseTalk/musetalk/utils/preprocessing.py
echo "Patched MuseTalk/musetalk/utils/preprocessing.py"

# Restore peft
pip install -q "peft>=0.17.0"

# --------------------------------------------------------------------------- #
# 5. FFmpeg — use system binary on Colab, download static build otherwise
# --------------------------------------------------------------------------- #
if command -v ffmpeg &>/dev/null; then
    FFMPEG_PATH=$(dirname "$(command -v ffmpeg)")
    echo "Using system ffmpeg at $FFMPEG_PATH"
else
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
# 6. Patch diffusers if it still uses the removed cached_download symbol
# --------------------------------------------------------------------------- #
DYNAMIC_UTILS=$(python -c "
import diffusers, os
p = os.path.join(os.path.dirname(diffusers.__file__), 'utils', 'dynamic_modules_utils.py')
print(p if os.path.exists(p) else '')
" 2>/dev/null || true)
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
    "$MODELS_DIR/musetalkV15" \
    "$MODELS_DIR/dwpose" \
    "$MODELS_DIR/face-parse-bisent" \
    "$MODELS_DIR/sd-vae" \
    "$MODELS_DIR/whisper"

dl() {
    local url="$1" dest="$2"
    if [ -s "$dest" ]; then
        echo "Already exists: $dest"
    else
        wget -q --show-progress "$url" -O "$dest"
    fi
}

echo "--- Downloading MuseTalk weights ---"
# musetalk.json is the model config — the newer repo expects it as config.json
dl "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/musetalk.json" \
   "$MODELS_DIR/musetalk/musetalk.json"
# Also save as config.json which unet.py opens directly
dl "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/musetalk.json" \
   "$MODELS_DIR/musetalk/config.json"
dl "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/pytorch_model.bin" \
   "$MODELS_DIR/musetalk/pytorch_model.bin"

# The updated MuseTalk repo also looks for musetalkV15/unet.pth — symlink to the v1 weights
ln -sfn "$(pwd)/$MODELS_DIR/musetalk/pytorch_model.bin" "$MODELS_DIR/musetalkV15/unet.pth"   2>/dev/null || true
ln -sfn "$(pwd)/$MODELS_DIR/musetalk/config.json"       "$MODELS_DIR/musetalkV15/config.json" 2>/dev/null || true

echo "--- Downloading DWPose ---"
dl "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth" \
   "$MODELS_DIR/dwpose/dw-ll_ucoco_384.pth"
dl "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx" \
   "$MODELS_DIR/dwpose/dw-ll_ucoco_384.onnx"
dl "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx" \
   "$MODELS_DIR/dwpose/yolox_l.onnx"

echo "--- Downloading face-parse-bisent ---"
if [ ! -s "$MODELS_DIR/face-parse-bisent/79999_iter.pth" ]; then
    gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O "$MODELS_DIR/face-parse-bisent/79999_iter.pth"
fi
dl "https://download.pytorch.org/models/resnet18-5c106cde.pth" \
   "$MODELS_DIR/face-parse-bisent/resnet18-5c106cde.pth"

echo "--- Downloading Stable Diffusion VAE ---"
dl "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json" \
   "$MODELS_DIR/sd-vae/config.json"
dl "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin" \
   "$MODELS_DIR/sd-vae/diffusion_pytorch_model.bin"

echo "--- Downloading Whisper tiny (HuggingFace format for AutoFeatureExtractor) ---"
# huggingface-cli was renamed to hf in huggingface_hub >= 1.12; fall back if needed
HF_CLI="hf"
command -v hf &>/dev/null || HF_CLI="huggingface-cli"
$HF_CLI download openai/whisper-tiny \
    --local-dir "$MODELS_DIR/whisper" \
    --include "preprocessor_config.json" "config.json" "tokenizer_config.json" \
              "vocab.json" "merges.txt" "normalizer.json" "added_tokens.json" \
              "special_tokens_map.json" "pytorch_model.bin" "model.safetensors"

echo ""
echo "Setup complete."
echo "  Inference : python inference.py --video data/video/your.mp4 --audio data/audio/your.wav"
echo "  Web UI    : python app.py --share"
