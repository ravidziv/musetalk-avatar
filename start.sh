#!/bin/bash
set -e

echo "=== MuseTalk Startup ==="

MODELS_DIR="/app/MuseTalk/models"
FINETUNED_DIR="/app/models/finetuned"
MARKER="$MODELS_DIR/.downloaded"

# HuggingFace repo for fine-tuned weights (set via env or default)
HF_FINETUNED_REPO="${HF_FINETUNED_REPO:-ravidsh/musetalk-ravid}"

# Download models only on first cold start (or if marker missing)
if [ ! -f "$MARKER" ]; then
    echo "First boot — downloading models..."

    # 1. MuseTalk pretrained weights
    echo "Downloading MuseTalk v1.5 weights..."
    python3 -c "
from huggingface_hub import snapshot_download
print('Downloading MuseTalk weights...')
snapshot_download(repo_id='TMElyralab/MuseTalk', local_dir='$MODELS_DIR/musetalk_hf')
print('MuseTalk done')
"
    # Organize weights into expected paths
    mkdir -p "$MODELS_DIR/musetalkV15" "$MODELS_DIR/whisper"
    cp -r "$MODELS_DIR/musetalk_hf/models/musetalkV15/"* "$MODELS_DIR/musetalkV15/" 2>/dev/null || true
    cp -r "$MODELS_DIR/musetalk_hf/musetalkV15/"* "$MODELS_DIR/musetalkV15/" 2>/dev/null || true
    cp -r "$MODELS_DIR/musetalk_hf/models/whisper/"* "$MODELS_DIR/whisper/" 2>/dev/null || true
    cp -r "$MODELS_DIR/musetalk_hf/whisper/"* "$MODELS_DIR/whisper/" 2>/dev/null || true

    # 2. VAE
    echo "Downloading VAE..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='stabilityai/sd-vae-ft-mse', local_dir='$MODELS_DIR/sd-vae-ft-mse')
print('VAE done')
"

    # 3. InsightFace (face detection)
    echo "Downloading InsightFace models..."
    python3 -c "
import insightface
from insightface.app import FaceAnalysis
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)
print('InsightFace ready')
" || echo "InsightFace download skipped (will retry on first request)"

    # 4. Fine-tuned UNet weights
    if [ ! -f "$FINETUNED_DIR/unet_finetuned.pth" ]; then
        echo "Downloading fine-tuned weights from $HF_FINETUNED_REPO..."
        python3 -c "
from huggingface_hub import hf_hub_download
import os
try:
    path = hf_hub_download(
        repo_id='$HF_FINETUNED_REPO',
        filename='unet_finetuned.pth',
        local_dir='$FINETUNED_DIR'
    )
    print(f'Fine-tuned weights downloaded to {path}')
except Exception as e:
    print(f'Could not download fine-tuned weights: {e}')
    print('Will use pretrained MuseTalk v1.5 instead')
"
    fi

    touch "$MARKER"
    echo "All models downloaded."
else
    echo "Models already downloaded, skipping."
fi

echo "Starting MuseTalk handler..."
exec python3 -u handler.py
