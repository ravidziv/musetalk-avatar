#!/bin/bash
# NO set -e — we want to continue even if some downloads fail

echo "=== MuseTalk Startup ==="
echo "Disk space:" && df -h /app 2>/dev/null || df -h /
echo "GPU info:" && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"

MODELS_DIR="/app/MuseTalk/models"
FINETUNED_DIR="/app/models/finetuned"
MARKER="$MODELS_DIR/.downloaded"

# HuggingFace repo for fine-tuned weights (set via env or default)
HF_FINETUNED_REPO="${HF_FINETUNED_REPO:-ravid/musetalk-ravid}"

# Download models only on first cold start (or if marker missing)
if [ ! -f "$MARKER" ]; then
    echo "First boot — downloading models..."

    # 1. MuseTalk pretrained weights
    echo "[1/4] Downloading MuseTalk v1.5 weights..."
    python3 -c "
from huggingface_hub import snapshot_download
import sys
try:
    print('Downloading MuseTalk weights...')
    snapshot_download(
        repo_id='TMElyralab/MuseTalk',
        local_dir='$MODELS_DIR/musetalk_hf',
        ignore_patterns=['*.mp4', '*.avi', '*.png', '*.jpg', '*.gif', 'demo/*', 'docs/*', '.git*']
    )
    print('MuseTalk done')
except Exception as e:
    print(f'ERROR downloading MuseTalk: {e}', file=sys.stderr)
    sys.exit(1)
" || echo "WARNING: MuseTalk download failed"

    # Organize weights into expected paths
    mkdir -p "$MODELS_DIR/musetalkV15" "$MODELS_DIR/whisper" "$MODELS_DIR/dwpose"
    # Try multiple possible source paths from the HF repo
    for src_dir in "$MODELS_DIR/musetalk_hf/models/musetalkV15" "$MODELS_DIR/musetalk_hf/musetalkV15"; do
        [ -d "$src_dir" ] && cp -r "$src_dir/"* "$MODELS_DIR/musetalkV15/" 2>/dev/null && echo "Copied musetalkV15 from $src_dir"
    done
    for src_dir in "$MODELS_DIR/musetalk_hf/models/whisper" "$MODELS_DIR/musetalk_hf/whisper"; do
        [ -d "$src_dir" ] && cp -r "$src_dir/"* "$MODELS_DIR/whisper/" 2>/dev/null && echo "Copied whisper from $src_dir"
    done
    for src_dir in "$MODELS_DIR/musetalk_hf/models/dwpose" "$MODELS_DIR/musetalk_hf/dwpose"; do
        [ -d "$src_dir" ] && cp -r "$src_dir/"* "$MODELS_DIR/dwpose/" 2>/dev/null && echo "Copied dwpose from $src_dir"
    done

    # Show what we got
    echo "Model files after MuseTalk download:"
    find "$MODELS_DIR" -type f -name "*.pt" -o -name "*.pth" -o -name "*.bin" -o -name "*.safetensors" -o -name "*.json" -o -name "*.onnx" 2>/dev/null | head -30

    # 2. VAE
    echo "[2/4] Downloading VAE..."
    python3 -c "
from huggingface_hub import snapshot_download
import sys
try:
    snapshot_download(repo_id='stabilityai/sd-vae-ft-mse', local_dir='$MODELS_DIR/sd-vae-ft-mse')
    print('VAE done')
except Exception as e:
    print(f'ERROR downloading VAE: {e}', file=sys.stderr)
" || echo "WARNING: VAE download failed"

    # 3. InsightFace (face detection)
    echo "[3/4] Downloading InsightFace models..."
    python3 -c "
import sys
try:
    import insightface
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)
    print('InsightFace ready')
except Exception as e:
    print(f'InsightFace download warning: {e}', file=sys.stderr)
    print('Will retry on first request')
" || echo "InsightFace download skipped (will retry on first request)"

    # 4. Fine-tuned UNet weights
    echo "[4/4] Downloading fine-tuned weights..."
    if [ ! -f "$FINETUNED_DIR/unet_finetuned.pth" ]; then
        python3 -c "
from huggingface_hub import hf_hub_download
import sys
try:
    path = hf_hub_download(
        repo_id='$HF_FINETUNED_REPO',
        filename='unet_finetuned.pth',
        local_dir='$FINETUNED_DIR'
    )
    print(f'Fine-tuned weights downloaded to {path}')
except Exception as e:
    print(f'Could not download fine-tuned weights: {e}', file=sys.stderr)
    print('Will use pretrained MuseTalk v1.5 instead')
"
    else
        echo "Fine-tuned weights already present."
    fi

    # Final disk check
    echo "Disk space after downloads:" && df -h /app 2>/dev/null || df -h /
    echo "Model directory size:" && du -sh "$MODELS_DIR" 2>/dev/null || echo "unknown"

    touch "$MARKER"
    echo "All models downloaded."
else
    echo "Models already downloaded, skipping."
fi

echo "=== Starting MuseTalk handler ==="
exec python3 -u handler.py
