# MuseTalk RunPod — Slim inference image
# Models downloaded at startup via start.sh
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MUSETALK_PATH=/app/MuseTalk
ENV HF_HOME=/app/cache/huggingface

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone MuseTalk (shallow, no .git history)
RUN git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git && \
    rm -rf /app/MuseTalk/.git

# Install ONLY inference deps (skip gradio/moviepy/dev tools from requirements.txt)
RUN pip install --no-cache-dir \
    runpod \
    diffusers==0.30.2 \
    transformers>=4.36.0 \
    accelerate \
    insightface>=0.7.3 \
    onnxruntime-gpu>=1.16.0 \
    librosa>=0.10.1 \
    soundfile>=0.12.1 \
    mediapipe>=0.10.8 \
    einops>=0.7.0 \
    omegaconf>=2.3.0 \
    huggingface_hub>=0.20.0 \
    opencv-python-headless>=4.8.0 \
    Pillow>=10.0.0 \
    scipy>=1.11.0 \
    ffmpeg-python>=0.2.0 \
    tqdm>=4.66.0 \
    pyyaml>=6.0

# Copy handler, startup script, and avatar image
COPY handler-full.py /app/MuseTalk/handler.py
COPY start.sh /app/start.sh
COPY ravid-photo.jpg /app/avatar.jpg

RUN chmod +x /app/start.sh && \
    mkdir -p /app/cache /app/avatars /app/models/finetuned /app/MuseTalk/models

WORKDIR /app/MuseTalk

CMD ["/app/start.sh"]
