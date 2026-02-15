"""
Full MuseTalk RunPod Serverless Handler
Real lip-sync avatar generation using MuseTalk v1.5 with fine-tuned weights.
"""

import os
import io
import sys
import copy
import base64
import tempfile
import time
import runpod
import numpy as np
import cv2
from PIL import Image

# Add MuseTalk to path
MUSETALK_PATH = os.environ.get("MUSETALK_PATH", "/app/MuseTalk")
sys.path.insert(0, MUSETALK_PATH)

# Fine-tuned weight path
FINETUNED_UNET_PATH = os.environ.get(
    "FINETUNED_UNET_PATH",
    "/app/models/finetuned/unet_finetuned.pth"
)

# Global model instances
musetalk_model = None
avatar_cache = {}
startup_time = time.time()
init_error = None


def initialize_model():
    """Initialize MuseTalk model once on cold start."""
    global musetalk_model, init_error

    if musetalk_model is not None:
        return musetalk_model

    print("Initializing MuseTalk model...")

    try:
        import torch

        # Change to MuseTalk dir so relative model paths work
        os.chdir(MUSETALK_PATH)

        # Check what model files exist
        models_dir = os.path.join(MUSETALK_PATH, "models")
        print(f"Models directory: {models_dir}")
        print(f"Models dir exists: {os.path.exists(models_dir)}")
        if os.path.exists(models_dir):
            for root, dirs, files in os.walk(models_dir):
                for f in files:
                    fpath = os.path.join(root, f)
                    fsize = os.path.getsize(fpath) / (1024*1024)
                    print(f"  {os.path.relpath(fpath, models_dir)}: {fsize:.1f} MB")

        from musetalk.utils.utils import load_all_model
        from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
        from musetalk.utils.blending import get_image
        from musetalk.whisper.audio2feature import Audio2Feature

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        if device.type == "cuda":
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.2f} GB")

        # Load MuseTalk models — returns (vae, unet, pe)
        print("Loading MuseTalk models...")
        vae, unet, pe = load_all_model(
            vae_type="sd-vae-ft-mse",
        )

        # Load fine-tuned UNet weights if available
        if os.path.exists(FINETUNED_UNET_PATH):
            print(f"Loading fine-tuned UNet weights from {FINETUNED_UNET_PATH}")
            state_dict = torch.load(FINETUNED_UNET_PATH, map_location=device)
            unet.model.load_state_dict(state_dict)
            print("Fine-tuned weights loaded successfully")
        else:
            print("No fine-tuned weights found, using pretrained v1.5")

        # Move models to device
        pe = pe.to(device)
        vae.vae = vae.vae.to(device)
        unet.model = unet.model.to(device)

        # Initialize audio processor (uses MuseTalk's custom whisper)
        whisper_path = os.path.join(MUSETALK_PATH, "models", "whisper", "tiny.pt")
        print(f"Loading whisper from {whisper_path}")
        audio_processor = Audio2Feature(model_path=whisper_path)

        timesteps = torch.tensor([0], device=device)

        musetalk_model = {
            "mock": False,
            "device": device,
            "vae": vae,
            "unet": unet,
            "pe": pe,
            "audio_processor": audio_processor,
            "timesteps": timesteps,
            "get_landmark_and_bbox": get_landmark_and_bbox,
            "read_imgs": read_imgs,
            "coord_placeholder": coord_placeholder,
            "get_image": get_image,
        }

        print("MuseTalk model initialized successfully")
        return musetalk_model

    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        init_error = str(e)
        musetalk_model = {"mock": True, "device": "cpu", "error": str(e)}
        return musetalk_model


def prepare_avatar(image_data: bytes, avatar_id: str) -> dict:
    """Prepare avatar image — extract face, compute latents."""
    global avatar_cache, musetalk_model

    model = initialize_model()
    print(f"Preparing avatar: {avatar_id}")

    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image_np = np.array(image)

    if model.get("mock", True):
        avatar_cache[avatar_id] = {
            "image": image_np,
            "width": image.width,
            "height": image.height,
            "mock": True,
        }
        return {
            "avatar_id": avatar_id,
            "width": image.width,
            "height": image.height,
            "status": "prepared",
            "mode": "mock",
        }

    try:
        import torch

        vae = model["vae"]
        get_landmark_and_bbox = model["get_landmark_and_bbox"]
        coord_placeholder = model["coord_placeholder"]

        # Save temp image for face detection
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f.name)
            temp_path = f.name

        try:
            # Extract face landmarks and bounding box
            coord_list, frame_list = get_landmark_and_bbox([temp_path], 0)
            if len(coord_list) == 0:
                raise ValueError("No face detected in image")

            # Pre-compute VAE latents for each frame (masked + ref concat)
            input_latent_list = []
            for bbox, frame in zip(coord_list, frame_list):
                if bbox == coord_placeholder:
                    continue
                x1, y1, x2, y2 = bbox
                crop_frame = frame[y1:y2, x1:x2]
                crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                latents = vae.get_latents_for_unet(crop_frame)
                input_latent_list.append(latents)

            avatar_cache[avatar_id] = {
                "image": image_np,
                "frame_list": frame_list,
                "coord_list": coord_list,
                "input_latent_list": input_latent_list,
                "width": image.width,
                "height": image.height,
                "mock": False,
            }

            return {
                "avatar_id": avatar_id,
                "width": image.width,
                "height": image.height,
                "status": "prepared",
                "mode": "real",
            }

        finally:
            os.unlink(temp_path)

    except Exception as e:
        print(f"Error preparing avatar: {e}")
        import traceback
        traceback.print_exc()
        avatar_cache[avatar_id] = {
            "image": image_np,
            "width": image.width,
            "height": image.height,
            "mock": True,
        }
        return {
            "avatar_id": avatar_id,
            "width": image.width,
            "height": image.height,
            "status": "prepared",
            "mode": "mock",
            "error": str(e),
        }


def generate_frames(audio_data: bytes, avatar_id: str) -> dict:
    """Generate lip-synced video from audio."""
    global musetalk_model, avatar_cache

    model = initialize_model()

    # Get or create avatar
    if avatar_id not in avatar_cache:
        print(f"Avatar {avatar_id} not found, using default")
        default_avatar_path = "/app/avatar.jpg"
        if os.path.exists(default_avatar_path):
            with open(default_avatar_path, "rb") as f:
                prepare_avatar(f.read(), avatar_id)
        else:
            placeholder = np.ones((512, 512, 3), dtype=np.uint8) * 200
            avatar_cache[avatar_id] = {
                "image": placeholder,
                "width": 512,
                "height": 512,
                "mock": True,
            }

    avatar = avatar_cache[avatar_id]

    if model.get("mock", True) or avatar.get("mock", True):
        return generate_mock_frames(audio_data, avatar)
    else:
        return generate_real_frames(audio_data, avatar, model)


def generate_mock_frames(audio_data: bytes, avatar: dict) -> dict:
    """Generate mock video for testing."""
    import struct

    if audio_data[:4] == b'RIFF':
        try:
            byte_rate = struct.unpack('<I', audio_data[28:32])[0]
            data_size = len(audio_data) - 44
            audio_duration = data_size / byte_rate if byte_rate > 0 else 2.0
        except Exception:
            audio_duration = 2.0
    else:
        audio_duration = len(audio_data) / (128000 / 8)

    num_frames = max(1, min(int(audio_duration * 25), 300))
    print(f"Generating {num_frames} mock frames")

    base_image = avatar["image"]
    target_size = 512
    if base_image.shape[0] != target_size or base_image.shape[1] != target_size:
        base_image = cv2.resize(base_image, (target_size, target_size))

    frame_bgr = cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_path = f.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 25.0, (target_size, target_size))
        for _ in range(num_frames):
            out.write(frame_bgr)
        out.release()

        with open(video_path, 'rb') as f:
            video_data = f.read()

        return {
            "type": "video",
            "format": "mp4",
            "fps": 25,
            "frame_count": num_frames,
            "duration": num_frames / 25.0,
            "width": target_size,
            "height": target_size,
            "video_base64": base64.b64encode(video_data).decode('utf-8'),
        }
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)


def generate_real_frames(audio_data: bytes, avatar: dict, model: dict) -> dict:
    """Generate lip-synced video using MuseTalk's masked inpainting pipeline."""
    import torch
    from musetalk.utils.utils import datagen

    print("Generating real lip-sync video...")

    device = model["device"]
    vae = model["vae"]
    unet = model["unet"]
    pe = model["pe"]
    audio_processor = model["audio_processor"]
    timesteps = model["timesteps"]
    get_image = model["get_image"]

    # Save audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_data)
        audio_path = f.name

    video_path = None

    try:
        # Extract audio features using whisper
        print("Processing audio...")
        whisper_feature = audio_processor.audio2feat(audio_path)
        whisper_chunks = audio_processor.feature2chunks(
            feature_array=whisper_feature, fps=25
        )
        # Convert numpy chunks to tensors
        whisper_chunks = [torch.from_numpy(c).float().to(device) for c in whisper_chunks]

        print(f"Generated {len(whisper_chunks)} audio chunks")

        # Get pre-computed avatar data
        input_latent_list = avatar["input_latent_list"]
        frame_list = avatar["frame_list"]
        coord_list = avatar["coord_list"]

        # Cycle forward+backward for smooth looping
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]

        # Batch inference with datagen
        gen = datagen(
            whisper_chunks=whisper_chunks,
            vae_encode_latents=input_latent_list_cycle,
            batch_size=8,
            delay_frame=0,
            device=device,
        )

        res_frame_list = []
        total_batches = int(np.ceil(len(whisper_chunks) / 8))

        print("Running inference...")
        with torch.no_grad():
            for batch_idx, (whisper_batch, latent_batch) in enumerate(gen):
                # Positional encoding on audio features
                audio_emb = pe(whisper_batch)
                latent_batch = latent_batch.to(dtype=unet.model.dtype)

                # UNet forward: [masked_latent, ref_latent] + audio → predicted latent
                pred_latents = unet.model(
                    latent_batch, timesteps,
                    encoder_hidden_states=audio_emb
                ).sample

                # Decode predicted latents back to BGR images
                recon = vae.decode_latents(pred_latents)
                for frame in recon:
                    res_frame_list.append(frame)

                if batch_idx % 5 == 0:
                    print(f"Batch {batch_idx}/{total_batches}")

        print(f"Generated {len(res_frame_list)} frames")

        # Blend generated faces back into original frames
        h, w = frame_list[0].shape[:2]
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 25.0, (w, h))

        for i, res_frame in enumerate(res_frame_list):
            bbox = coord_list_cycle[i % len(coord_list_cycle)]
            ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])
            x1, y1, x2, y2 = bbox

            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except Exception:
                continue

            combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2])
            out.write(combine_frame)

        out.release()

        with open(video_path, 'rb') as f:
            video_data = f.read()

        video_b64 = base64.b64encode(video_data).decode('utf-8')
        print(f"Video: {len(video_data)} bytes, {len(res_frame_list)} frames")

        return {
            "type": "video",
            "format": "mp4",
            "fps": 25,
            "frame_count": len(res_frame_list),
            "duration": len(res_frame_list) / 25.0,
            "width": w,
            "height": h,
            "video_base64": video_b64,
        }

    except Exception as e:
        print(f"Error in real frame generation: {e}")
        import traceback
        traceback.print_exc()
        return generate_mock_frames(audio_data, avatar)

    finally:
        os.unlink(audio_path)
        if video_path and os.path.exists(video_path):
            os.unlink(video_path)


def handler(event):
    """RunPod serverless handler."""
    try:
        job_input = event.get("input", {})
        action = job_input.get("action", "generate")
        avatar_id = job_input.get("avatar_id", "default")

        print(f"Handler: action={action}, avatar_id={avatar_id}")

        if action == "status":
            import torch
            gpu_info = {}
            if torch.cuda.is_available():
                gpu_info = {
                    "name": torch.cuda.get_device_name(0),
                    "memory_total_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2),
                    "memory_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
                    "memory_reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2),
                }
            return {
                "status": "ok",
                "uptime_seconds": round(time.time() - startup_time, 1),
                "model_loaded": musetalk_model is not None,
                "model_mode": "real" if (musetalk_model and not musetalk_model.get("mock")) else "mock",
                "init_error": init_error,
                "gpu": gpu_info,
                "avatars_cached": list(avatar_cache.keys()),
                "finetuned_weights": os.path.exists(FINETUNED_UNET_PATH),
                "models_dir_exists": os.path.exists(os.path.join(MUSETALK_PATH, "models")),
            }

        elif action == "prepare":
            image_b64 = job_input.get("image_base64", "")
            if not image_b64:
                return {"error": "image_base64 is required for prepare action"}
            image_data = base64.b64decode(image_b64)
            return prepare_avatar(image_data, avatar_id)

        elif action == "generate":
            audio_b64 = job_input.get("audio_base64", "")
            if not audio_b64:
                return {"error": "audio_base64 is required for generate action"}
            audio_data = base64.b64decode(audio_b64)
            result = generate_frames(audio_data, avatar_id)
            result["avatar_id"] = avatar_id
            return result

        else:
            return {"error": f"Unknown action: {action}. Valid actions: status, prepare, generate"}

    except Exception as e:
        print(f"Handler error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# Initialize on import (cold start optimization)
print("Starting MuseTalk handler...")
try:
    initialize_model()
except Exception as e:
    print(f"WARNING: Model initialization failed: {e}")
    print("Handler will run in mock mode")

# RunPod serverless entrypoint
runpod.serverless.start({"handler": handler})
