#!/usr/bin/env python3
"""
AnimateLCM vid2vid augmentation: vary gloves + skin tone clearly.

Edit the CONSTANTS section below to tweak behaviour.
"""

import os, time, random
import torch, imageio, numpy as np
from PIL import Image
from diffusers import AnimateDiffVideoToVideoPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

# ===========================
# ========= CONSTANTS ========
# ===========================

# --- I/O ---
VIDEO_PATH   = "1.mp4"        # source video
OUT_DIR      = "out_variants" # output folder
GIF_FPS      = 8

# --- Model / pipeline ---
BASE_MODEL   = "emilianJR/epiCRealism"   # SD1.5 finetune
ADAPTER_REPO = "wangfuyun/AnimateLCM"    # AnimateLCM motion adapter + LoRA
LCM_LORA_W   = "AnimateLCM_sd15_t2v_lora.safetensors"
LCM_SCALE    = 0.8                       # LoRA scale

# --- Generation params ---
RESOLUTION   = 512        # resize frames to RESOLUTION x RESOLUTION (must be multiple of 64)
NUM_FRAMES   = 16         # trim/pad to this many frames
STEPS        = 14         # diffusion steps (LCM works ~6-16; higher = more change)
GUIDANCE     = 2.2        # classifier-free guidance (LCM: 1.5–2.5)
STRENGTH     = 0.72       # how far to deviate from source video (0=keep, 1=ignore)
FIXED_SEED   = -1         # -1 = random each variant, else int seed

# --- GPU memory ---
ENABLE_CPU_OFFLOAD = False # set True if VRAM is low

# --- Color channel handling ---
FORCE_BGR_SWAP = False      # set True if you know frames are BGR

# --- Variant options ---
GLOVE_OPTIONS = [
    "wearing bright blue nitrile gloves, glove texture clearly visible",
    "bare skin fully visible on fingers and palm"
]

SKIN_OPTIONS = [
    "very light (pale) skin tone",
    "light yellow tan skin tone",
    "medium brown skin tone",
    "dark brown skin tone",
    "very dark (deep black) skin tone"
]

# ===========================
# ====== HELPERS =============
# ===========================

def load_video_bgr_safe(path, max_frames=16, size=512, force_bgr=False):
    """
    Read video, heuristic BGR->RGB swap if needed, resize, pad/trim.
    """
    reader = imageio.get_reader(path)
    frames = []
    use_bgr_swap = False
    for i, arr in enumerate(reader):
        if i >= max_frames:
            break
        if force_bgr:
            arr = arr[..., ::-1]
        elif i == 0:
            rgb_guess  = Image.fromarray(arr).convert("RGB")
            bgr_swap   = Image.fromarray(arr[..., ::-1]).convert("RGB")
            use_bgr_swap = np.mean(np.array(bgr_swap)[:, :, 0]) > np.mean(np.array(rgb_guess)[:, :, 0])
            if use_bgr_swap: arr = arr[..., ::-1]
        else:
            if use_bgr_swap: arr = arr[..., ::-1]
        img = Image.fromarray(arr).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        frames.append(img)
    reader.close()
    if not frames:
        raise ValueError(f"{path} had 0 frames")
    while len(frames) < max_frames:
        frames.append(frames[-1].copy())
    return frames

def build_prompt(glove_desc, skin_desc):
    """
    Strong positive prompt pushing the model to show what we want.
    """
    return (
        "chemistry titration experiment in a laboratory, "
        "one right hand clearly visible operating the burette stopcock, "
        "another hand holding the conical flask, "
        f"{glove_desc}, {skin_desc} on the visible skin areas, "
        "highly detailed fingers, realistic skin shading, clear glassware, sharp focus, high quality"
    )

def build_negative(glove_desc):
    """
    Add explicit negatives for the opposite of what we want + common artifacts.
    """
    negs = [
        "low quality", "blurry", "distorted hands", "extra fingers", "warped glass",
        "static video", "mutated hand", "mangled fingers"
    ]
    if "no gloves" in glove_desc:
        negs += ["gloves", "latex gloves", "nitrile gloves"]
    else:
        negs += ["bare hands", "no gloves"]
    return ", ".join(negs)

# ===========================
# ========= MAIN =============
# ===========================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"{VIDEO_PATH} not found.")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading AnimateLCM pipeline...")
    adapter = MotionAdapter.from_pretrained(ADAPTER_REPO, torch_dtype=dtype)
    pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(
        BASE_MODEL, motion_adapter=adapter, torch_dtype=dtype
    ).to(device)

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
    pipe.load_lora_weights(ADAPTER_REPO, weight_name=LCM_LORA_W, adapter_name="lcm-lora")
    pipe.set_adapters(["lcm-lora"], [LCM_SCALE])

    pipe.enable_vae_slicing()
    if ENABLE_CPU_OFFLOAD and device == "cuda":
        pipe.enable_model_cpu_offload()

    # Load frames once
    frames = load_video_bgr_safe(
        VIDEO_PATH, max_frames=NUM_FRAMES, size=RESOLUTION, force_bgr=FORCE_BGR_SWAP
    )
    base = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

    for gv in GLOVE_OPTIONS:
        for sk in SKIN_OPTIONS:
            prompt = build_prompt(gv, sk)
            negative = build_negative(gv)
            seed = random.randint(0, 999999) if FIXED_SEED < 0 else FIXED_SEED

            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            out = pipe(
                video=frames,
                prompt=prompt,
                negative_prompt=negative,
                strength=STRENGTH,
                guidance_scale=GUIDANCE,
                num_inference_steps=STEPS,
                generator=torch.Generator(device).manual_seed(seed),
            )
            t1 = time.time()

            name = f"{base}_{'nogloves' if 'no gloves' in gv else 'gloves'}_{sk.replace(' ','-')}.gif"
            gif_path = os.path.join(OUT_DIR, name)
            export_to_gif(out.frames[0], gif_path, fps=GIF_FPS)

            vram = torch.cuda.max_memory_allocated()/(1024**2) if device == "cuda" else 0
            print(f"Saved {gif_path} | {t1-t0:.1f}s | ~{vram:.0f}MB VRAM")

if __name__ == "__main__":
    main()
