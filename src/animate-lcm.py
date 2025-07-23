#!/usr/bin/env python3
"""
AnimateLCM video-to-video generator with pluggable augmentations.

Features you can toggle via CLI flags:
  - BGR->RGB conversion
  - Frame trim/pad, resize
  - Horizontal flip to swap hand side
  - Prompt components: swirl speed, direction, hand side, glove presence, skin tone, lighting, background, glassware
  - "Trash" artifacts keywords
  - Random seeds per run
  - Strength / guidance / steps per variant (optional maps)
  - Motion LoRAs blending (zoom/pan, etc.)
  - CSV logging of timings/VRAM

Usage examples:
  python animatelcm_aug_pipeline.py --videos 1.mp4 2.mp4 \
      --include_gloves --include_skin --include_speed --include_direction \
      --hand_sides right left --glove_opts "wearing blue nitrile gloves" "no gloves" \
      --skin_tones light tan dark

  # Minimal: only change gloves
  python animatelcm_aug_pipeline.py --videos 1.mp4 --include_gloves

  # Add motion LoRA
  python animatelcm_aug_pipeline.py --videos 1.mp4 --motion_loras guoyww/animatediff-motion-lora-zoom-out:0.6

  # Log to CSV
  python animatelcm_aug_pipeline.py --log_csv runs.csv
"""

import os, time, random, argparse, csv
import numpy as np
import torch, imageio
from PIL import Image, ImageOps
from diffusers import (
    AnimateDiffVideoToVideoPipeline, LCMScheduler, MotionAdapter
)
from diffusers.utils import export_to_gif

# ------------------ helpers ------------------ #
def load_video(path, max_frames=16, size=512, force_bgr=False):
    """Read video, convert to RGB, resize, pad/trim to max_frames.
    If force_bgr=True, always swap channels.
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
            # quick heuristic to choose swap or not only once
            rgb_guess = Image.fromarray(arr).convert("RGB")
            bgr_swap = Image.fromarray(arr[..., ::-1]).convert("RGB")
            use_bgr_swap = np.mean(np.array(bgr_swap)[:, :, 0]) > np.mean(np.array(rgb_guess)[:, :, 0])
            arr = arr[..., ::-1] if use_bgr_swap else arr
        else:
            arr = arr[..., ::-1] if use_bgr_swap else arr
        img = Image.fromarray(arr).convert("RGB").resize((size, size), Image.BICUBIC)
        frames.append(img)
    reader.close()
    if not frames:
        raise ValueError(f"{path} had 0 frames")
    while len(frames) < max_frames:
        frames.append(frames[-1].copy())
    return frames

def flip_frames(frames):
    return [ImageOps.mirror(f) for f in frames]

# Prompt assembly
def build_prompt(args, vars_dict):
    parts = ["chemistry titration experiment,"]

    # Hand side
    if args.include_hand and vars_dict.get("hand_side"):
        parts.append(f"{vars_dict['hand_side']} hand operating the stopcock,")

    # Swirl speed
    if args.include_speed and vars_dict.get("speed"):
        parts.append(f"{vars_dict['speed']} swirling,")

    # Direction
    if args.include_direction and vars_dict.get("direction"):
        parts.append(f"in {vars_dict['direction']} direction,")

    # Gloves
    if args.include_gloves and vars_dict.get("gloves"):
        parts.append(f"{vars_dict['gloves']},")

    # Skin tone
    if args.include_skin and vars_dict.get("skin"):
        parts.append(f"{vars_dict['skin']} skin,")

    # Lighting
    if args.include_lighting and vars_dict.get("lighting"):
        parts.append(f"{vars_dict['lighting']},")

    # Background
    if args.include_background and vars_dict.get("background"):
        parts.append(f"{vars_dict['background']},")

    # Glassware
    if args.include_glass and vars_dict.get("glass"):
        parts.append(f"{vars_dict['glass']},")

    # Trash artifacts
    if args.include_trash and vars_dict.get("trash"):
        parts.append(f"{vars_dict['trash']},")

    parts.append("realistic, high quality")
    return " ".join(parts)

# CSV logger
def log_row(csv_path, fieldnames, row):
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)

# ------------------ main ------------------ #
def main():
    ap = argparse.ArgumentParser()
    # videos & output
    ap.add_argument("--videos", nargs="+", default=["1.mp4"], help="List of source videos")
    ap.add_argument("--out_dir", default="out_variants")
    # pipeline params
    ap.add_argument("--base_model", default="emilianJR/epiCRealism")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--guidance", type=float, default=1.9)
    ap.add_argument("--strength", type=float, default=0.7)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--force_bgr", action="store_true")
    ap.add_argument("--seed", type=int, default=-1, help="fixed seed; -1 random each run")
    # motion loras
    ap.add_argument("--motion_loras", nargs="*", default=[],
                    help="Format: repo_or_path[:weight]. e.g. guoyww/animatediff-motion-lora-zoom-out:0.6")
    # toggles
    ap.add_argument("--include_speed", action="store_true")
    ap.add_argument("--include_direction", action="store_true")
    ap.add_argument("--include_hand", action="store_true")
    ap.add_argument("--include_gloves", action="store_true")
    ap.add_argument("--include_skin", action="store_true")
    ap.add_argument("--include_lighting", action="store_true")
    ap.add_argument("--include_background", action="store_true")
    ap.add_argument("--include_glass", action="store_true")
    ap.add_argument("--include_trash", action="store_true")

    # option lists
    ap.add_argument("--speed_opts", nargs="*", default=[
        "no swirling, flask held still",
        "slow, gentle circular",
        "rapid, vigorous",
        "extremely fast, frantic with motion blur"
    ])
    ap.add_argument("--direction_opts", nargs="*", default=["clockwise", "counterclockwise"])
    ap.add_argument("--hand_sides", nargs="*", default=["right", "left"])
    ap.add_argument("--glove_opts", nargs="*", default=["wearing blue nitrile gloves", "no gloves"])
    ap.add_argument("--skin_tones", nargs="*", default=["light", "tan", "dark"])
    ap.add_argument("--lighting_opts", nargs="*", default=[
        "bright fluorescent lighting",
        "warm tungsten light",
        "dim lab lighting"
    ])
    ap.add_argument("--background_opts", nargs="*", default=[
        "sterile white lab bench",
        "cluttered lab bench",
        "wooden table"
    ])
    ap.add_argument("--glass_opts", nargs="*", default=[
        "clear conical flask and burette",
        "250ml Erlenmeyer flask and burette",
        "100ml conical flask"
    ])
    ap.add_argument("--trash_opts", nargs="*", default=[
        "oversaturated colors",
        "weird reflections",
        "mild motion blur"
    ])

    # logging
    ap.add_argument("--log_csv", default="", help="CSV file to log runs")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    print("Loading AnimateLCM...")
    adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=dtype)
    pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(
        args.base_model, motion_adapter=adapter, torch_dtype=dtype
    ).to(device)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
    pipe.load_lora_weights("wangfuyun/AnimateLCM",
                           weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                           adapter_name="lcm-lora")
    adapter_names = ["lcm-lora"]
    adapter_scales = [0.8]

    # extra motion loras
    for spec in args.motion_loras:
        if ":" in spec:
            repo, w = spec.split(":", 1)
            w = float(w)
        else:
            repo, w = spec, 0.8
        name = os.path.basename(repo)
        pipe.load_lora_weights(repo, adapter_name=name)
        adapter_names.append(name)
        adapter_scales.append(w)

    pipe.set_adapters(adapter_names, adapter_scales)
    pipe.enable_vae_slicing()
    if args.offload and device == "cuda":
        pipe.enable_model_cpu_offload()

    neg = "low quality, distorted hands, warped glass"

    # Build variants WITHOUT cross combinations (one factor varied at a time)
    def choices(lst, enable):
        return lst if enable else [None]

    speed_list = choices(args.speed_opts, args.include_speed)
    dir_list   = choices(args.direction_opts, args.include_direction)
    hand_list  = choices(args.hand_sides, args.include_hand)
    glove_list = choices(args.glove_opts, args.include_gloves)
    skin_list  = choices(args.skin_tones, args.include_skin)
    light_list = choices(args.lighting_opts, args.include_lighting)
    bg_list    = choices(args.background_opts, args.include_background)
    glass_list = choices(args.glass_opts, args.include_glass)
    trash_list = choices(args.trash_opts, args.include_trash)

    # baseline (no aug)
    baseline = dict(speed=None, direction=None, hand_side=None, gloves=None, skin=None,
                    lighting=None, background=None, glass=None, trash=None)
    variants = [baseline]

    def add_factor(key, values):
        nonlocal variants
        if values == [None]:
            return
        new = []
        for v in values:
            d = baseline.copy()
            d[key] = v
            new.append(d)
        variants += new

    add_factor('speed', speed_list)
    add_factor('direction', dir_list)
    add_factor('hand_side', hand_list)
    add_factor('gloves', glove_list)
    add_factor('skin', skin_list)
    add_factor('lighting', light_list)
    add_factor('background', bg_list)
    add_factor('glass', glass_list)
    add_factor('trash', trash_list)

    print(f"Total variant combos (no cross): {len(variants)}")

    # CSV header
    if args.log_csv:
        header = ["src_video", "outfile", "prompt", "steps", "guidance", "strength",
                  "time_s", "vram_MB", "seed"]

    for vid in args.videos:
        if not os.path.exists(vid):
            print(f"WARNING: {vid} not found, skipping.")
            continue
        base = os.path.splitext(os.path.basename(vid))[0]
        original = load_video(vid, max_frames=args.frames, size=args.res, force_bgr=args.force_bgr)

        for var in variants:
            # choose frames/flip based on hand side
            frames = flip_frames(original) if var.get("hand_side") == "left" else original

            prompt = build_prompt(args, var)
            seed = args.seed if args.seed >= 0 else random.randint(0, 999999)

            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            out = pipe(
                video=frames,
                prompt=prompt,
                negative_prompt=neg,
                strength=args.strength,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                generator=torch.Generator(device).manual_seed(seed),
            )
            t1 = time.time()

            # filename pieces
            labels = []
            for k in ["speed", "direction", "hand_side", "gloves", "skin", "lighting", "background", "glass", "trash"]:
                vlabel = var.get(k)
                if vlabel:
                    short = k[:3]  # short key
                    # compress spaces
                    vlabel = vlabel.replace(" ", "-")[:30]
                    labels.append(f"{short}-{vlabel}")
            outfile = f"{base}_" + ("_".join(labels) if labels else "base") + ".gif"
            gif_path = os.path.join(args.out_dir, outfile)
            export_to_gif(out.frames[0], gif_path, fps=args.fps)

            vram = torch.cuda.max_memory_allocated()/(1024**2) if device == "cuda" else 0
            print(f"Saved {gif_path} | {t1-t0:.1f}s | ~{vram:.0f}MB VRAM")

            if args.log_csv:
                row = dict(src_video=vid, outfile=gif_path, prompt=prompt,
                           steps=args.steps, guidance=args.guidance, strength=args.strength,
                           time_s=round(t1-t0,2), vram_MB=round(vram,1), seed=seed)
                log_row(args.log_csv, header, row)

if __name__ == "__main__":
    main()
