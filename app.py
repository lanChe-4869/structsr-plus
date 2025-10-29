#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gradio app for SeeSR pipeline.
- Load models once at startup (GPU) and reuse them for each inference.
- Sliders for: num_inference_steps, guidance_scale, process_size,
               lamda, lamda_psg, lamda_rade, alpha_ema, eps
- Upload an input image, get one output image.

Env overrides (optional):
  PRETRAINED_MODEL_PATH, SEESR_MODEL_PATH, RAM_FT_PATH
"""

import os
import gc
import copy
from types import SimpleNamespace
from datetime import datetime

import gradio as gr
from PIL import Image

import torch
from accelerate import Accelerator

# ---- Import your own pipeline utilities from test_seesr.py ----
# They encapsulate model construction & prompt building
from test_seesr import (
    load_seesr_pipeline,
    load_tag_model,
    get_validation_prompt,
)
from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix

# ------------------ Global Config ------------------

# Default paths (can override with env vars)
PRETRAINED_MODEL_PATH = os.environ.get(
    "PRETRAINED_MODEL_PATH",
    "preset/models/stable-diffusion-2-base",
)
SEESR_MODEL_PATH = os.environ.get(
    "SEESR_MODEL_PATH",
    "preset/models/seesr",
)
RAM_FT_PATH = os.environ.get(
    "RAM_FT_PATH",
    "preset/models/DAPE.pth",
)

# Output directory (images saved here time-stamped)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "./gradio_outputs"
)

# Accelerator (mixed precision for VRAM saving)
accelerator = Accelerator(mixed_precision="fp16")
DEVICE = accelerator.device

# Some stable defaults (can be changed per-request)
BASE_ARGS = SimpleNamespace(
    # model paths
    pretrained_model_path=PRETRAINED_MODEL_PATH,
    seesr_model_path=SEESR_MODEL_PATH,
    ram_ft_path=RAM_FT_PATH,

    # fixed or rarely changed
    mixed_precision="fp16",
    align_method="adain",         # 'wavelet' | 'adain' | 'nofix'
    start_point="lr",
    seed=None,
    sample_times=1,
    # model-size-related
    upscale=4,
    process_size=512,             # will be overwritten per request
    vae_decoder_tiled_size=224,   # latent side for ~24G
    vae_encoder_tiled_size=1024,  # image side for ~13G
    latent_tiled_size=96,
    latent_tiled_overlap=4,
    conditioning_scale=1.0,
    blending_alpha=1.0,

    # prompts
    prompt="",                    # extra prompt by user
    added_prompt="clean, high-resolution, 8k",
    negative_prompt="dotted, noise, blur, lowres, smooth",

    # classic sampler knobs (overwritten per request)
    num_inference_steps=50,
    guidance_scale=5.5,

    # SeeSR custom knobs (overwritten per request)
    # lamda=0.3,
    lamda_psg=2.0,
    lamda_rade=2.0,
    alpha_ema=0.9,
    eps=0.03,
)

# ------------------ One-time Model Loading ------------------

print("[Init] Loading SeeSR pipeline (this happens once at startup)...")
pipeline = load_seesr_pipeline(
    BASE_ARGS,
    accelerator,
    enable_xformers_memory_efficient_attention=True,
)
print("[Init] Loading RAM tagging model ...")
ram_model = load_tag_model(BASE_ARGS, device=DEVICE)

# Move core modules to device with desired dtype is already handled inside
# load_seesr_pipeline / load_tag_model. Nothing else to do here.

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("[Init] Done. Ready to serve requests.")


# ------------------ Inference Helper ------------------

def _ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

def _resize_flow(validation_image: Image.Image, process_size: int, upscale: int):
    """
    Mimic test_seesr.py resizing logic:
      - If min side < process_size // upscale -> enlarge.
      - Then upscale * 1st stage
      - Then make divisible by 8
    """
    ori_w, ori_h = validation_image.size
    resize_flag = False

    if ori_w < process_size // upscale or ori_h < process_size // upscale:
        scale = (process_size // upscale) / min(ori_w, ori_h)
        tmp_w, tmp_h = int(scale * ori_w), int(scale * ori_h)
        validation_image = validation_image.resize((tmp_w, tmp_h))
        resize_flag = True

    # Always upscale × r
    tmp_w, tmp_h = validation_image.size
    validation_image = validation_image.resize((tmp_w * upscale, tmp_h * upscale))

    # snap to multiples of 8
    snap_w = (validation_image.size[0] // 8) * 8
    snap_h = (validation_image.size[1] // 8) * 8
    validation_image = validation_image.resize((snap_w, snap_h))

    width, height = validation_image.size
    resize_flag = True  # keep consistent with original script

    return validation_image, (ori_w, ori_h), (width, height), resize_flag


# ------------------ Gradio Callback ------------------

def infer(
    input_image: Image.Image,
    user_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    process_size: int,
    # lamda: float,
    lamda_psg: float,
    lamda_rade: float,
    alpha_ema: float,
    eps: float,
    align_method: str,
    seed: int = 66,
):
    """
    Single-image inference using already-loaded pipeline + RAM model.
    Returns: PIL image (and saves to OUTPUT_DIR with timestamp).
    """

    if input_image is None:
        raise gr.Error("请先上传一张图片。")

    # 1) Build per-request args from global base
    args = copy.deepcopy(BASE_ARGS)
    args.prompt = user_prompt or ""
    args.num_inference_steps = int(num_inference_steps)
    args.guidance_scale = float(guidance_scale)
    args.process_size = int(process_size)
    # args.lamda = float(lamda)
    args.lamda_psg = float(lamda_psg)
    args.lamda_rade = float(lamda_rade)
    args.alpha_ema = float(alpha_ema)
    args.eps = float(eps)
    args.align_method = align_method
    args.seed = int(seed) if (seed is not None and seed != -1) else None

    # 2) Prepare image, prompt & tag embeddings
    pil_img = _ensure_rgb(input_image)
    validation_prompt, ram_encoder_hidden_states = get_validation_prompt(
        args, pil_img, ram_model, device=DEVICE
    )
    # Append user prompt & added prompt
    validation_prompt = f"{validation_prompt}{args.added_prompt}"
    negative_prompt = args.negative_prompt

    # 3) Resize flow (same as script)
    rscale = args.upscale
    resized_img, (ori_w, ori_h), (width, height), resize_flag = _resize_flow(
        pil_img, process_size=args.process_size, upscale=rscale
    )

    # 4) Seed & generator
    generator = torch.Generator(device=DEVICE)
    if args.seed is not None:
        generator.manual_seed(args.seed)

    # 5) Inference (no model reloading here)
    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
        image = pipeline(
            validation_prompt, resized_img,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            height=height, width=width,
            guidance_scale=args.guidance_scale,
            negative_prompt=negative_prompt,
            conditioning_scale=args.conditioning_scale,
            start_point=args.start_point,
            ram_encoder_hidden_states=ram_encoder_hidden_states,
            latent_tiled_size=args.latent_tiled_size,
            latent_tiled_overlap=args.latent_tiled_overlap,
            args=args,  # your custom pipeline may read lamda/ema/eps from here
        ).images[0]

    # 6) Optional color alignment (same choices as script)
    if args.align_method == 'wavelet':
        image = wavelet_color_fix(image, resized_img)
    elif args.align_method == 'adain':
        image = adain_color_fix(image, resized_img)
    # elif 'nofix' -> keep image

    # 7) If resized earlier, bring back to ori size * upscale
    if resize_flag:
        image = image.resize((ori_w * rscale, ori_h * rscale))

    # 8) Save to disk & free some cache
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    save_path = os.path.join(OUTPUT_DIR, f"seesr_{ts}.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image.save(save_path)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return image, save_path


# ------------------ Gradio UI ------------------

DESCRIPTION = (
    "# StructSR++ Demo (based SeeSR)\n"
)

with gr.Blocks(title="StructSR++ Demo") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            in_img = gr.Image(label="上传原图（RGB）", type="pil")
            user_prompt = gr.Textbox(label="附加 Prompt（可留空）", value="")

            with gr.Accordion("参数", open=True):
                num_inference_steps = gr.Slider(
                    label="num_inference_steps", minimum=1, maximum=150, step=1, value=50
                )
                guidance_scale = gr.Slider(
                    label="guidance_scale", minimum=0.1, maximum=15.0, step=0.1, value=5.5
                )
                process_size = gr.Slider(
                    label="process_size", minimum=128, maximum=2048, step=32, value=512
                )
                # lamda = gr.Slider(
                #     label="lamda", minimum=0.0, maximum=5.0, step=0.05, value=0.5
                # )
                lamda_psg = gr.Slider(
                    label="lamda_psg", minimum=0.0, maximum=10.0, step=0.1, value=0.0
                )
                lamda_rade = gr.Slider(
                    label="lamda_rade", minimum=0.0, maximum=10.0, step=0.1, value=2.0
                )
                alpha_ema = gr.Slider(
                    label="alpha_ema", minimum=0.0, maximum=1.0, step=0.01, value=0.9
                )
                eps = gr.Slider(
                    label="eps", minimum=0.0, maximum=0.2, step=0.001, value=0.03
                )
                align_method = gr.Radio(
                    label="对齐方式（颜色一致性）",
                    choices=["adain", "wavelet", "nofix"],
                    value="adain"
                )
                seed = gr.Number(
                    label="seed（-1 表示随机）", value=-1, precision=0
                )

            run_btn = gr.Button("🚀 生成", variant="primary")

        with gr.Column(scale=1):
            out_img = gr.Image(label="生成结果", type="pil")
            save_path_box = gr.Textbox(label="保存路径", interactive=False)

    run_btn.click(
        fn=infer,
        inputs=[
            in_img, user_prompt,
            num_inference_steps, guidance_scale, process_size,
            # lamda, 
            lamda_psg, lamda_rade, alpha_ema, eps,
            align_method, seed
        ],
        outputs=[out_img, save_path_box],
        api_name="run"
    )

if __name__ == "__main__":
    # Tip: control GPU via CUDA_VISIBLE_DEVICES env var when launching
    # e.g., CUDA_VISIBLE_DEVICES=1 python app.py
    demo.queue(max_size=32).launch(server_name="0.0.0.0", server_port=7860)
