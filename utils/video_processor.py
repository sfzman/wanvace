"""
视频处理模块
包含首尾帧视频生成和 pipeline 初始化逻辑。
"""
from __future__ import annotations

import gc
import random
import time

import torch
from PIL import Image
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

from utils.model_config import (
    ANISORA_MODEL,
    INP_MODELS,
    get_model_backend,
)
from utils.video_utils import clean_temp_videos
from utils.ltx2_processor import clear_ltx2_pipeline, initialize_ltx2_pipeline, process_ltx2_video

DEFAULT_MODEL = ANISORA_MODEL

# 兼容既有 clear_vram 逻辑：pipe 仍指向 Anisora pipeline。
pipe: WanVideoPipeline = None
selected_model = DEFAULT_MODEL
last_used_model = None  # 记录上一次处理时使用的模型，用于切换模型时释放旧 backend


def _clear_anisora_pipeline() -> None:
    global pipe, selected_model
    if pipe is not None:
        del pipe
        pipe = None
    selected_model = DEFAULT_MODEL
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def release_all_pipelines() -> None:
    """释放所有后端 pipeline，供显存清理入口统一调用。"""
    global last_used_model
    _clear_anisora_pipeline()
    clear_ltx2_pipeline()
    last_used_model = None


def _initialize_anisora_pipeline(model_id=DEFAULT_MODEL, vram_limit=6.0):
    """初始化 Anisora 首尾帧视频生成 Pipeline。"""
    global pipe, selected_model

    if model_id != ANISORA_MODEL:
        return f"不支持的 Anisora 模型：{model_id}"

    if pipe is None:
        print(f"正在初始化模型: {model_id}")
        selected_model = model_id

        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(path=[
                    "/home/arkstone/workspace/anisora-models/V3.2/high_noise_model/model_part1.safetensors",
                    "/home/arkstone/workspace/anisora-models/V3.2/high_noise_model/model_part2.safetensors",
                ], offload_device="cpu"),
                ModelConfig(path=[
                    "/home/arkstone/workspace/anisora-models/V3.2/low_noise_model/model_part1.safetensors",
                    "/home/arkstone/workspace/anisora-models/V3.2/low_noise_model/model_part2.safetensors",
                ], offload_device="cpu"),
                ModelConfig(
                    model_id="Wan-AI/Wan2.2-Animate-14B",
                    origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                    offload_device="cpu",
                ),
                ModelConfig(
                    model_id="Wan-AI/Wan2.2-Animate-14B",
                    origin_file_pattern="Wan2.1_VAE.pth",
                    offload_device="cpu",
                ),
                ModelConfig(
                    model_id="Wan-AI/Wan2.2-Animate-14B",
                    origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                    offload_device="cpu",
                ),
            ],
        )

        # 将GB转换为字节
        num_persistent_param_in_dit = int(vram_limit * 1024**3)
        pipe.enable_vram_management(num_persistent_param_in_dit=num_persistent_param_in_dit)

        print(f"Pipeline初始化完成，使用模型: {model_id}")
        print("输入模式: 首尾帧")
        print(f"显存限制: {vram_limit}GB")
    return f"Pipeline初始化完成！使用模型: {model_id} (模式: 首尾帧)"


def initialize_pipeline(model_id=DEFAULT_MODEL, vram_limit=6.0):
    """根据模型初始化对应的 pipeline。"""
    backend = get_model_backend(model_id)
    if backend is None:
        return f"不支持的模型：{model_id}。当前支持 {', '.join(INP_MODELS)}"

    if backend == "anisora":
        return _initialize_anisora_pipeline(model_id=model_id, vram_limit=vram_limit)
    if backend == "ltx2_ti2vid_hq":
        return initialize_ltx2_pipeline(model_id=model_id, vram_limit=vram_limit)

    return f"不支持的后端：{backend}"


def _load_frame_image(frame, width, height, frame_name):
    """加载并缩放 Gradio 传入的 PIL 图片或文件路径。"""
    print(f"调试信息：{frame_name} = {frame}, 类型 = {type(frame)}")
    if isinstance(frame, str):
        return Image.open(frame).resize((width, height)).convert("RGB")
    return frame.resize((width, height)).convert("RGB")


def _build_anisora_prompt(base_prompt: str, motion_score: float | None) -> str:
    resolved_motion_score = 2.5 if motion_score is None else float(motion_score)
    resolved_motion_score = max(2.0, min(5.0, resolved_motion_score))
    suffix = (
        f"aesthetic score: 5.0. motion score: {resolved_motion_score:.1f}. "
        "There is no text in the video."
    )
    return f"{base_prompt.strip()} {suffix}".strip()


def _process_anisora_video(
    prompt,
    negative_prompt,
    seed,
    fps,
    quality,
    height,
    width,
    num_frames,
    num_inference_steps,
    vram_limit,
    model_id=DEFAULT_MODEL,
    first_frame=None,
    last_frame=None,
    tiled=False,
    save_folder_path="./outputs",
    cfg_scale=1.0,
    sigma_shift=5.0,
    motion_score=2.5,
):
    global pipe

    if model_id != ANISORA_MODEL:
        return None, f"错误：不支持的模型 {model_id}"
    if first_frame is None:
        return None, "错误：首尾帧模式需要上传首帧图片"

    if seed < 0:
        seed = random.randint(1, 2**32 - 1)

    if pipe is None:
        status = _initialize_anisora_pipeline(model_id, vram_limit)
        if "完成" not in status:
            return None, f"model_id初始化失败：{status}"

    if not prompt:
        prompt = "两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。"
    prompt = _build_anisora_prompt(prompt, motion_score)

    if not negative_prompt:
        negative_prompt = (
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
            "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
            "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        )
    resolved_cfg_scale = float(cfg_scale) if cfg_scale is not None else 1.0
    resolved_sigma_shift = float(sigma_shift) if sigma_shift is not None else 5.0

    first_frame_img = _load_frame_image(first_frame, width, height, "首帧图片")
    last_frame_img = None
    if last_frame is not None:
        last_frame_img = _load_frame_image(last_frame, width, height, "尾帧图片")

    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_image=first_frame_img,
        end_image=last_frame_img,
        seed=seed,
        tiled=tiled,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        cfg_scale=resolved_cfg_scale,
        sigma_shift=resolved_sigma_shift,
    )

    timestamp = int(time.time())
    output_path = f"output_video_{seed}_{timestamp}.mp4"
    del save_folder_path
    save_video(video, output_path, fps=fps, quality=quality)

    print("cleaning temp videos...")
    clean_temp_videos()

    return output_path, f"视频生成成功！已保存为 {output_path}"


def _release_pipeline_for_model(model_id: str | None) -> None:
    if not model_id:
        return
    backend = get_model_backend(model_id)
    if backend == "anisora":
        _clear_anisora_pipeline()
    elif backend == "ltx2_ti2vid_hq":
        clear_ltx2_pipeline()


def process_video(
    prompt,
    negative_prompt,
    seed,
    fps,
    quality,
    height,
    width,
    num_frames,
    num_inference_steps,
    vram_limit,
    model_id=DEFAULT_MODEL,
    first_frame=None,
    last_frame=None,
    tiled=False,
    save_folder_path="./outputs",
    cfg_scale=1.0,
    sigma_shift=5.0,
    motion_score=2.5,
):
    """根据所选模型处理视频生成。"""
    global last_used_model
    try:
        if model_id not in INP_MODELS:
            return None, f"错误：不支持的模型 {model_id}，当前仅支持 {', '.join(INP_MODELS)}"

        if last_used_model is not None and last_used_model != model_id:
            print(f"检测到模型切换（{last_used_model} -> {model_id}），释放上一个 backend pipeline")
            _release_pipeline_for_model(last_used_model)

        backend = get_model_backend(model_id)
        if backend == "anisora":
            output_path, message = _process_anisora_video(
                prompt,
                negative_prompt,
                seed,
                fps,
                quality,
                height,
                width,
                num_frames,
                num_inference_steps,
                vram_limit,
                model_id,
                first_frame,
                last_frame,
                tiled,
                save_folder_path,
                cfg_scale,
                sigma_shift,
                motion_score,
            )
        elif backend == "ltx2_ti2vid_hq":
            output_path, message = process_ltx2_video(
                prompt,
                negative_prompt,
                seed,
                fps,
                quality,
                height,
                width,
                num_frames,
                num_inference_steps,
                vram_limit,
                model_id,
                first_frame,
                last_frame,
                tiled,
                save_folder_path,
                cfg_scale,
                sigma_shift,
            )
        else:
            return None, f"错误：未知模型后端 {backend}"

        if output_path:
            last_used_model = model_id

        return output_path, message

    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        return None, f"生成过程中出现错误：{str(e)}\n\n详细错误信息：\n{error_trace}"
