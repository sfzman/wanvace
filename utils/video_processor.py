"""
视频处理模块
包含首尾帧视频生成和 pipeline 初始化逻辑。
"""
import gc
import random
import time

import torch
from PIL import Image
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

from utils.video_utils import clean_temp_videos
from utils.vram_utils import clear_vram
from utils.model_config import INP_MODELS

DEFAULT_MODEL = "AnisoraV3.2"

# 全局变量存储pipeline和模型选择
pipe: WanVideoPipeline = None
selected_model = DEFAULT_MODEL
last_used_model = None  # 记录上一次处理时使用的模型，用于判断是否需要清理显存


def initialize_pipeline(model_id=DEFAULT_MODEL, vram_limit=6.0):
    """初始化首尾帧视频生成 Pipeline。"""
    global pipe, selected_model

    if model_id not in INP_MODELS:
        return f"不支持的模型：{model_id}。当前仅支持 {', '.join(INP_MODELS)}"

    # 如果模型改变了，需要重新初始化
    if pipe is not None and selected_model != model_id:
        print(f"模型从 {selected_model} 切换到 {model_id}，重新初始化...")
        del pipe
        pipe = None
        torch.cuda.empty_cache()
        gc.collect()

    if pipe is None:
        print(f"正在初始化模型: {model_id}")
        selected_model = model_id

        if model_id == DEFAULT_MODEL:
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
                    ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
                    ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
                    ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
                ],
            )

        # 将GB转换为字节
        num_persistent_param_in_dit = int(vram_limit * 1024**3)
        pipe.enable_vram_management(num_persistent_param_in_dit=num_persistent_param_in_dit)

        print(f"Pipeline初始化完成，使用模型: {model_id}")
        print("输入模式: 首尾帧")
        print(f"显存限制: {vram_limit}GB")
    return f"Pipeline初始化完成！使用模型: {model_id} (模式: 首尾帧)"


def _load_frame_image(frame, width, height, frame_name):
    """加载并缩放 Gradio 传入的 PIL 图片或文件路径。"""
    print(f"调试信息：{frame_name} = {frame}, 类型 = {type(frame)}")
    if isinstance(frame, str):
        return Image.open(frame).resize((width, height)).convert("RGB")
    return frame.resize((width, height)).convert("RGB")


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
    sigma_shift=5.0
):
    """处理首尾帧视频生成。"""
    global pipe, last_used_model
    try:
        if model_id not in INP_MODELS:
            return None, f"错误：不支持的模型 {model_id}，当前仅支持 {', '.join(INP_MODELS)}"
        if first_frame is None:
            return None, "错误：首尾帧模式需要上传首帧图片"

        if seed < 0:
            seed = random.randint(1, 2**32 - 1)

        # 自动初始化pipeline（如果还没有初始化）
        if pipe is None:
            status = initialize_pipeline(model_id, vram_limit)
            if "完成" not in status:
                return None, f"model_id初始化失败：{status}"

        if not prompt:
            prompt = "两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。"

        if not negative_prompt:
            negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

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
            cfg_scale=cfg_scale,
            sigma_shift=sigma_shift,
        )

        timestamp = int(time.time())
        output_path = f"output_video_{seed}_{timestamp}.mp4"
        save_video(video, output_path, fps=fps, quality=quality)

        print("cleaning temp videos...")
        clean_temp_videos()

        # 只有当模型切换时才清理显存
        if last_used_model is not None and last_used_model != model_id:
            print(f"检测到模型切换（{last_used_model} -> {model_id}），清理显存...")
            clear_vram()
        else:
            print("使用相同模型，跳过显存清理")

        # 更新上一次使用的模型
        last_used_model = model_id

        # 不再在此处保存/复制；统一由后台线程在任务成功后剪切
        return output_path, f"视频生成成功！已保存为 {output_path}"

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return None, f"生成过程中出现错误：{str(e)}\n\n详细错误信息：\n{error_trace}"
