"""
视频处理模块
包含视频生成、pipeline初始化、模板视频预处理等核心逻辑
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random
import torch
import gc
import shutil
import tempfile
import time
import numpy as np
from PIL import Image

try:
    from diffsynth.utils.data import save_video, VideoData
except ImportError:
    # 兼容旧版本 DiffSynth 在包根目录导出工具函数的方式
    from diffsynth import save_video, VideoData

try:
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
except ImportError:
    # 兼容旧版本模块名
    from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

try:
    from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline
    from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2
except ImportError:
    LTX2AudioVideoPipeline = None
    write_video_audio_ltx2 = None

from utils.video_utils import clean_temp_videos, reencode_video_to_16fps
from utils.vram_utils import clear_vram
from utils.model_config import (
    ANIMATE_MODELS, DEFAULT_MEMORY_MODE, INP_MODELS, MEMORY_MODE_BALANCED,
    MEMORY_MODE_EXTREME, VACE_MODELS, LTX_TWO_STAGE_MODEL, get_default_cfg_scale,
    is_animate_model, is_ltx_model, normalize_ltx_generation_params
)

# 全局变量存储pipeline和模型选择
pipe = None
selected_model = "PAI/Wan2.2-VACE-Fun-A14B"  # 默认选择14B模型
selected_memory_mode = None  # 记录当前pipeline使用的显存模式
selected_vram_limit = None  # 记录当前pipeline使用的显存限制
input_mode = "vace"  # 默认输入模式：vace（深度视频+参考图片）或 inp（首尾帧）
last_used_model = None  # 记录上一次处理时使用的模型，用于判断是否需要清理显存

ANISORA_ROOT = "/home/arkstone/workspace/anisora-models"
ANISORA_V31_DIR = os.path.join(ANISORA_ROOT, "V3.1")
ANISORA_V32_DIR = os.path.join(ANISORA_ROOT, "V3.2")


def set_wan_ditblock_vram_wrapper(use_safe_wrapper=False):
    """按模型切换 Wan DiTBlock 的显存包装器，规避 AniSora V3.2 在 720p 下的黑屏回归。"""
    try:
        from diffsynth.configs.vram_management_module_maps import VRAM_MANAGEMENT_MODULE_MAPS
    except ImportError:
        return

    model_key = "diffsynth.models.wan_video_dit.WanModel"
    block_key = "diffsynth.models.wan_video_dit.DiTBlock"
    wrapper = (
        "diffsynth.core.vram.layers.AutoWrappedModule"
        if use_safe_wrapper
        else "diffsynth.core.vram.layers.AutoWrappedNonRecurseModule"
    )

    if model_key not in VRAM_MANAGEMENT_MODULE_MAPS:
        return
    if VRAM_MANAGEMENT_MODULE_MAPS[model_key].get(block_key) == wrapper:
        return

    VRAM_MANAGEMENT_MODULE_MAPS[model_key][block_key] = wrapper
    print(f"Wan DiTBlock VRAM wrapper已切换为: {wrapper}")


def get_auto_vram_limit():
    """自动计算显存管理阈值：总显存减去约 2GB 缓冲。"""
    total_vram_gb = torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3)
    return max(total_vram_gb - 2.0, 0.0)


def get_ltx_vram_limit():
    """LTX 参考测试脚本使用更紧的缓冲，尽量贴近其显存行为。"""
    total_vram_gb = torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3)
    return max(total_vram_gb - 0.5, 0.0)


def normalize_vram_limit(vram_limit):
    """将显存限制标准化为 GB；0 表示最激进的显存管理模式。"""
    if vram_limit is None:
        return None
    try:
        vram_limit = float(vram_limit)
    except (TypeError, ValueError):
        return None
    if vram_limit < 0:
        return None
    return vram_limit


def resolve_memory_profile(memory_mode):
    """将显存模式解析为内部配置；兼容历史任务中的 vram_limit 数值。"""
    if isinstance(memory_mode, (int, float)):
        vram_limit = normalize_vram_limit(memory_mode)
        return {
            "mode_key": "legacy_manual",
            "mode_label": f"手动显存限制 ({vram_limit} GB)",
            "vram_limit": vram_limit,
            "use_disk_offload": False,
        }

    if isinstance(memory_mode, str):
        stripped = memory_mode.strip()
        if stripped:
            try:
                vram_limit = normalize_vram_limit(float(stripped))
            except ValueError:
                pass
            else:
                return {
                    "mode_key": "legacy_manual",
                    "mode_label": f"手动显存限制 ({vram_limit} GB)",
                    "vram_limit": vram_limit,
                    "use_disk_offload": False,
                }
            if stripped == MEMORY_MODE_EXTREME:
                return {
                    "mode_key": MEMORY_MODE_EXTREME,
                    "mode_label": MEMORY_MODE_EXTREME,
                    "vram_limit": get_auto_vram_limit(),
                    "use_disk_offload": True,
                }
            if stripped == MEMORY_MODE_BALANCED:
                return {
                    "mode_key": MEMORY_MODE_BALANCED,
                    "mode_label": MEMORY_MODE_BALANCED,
                    "vram_limit": get_auto_vram_limit(),
                    "use_disk_offload": False,
                }

    return {
        "mode_key": DEFAULT_MEMORY_MODE,
        "mode_label": DEFAULT_MEMORY_MODE,
        "vram_limit": get_auto_vram_limit(),
        "use_disk_offload": False,
    }


def build_model_config(use_disk_offload=False, **kwargs):
    if use_disk_offload:
        return ModelConfig(
            offload_dtype="disk",
            offload_device="disk",
            onload_dtype=torch.bfloat16,
            onload_device="cpu",
            preparing_dtype=torch.bfloat16,
            preparing_device="cuda",
            computation_dtype=torch.bfloat16,
            computation_device="cuda",
            **kwargs,
        )
    return ModelConfig(
        offload_dtype=torch.bfloat16,
        offload_device="cpu",
        onload_dtype=torch.bfloat16,
        onload_device="cpu",
        preparing_dtype=torch.bfloat16,
        preparing_device="cuda",
        computation_dtype=torch.bfloat16,
        computation_device="cuda",
        **kwargs,
    )
def build_ltx_model_config(use_disk_offload=False, **kwargs):
    # LTX 在当前环境下采用 test.py 同类的低显存配置；
    # 即使在“均衡模式”下也保持 CPU onload/offload，避免初始化直接 OOM。
    if hasattr(torch, "float8_e5m2"):
        return ModelConfig(
            offload_dtype=torch.float8_e5m2,
            offload_device="cpu",
            onload_dtype=torch.float8_e5m2,
            onload_device="cpu",
            preparing_dtype=torch.float8_e5m2,
            preparing_device="cuda",
            computation_dtype=torch.bfloat16,
            computation_device="cuda",
            **kwargs,
        )
    return ModelConfig(
        offload_dtype=torch.bfloat16,
        offload_device="cpu",
        onload_dtype=torch.bfloat16,
        onload_device="cpu",
        preparing_dtype=torch.bfloat16,
        preparing_device="cuda",
        computation_dtype=torch.bfloat16,
        computation_device="cuda",
        **kwargs,
    )


def _create_pipeline(model_configs, vram_limit):
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        vram_limit=vram_limit,
    )


def _create_ltx_pipeline(vram_limit, use_disk_offload):
    if LTX2AudioVideoPipeline is None:
        raise RuntimeError("当前 DiffSynth 版本未包含 LTX2AudioVideoPipeline，请先更新 DiffSynth-Studio。")

    return LTX2AudioVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            build_ltx_model_config(
                model_id="google/gemma-3-12b-it-qat-q4_0-unquantized",
                origin_file_pattern="model-*.safetensors",
            ),
            build_ltx_model_config(
                model_id="DiffSynth-Studio/LTX-2.3-Repackage",
                origin_file_pattern="transformer.safetensors",
            ),
            build_ltx_model_config(
                model_id="DiffSynth-Studio/LTX-2.3-Repackage",
                origin_file_pattern="text_encoder_post_modules.safetensors",
            ),
            build_ltx_model_config(
                model_id="DiffSynth-Studio/LTX-2.3-Repackage",
                origin_file_pattern="video_vae_decoder.safetensors",
            ),
            build_ltx_model_config(
                model_id="DiffSynth-Studio/LTX-2.3-Repackage",
                origin_file_pattern="audio_vae_decoder.safetensors",
            ),
            build_ltx_model_config(
                model_id="DiffSynth-Studio/LTX-2.3-Repackage",
                origin_file_pattern="audio_vocoder.safetensors",
            ),
            build_ltx_model_config(
                model_id="DiffSynth-Studio/LTX-2.3-Repackage",
                origin_file_pattern="video_vae_encoder.safetensors",
            ),
            build_ltx_model_config(
                model_id="Lightricks/LTX-2.3",
                origin_file_pattern="ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
            ),
        ],
        tokenizer_config=ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized"),
        stage2_lora_config=ModelConfig(
            model_id="Lightricks/LTX-2.3",
            origin_file_pattern="ltx-2.3-22b-distilled-lora-384.safetensors",
        ),
        vram_limit=get_ltx_vram_limit(),
    )


def analyze_video_frames(video):
    """输出视频帧统计，帮助区分“推理全黑”和“保存失败”."""
    if video is None:
        return {"frame_count": 0, "all_black": False, "mean": None, "min": None, "max": None}

    try:
        frame_count = len(video)
    except TypeError:
        video = list(video)
        frame_count = len(video)

    if frame_count == 0:
        return {"frame_count": 0, "all_black": False, "mean": None, "min": None, "max": None}

    sample_indexes = sorted({0, frame_count // 2, frame_count - 1})
    sample_frames = []
    for idx in sample_indexes:
        frame = np.asarray(video[idx])
        sample_frames.append(frame)

    stacked = np.stack(sample_frames, axis=0)
    min_value = int(stacked.min())
    max_value = int(stacked.max())
    mean_value = float(stacked.mean())
    all_black = max_value == 0
    return {
        "frame_count": frame_count,
        "all_black": all_black,
        "mean": mean_value,
        "min": min_value,
        "max": max_value,
    }


def preprocess_template_video(template_video_path, reference_image_path, width, height, num_frames):
    """预处理模板视频，生成pose和face视频"""
    try:
        from utils.animate.preprocess.process_pipepline import ProcessPipeline

        # 创建临时目录用于存储预处理结果
        temp_dir = tempfile.mkdtemp(prefix="wanvace_preprocess_")
        
        # 构建模型路径
        ckpt_path = "./models"
        pose2d_checkpoint_path = os.path.join(ckpt_path, 'pose2d/vitpose_h_wholebody.onnx')
        det_checkpoint_path = os.path.join(ckpt_path, 'det/yolov10m.onnx')
        
        # 检查模型文件是否存在
        if not os.path.exists(pose2d_checkpoint_path):
            error_msg = f"姿态检测模型文件不存在: {pose2d_checkpoint_path}"
            print(error_msg)
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None, None, error_msg
        
        if not os.path.exists(det_checkpoint_path):
            error_msg = f"目标检测模型文件不存在: {det_checkpoint_path}"
            print(error_msg)
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None, None, error_msg
        
        print(f"开始预处理模板视频...")
        print(f"模板视频: {template_video_path}")
        print(f"参考图片: {reference_image_path}")
        print(f"目标分辨率: {width}x{height}")
        print(f"输出目录: {temp_dir}")
        
        # 创建ProcessPipeline实例
        process_pipeline = ProcessPipeline(
            det_checkpoint_path=det_checkpoint_path,
            pose2d_checkpoint_path=pose2d_checkpoint_path,
            sam_checkpoint_path=None,  # 不使用SAM
            flux_kontext_path=None     # 不使用FLUX
        )
        
        # 调用预处理
        process_pipeline(
            video_path=template_video_path,
            refer_image_path=reference_image_path,
            output_path=temp_dir,
            resolution_area=[width, height],
            fps=30,  # 使用默认FPS
            iterations=3,
            k=7,
            w_len=1,
            h_len=1,
            retarget_flag=True,  # 启用姿态重定向
            use_flux=False,
            replace_flag=False
        )
        
        # 查找生成的pose和face视频
        pose_video_path = os.path.join(temp_dir, "src_pose.mp4")
        face_video_path = os.path.join(temp_dir, "src_face.mp4")
        reference_image_path = os.path.join(temp_dir, "src_ref.png")
        
        if not os.path.exists(pose_video_path):
            error_msg = "预处理未生成pose视频文件"
            print(error_msg)
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None, None, error_msg
        
        if not os.path.exists(face_video_path):
            print("警告: 未找到face视频文件，将只使用pose视频")
            face_video_path = None

        if not os.path.exists(reference_image_path):
            print("警告: 未找到reference image文件，将只使用视频")
            reference_image_path = None
        
        print(f"预处理成功完成")
        print(f"Pose视频: {pose_video_path}")
        print(f"Face视频: {face_video_path}")
        print(f"Reference Image: {reference_image_path}")
        return pose_video_path, face_video_path, reference_image_path, temp_dir
        
    except Exception as e:
        error_msg = f"预处理过程中出现错误: {str(e)}"
        print(error_msg)
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        return None, None, error_msg


def initialize_pipeline(model_id="PAI/Wan2.2-VACE-Fun-A14B", memory_mode=DEFAULT_MEMORY_MODE):
    """初始化WanVideoPipeline"""
    global pipe, selected_memory_mode, selected_model, selected_vram_limit, input_mode
    profile = resolve_memory_profile(memory_mode)
    memory_mode_key = profile["mode_key"]
    memory_mode_label = profile["mode_label"]
    vram_limit = profile["vram_limit"]
    use_disk_offload = profile["use_disk_offload"]
    
    should_reinitialize = (
        pipe is None
        or selected_model != model_id
        or selected_memory_mode != memory_mode_key
        or selected_vram_limit != vram_limit
    )

    if pipe is not None and should_reinitialize:
        print(
            f"重新初始化Pipeline: model {selected_model} -> {model_id}, "
            f"memory_mode {selected_memory_mode} -> {memory_mode_key}, "
            f"vram_limit {selected_vram_limit} -> {vram_limit}"
        )
        del pipe
        pipe = None
        torch.cuda.empty_cache()
        gc.collect()
    
    if should_reinitialize:
        print(f"正在初始化模型: {model_id}")
        selected_model = model_id
        selected_memory_mode = memory_mode_key
        selected_vram_limit = vram_limit
        set_wan_ditblock_vram_wrapper(use_safe_wrapper=(model_id == "AnisoraV3.2"))
        
        # 根据模型类型设置输入模式
        if model_id in INP_MODELS:
            input_mode = "inp"
        elif is_animate_model(model_id):
            input_mode = "animate"
        else:
            input_mode = "vace"
        
        if is_ltx_model(model_id):
            pipe = _create_ltx_pipeline(
                vram_limit=vram_limit,
                use_disk_offload=use_disk_offload,
            )
        elif model_id == "PAI/Wan2.2-VACE-Fun-A14B":
            # 14B VACE模型配置
            pipe = _create_pipeline(
                model_configs=[
                    build_model_config(use_disk_offload=use_disk_offload, model_id="PAI/Wan2.2-VACE-Fun-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors"),
                    build_model_config(use_disk_offload=use_disk_offload, model_id="PAI/Wan2.2-VACE-Fun-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors"),
                    build_model_config(use_disk_offload=use_disk_offload, model_id="PAI/Wan2.2-VACE-Fun-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
                    build_model_config(use_disk_offload=use_disk_offload, model_id="PAI/Wan2.2-VACE-Fun-A14B", origin_file_pattern="Wan2.1_VAE.pth"),
                ],
                vram_limit=vram_limit,
            )
        elif model_id == "Wan-AI/Wan2.1-VACE-1.3B":
            # 1.3B VACE模型配置
            pipe = _create_pipeline(
                model_configs=[
                    build_model_config(use_disk_offload=use_disk_offload, model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
                    build_model_config(use_disk_offload=use_disk_offload, model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
                    build_model_config(use_disk_offload=use_disk_offload, model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="Wan2.1_VAE.pth"),
                ],
                vram_limit=vram_limit,
            )
        elif model_id == "PAI/Wan2.2-Fun-A14B-InP":
            # 14B InP模型配置
            pipe = _create_pipeline(
                model_configs=[
                    build_model_config(use_disk_offload=use_disk_offload, model_id="PAI/Wan2.2-Fun-A14B-InP", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors"),
                    build_model_config(use_disk_offload=use_disk_offload, model_id="PAI/Wan2.2-Fun-A14B-InP", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors"),
                    build_model_config(use_disk_offload=use_disk_offload, model_id="PAI/Wan2.2-Fun-A14B-InP", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
                    build_model_config(use_disk_offload=use_disk_offload, model_id="PAI/Wan2.2-Fun-A14B-InP", origin_file_pattern="Wan2.1_VAE.pth"),
                ],
                vram_limit=vram_limit,
            )
        elif model_id == "PAI/Wan2.1-Fun-V1.1-1.3B-InP":
            # 1.3B InP模型配置
            pipe = _create_pipeline(
                model_configs=[
                    build_model_config(use_disk_offload=use_disk_offload, model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
                    build_model_config(use_disk_offload=use_disk_offload, model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
                    build_model_config(use_disk_offload=use_disk_offload, model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP", origin_file_pattern="Wan2.1_VAE.pth"),
                    build_model_config(use_disk_offload=use_disk_offload, model_id="PAI/Wan2.1-Fun-1.3B-InP", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
                ],
                vram_limit=vram_limit,
            )
        elif model_id == "Wan-AI/Wan2.1-I2V-14B-480P":
            print(f"正在初始化480P I2V模型: {model_id}")
            # 480P I2V模型配置
            pipe = _create_pipeline(
                model_configs=[
                    build_model_config(use_disk_offload=use_disk_offload, model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
                    build_model_config(use_disk_offload=use_disk_offload, model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
                    build_model_config(use_disk_offload=use_disk_offload, model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="Wan2.1_VAE.pth"),
                    build_model_config(use_disk_offload=use_disk_offload, model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
                ],
                vram_limit=vram_limit,
            )
        elif model_id == "Wan-AI/Wan2.2-I2V-A14B":
            print(f"正在初始化Wan2.2 I2V模型: {model_id}")
            # Wan2.2 I2V模型配置
            pipe = _create_pipeline(
                model_configs=[
                    build_model_config(use_disk_offload=use_disk_offload, model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors"),
                    build_model_config(use_disk_offload=use_disk_offload, model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors"),
                    build_model_config(use_disk_offload=use_disk_offload, model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
                    build_model_config(use_disk_offload=use_disk_offload, model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="Wan2.1_VAE.pth"),
                ],
                vram_limit=vram_limit,
            )
        elif model_id == "Wan-AI/Wan2.2-Animate-14B":
            # 14B Animate模型配置（与test.py保持一致）
            pipe = _create_pipeline(
                model_configs=[
                    build_model_config(use_disk_offload=use_disk_offload, model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
                    build_model_config(use_disk_offload=use_disk_offload, model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
                    build_model_config(use_disk_offload=use_disk_offload, model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="Wan2.1_VAE.pth"),
                    build_model_config(use_disk_offload=use_disk_offload, model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
                ],
                vram_limit=vram_limit,
            )
        elif model_id == "AnisoraV3.2":
            # AniSora V3.2 结构对齐 Wan2.2 I2V，优先使用模型目录自带的 T5 / VAE 文件。
            pipe = _create_pipeline(
                model_configs=[
                    build_model_config(use_disk_offload=use_disk_offload, path=[
                        os.path.join(ANISORA_V32_DIR, "high_noise_model", "model_part1.safetensors"),
                        os.path.join(ANISORA_V32_DIR, "high_noise_model", "model_part2.safetensors"),
                    ]),
                    build_model_config(use_disk_offload=use_disk_offload, path=[
                        os.path.join(ANISORA_V32_DIR, "low_noise_model", "model_part1.safetensors"),
                        os.path.join(ANISORA_V32_DIR, "low_noise_model", "model_part2.safetensors"),
                    ]),
                    build_model_config(use_disk_offload=use_disk_offload, path=os.path.join(ANISORA_V32_DIR, "models_t5_umt5-xxl-enc-bf16.pth")),
                    build_model_config(use_disk_offload=use_disk_offload, path=os.path.join(ANISORA_V32_DIR, "Wan2.1_VAE.pth")),
                ],
                vram_limit=vram_limit,
            )
        elif model_id == "AnisoraV3.1":
            # AniSora V3.1 结构更接近 Wan2.1 Fun InP，使用模型目录自带的配套组件。
            pipe = _create_pipeline(
                model_configs=[
                    build_model_config(use_disk_offload=use_disk_offload, path=[
                        os.path.join(ANISORA_V31_DIR, "model_part1.safetensors"),
                        os.path.join(ANISORA_V31_DIR, "model_part2.safetensors"),
                    ]),
                    build_model_config(use_disk_offload=use_disk_offload, path=os.path.join(ANISORA_V31_DIR, "models_t5_umt5-xxl-enc-bf16.pth")),
                    build_model_config(use_disk_offload=use_disk_offload, path=os.path.join(ANISORA_V31_DIR, "Wan2.1_VAE.pth")),
                    build_model_config(use_disk_offload=use_disk_offload, path=os.path.join(ANISORA_V31_DIR, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")),
                ],
                vram_limit=vram_limit,
            )
        
        print(f"Pipeline初始化完成，使用模型: {model_id}")
        print(f"输入模式: {input_mode}")
        print(f"显存模式: {memory_mode_label}")
        print(f"显存限制: {vram_limit if vram_limit is not None else '未限制'}GB")
    return f"Pipeline初始化完成！使用模型: {model_id} (模式: {input_mode})"


def process_video(
    depth_video,
    reference_image,
    prompt,
    negative_prompt,
    seed,
    fps,
    quality,
    height,
    width,
    num_frames,
    num_inference_steps,
    memory_mode,
    model_id,
    first_frame=None,
    last_frame=None,
    tiled=False,
    animate_reference_image=None,
    template_video=None,
    save_folder_path="./outputs",
    cfg_scale=None,
    sigma_shift=5.0
):
    """处理视频生成"""
    global pipe, last_used_model
    try:
        normalized_generation = normalize_ltx_generation_params(
            model_id=model_id,
            fps=fps,
            width=width,
            height=height,
            num_frames=num_frames,
            tiled=tiled,
        )
        fps = normalized_generation["fps"]
        width = normalized_generation["width"]
        height = normalized_generation["height"]
        num_frames = normalized_generation["num_frames"]
        tiled = normalized_generation["tiled"]

        # 根据模型类型判断输入模式
        is_inp_mode = model_id in INP_MODELS
        is_animate_mode = is_animate_model(model_id)
        is_ltx_inp_mode = is_ltx_model(model_id)
        
        if is_inp_mode:
            # InP模式：需要首帧，尾帧可选
            has_first_frame = first_frame is not None
            if not has_first_frame:
                return None, "错误：首尾帧模式需要上传首帧图片"
        elif is_animate_mode:
            # Animate模式：需要参考图片和模板视频
            has_reference_image = animate_reference_image is not None
            has_template_video = template_video is not None and template_video != ""
            if not has_reference_image:
                return None, "错误：Animate模式需要上传参考图片"
            if not has_template_video:
                return None, "错误：Animate模式需要上传模板视频"
        else:
            # VACE模式：需要深度视频或参考图片
            has_depth_video = depth_video is not None and depth_video != ""
            has_reference_image = reference_image is not None
            if not has_depth_video and not has_reference_image:
                return None, "错误：请至少上传深度视频或参考图片中的一种"
        
        if seed < 0:
            seed = random.randint(1, 2**32 - 1)

        if cfg_scale is None:
            cfg_scale = get_default_cfg_scale(model_id)
        
        # 自动初始化pipeline；模型或显存限制变化时会重新初始化
        status = initialize_pipeline(model_id, memory_mode)
        if "完成" not in status:
            return None, f"model_id初始化失败：{status}"
        
        vace_video = None
        vace_reference_image = None
        animate_reference_img = None
        template_video_data = None

        if not prompt:
            prompt = "两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。"
        
        if not negative_prompt:
            negative_prompt = "过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

        if is_inp_mode:
            # InP模式：处理首尾帧
            if has_first_frame:
                print(f"调试信息：首帧图片 = {first_frame}, 类型 = {type(first_frame)}")
                if isinstance(first_frame, str):
                    first_frame_img = Image.open(first_frame).resize((width, height)).convert("RGB")
                else:
                    first_frame_img = first_frame.resize((width, height)).convert("RGB")
                
                # 如果有尾帧，也处理
                last_frame_img = None
                if last_frame is not None:
                    print(f"调试信息：尾帧图片 = {last_frame}, 类型 = {type(last_frame)}")
                    if isinstance(last_frame, str):
                        last_frame_img = Image.open(last_frame).resize((width, height)).convert("RGB")
                    else:
                        last_frame_img = last_frame.resize((width, height)).convert("RGB")
                
                if is_ltx_inp_mode:
                    input_images = [first_frame_img]
                    input_images_indexes = [0]
                    if last_frame_img is not None:
                        input_images.append(last_frame_img)
                        input_images_indexes.append(num_frames - 1)

                    video, audio = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        seed=seed,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        frame_rate=fps,
                        tiled=tiled,
                        cfg_scale=cfg_scale,
                        num_inference_steps=num_inference_steps,
                        use_two_stage_pipeline=True,
                        clear_lora_before_state_two=True,
                        input_images=input_images,
                        input_images_indexes=input_images_indexes,
                        input_images_strength=1.0,
                    )
                else:
                    # 调用InP模型的pipeline
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
            else:
                return None, "首尾帧模式需要上传首帧图片"
        elif is_animate_mode:
            # Animate模式：处理参考图片和模板视频
            if has_reference_image:
                print(f"调试信息：Animate参考图片 = {animate_reference_image}, 类型 = {type(animate_reference_image)}")
                if isinstance(animate_reference_image, str):
                    animate_reference_img = Image.open(animate_reference_image).resize((width, height)).convert("RGB")
                else:
                    animate_reference_img = animate_reference_image.resize((width, height)).convert("RGB")
            
            if has_template_video:
                # 预处理模板视频
                temp_template_path = template_video
                print(f"调试信息：模板视频路径 = {temp_template_path}, 类型 = {type(temp_template_path)}")
                print(f"目标尺寸: {width}x{height}")
                
                # 保存参考图片到临时文件用于预处理
                temp_ref_path = None
                if isinstance(animate_reference_image, str):
                    temp_ref_path = animate_reference_image
                else:
                    temp_ref_path = tempfile.mktemp(suffix=".png")
                    animate_reference_image.save(temp_ref_path)
                
                # 调用预处理函数
                pose_video_path, face_video_path, reference_image_path, temp_dir = preprocess_template_video(
                    temp_template_path, temp_ref_path, width, height, num_frames
                )
                animate_reference_img = Image.open(reference_image_path).resize((width, height)).convert("RGB")
                
                if pose_video_path is None:
                    return None, f"模板视频预处理失败：{face_video_path}"  # face_video_path此时包含错误信息
                
                try:
                    # 加载预处理后的pose视频
                    pose_video_data = VideoData(pose_video_path, height=height, width=width).raw_data()[:num_frames-4]
                    print(f"成功加载pose视频数据，帧数: {len(pose_video_data)}")
                    
                    # 如果有face视频，也加载
                    face_video_data = None
                    if face_video_path and os.path.exists(face_video_path):
                        face_video_data = VideoData(face_video_path).raw_data()[:num_frames-4]
                        print(f"成功加载face视频数据，帧数: {len(face_video_data)}")
                    
                except Exception as e:
                    print(f"预处理视频加载失败: {e}")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return None, f"预处理视频加载失败：{str(e)}"
            
            # 调用Animate模型的pipeline（与test.py保持一致）
            video = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                input_image=animate_reference_img,
                animate_pose_video=pose_video_data,
                animate_face_video=face_video_data,
                seed=seed,
                tiled=tiled,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                cfg_scale=cfg_scale,
                sigma_shift=sigma_shift,
            )
            
            # 清理临时文件
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            # VACE模式：处理深度视频和参考图片
            if has_depth_video:
                # Gradio Video组件返回的是文件路径字符串
                temp_video_path = depth_video
                print(f"调试信息：深度视频路径 = {temp_video_path}, 类型 = {type(temp_video_path)}")
                print(f"目标尺寸: {width}x{height}")
                # 把目标视频宽高和帧率转换为16fps
                temp_video_path = reencode_video_to_16fps(temp_video_path, num_frames, width, height)
                
                # 创建VideoData时指定目标尺寸，让系统自动调整
                try:
                    vace_video = VideoData(temp_video_path, height=height, width=width)
                    print(f"成功创建VideoData，尺寸: {width}x{height}")
                except Exception as e:
                    print(f"VideoData创建失败: {e}")
                    return None, f"深度视频处理失败：{str(e)}\n请确保视频尺寸与目标尺寸兼容"
            
            if has_reference_image:
                print(f"调试信息：参考图片 = {reference_image}, 类型 = {type(reference_image)}")
                if isinstance(reference_image, str):
                    vace_reference_image = Image.open(reference_image).resize((width, height)).convert("RGB")
                else:
                    vace_reference_image = reference_image.resize((width, height)).convert("RGB")
        
            video = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                vace_video=vace_video,
                vace_reference_image=vace_reference_image,
                seed=seed,
                tiled=tiled,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                cfg_scale=cfg_scale,
                sigma_shift=sigma_shift,
                vace_scale=0.8,
            )
        
        timestamp = int(time.time())
        output_path = f"output_video_{seed}_{timestamp}.mp4"
        video_stats = analyze_video_frames(video)
        print(
            "生成结果统计: "
            f"frames={video_stats['frame_count']}, "
            f"mean={video_stats['mean']}, "
            f"min={video_stats['min']}, "
            f"max={video_stats['max']}, "
            f"all_black={video_stats['all_black']}"
        )
        if is_ltx_inp_mode:
            audio_sample_rate = None
            if audio is not None and hasattr(pipe, "audio_vocoder"):
                audio_sample_rate = getattr(pipe.audio_vocoder, "output_sampling_rate", None)
            write_video_audio_ltx2(
                video=video,
                audio=audio,
                output_path=output_path,
                fps=fps,
                audio_sample_rate=audio_sample_rate,
            )
            if hasattr(pipe, "clear_lora"):
                pipe.clear_lora(verbose=0)
        else:
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
        if video_stats["all_black"]:
            return output_path, f"视频生成完成，但输出帧全黑，已保存为 {output_path}。这通常说明推理阶段异常，不是保存阶段异常。"
        return output_path, f"视频生成成功！已保存为 {output_path}"
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return None, f"生成过程中出现错误：{str(e)}\n\n详细错误信息：\n{error_trace}"
