"""
模型配置模块
包含模型列表、宽高比配置等
"""

# 定义不同模式对应的模型列表
VACE_MODELS = [
    "PAI/Wan2.2-VACE-Fun-A14B",
    "Wan-AI/Wan2.1-VACE-14B",
    "Wan-AI/Wan2.1-VACE-1.3B"
]

INP_MODELS = [
    "PAI/Wan2.2-Fun-A14B-InP",
    "PAI/Wan2.1-Fun-14B-InP",
    "PAI/Wan2.1-Fun-V1.1-1.3B-InP",
    "Lightricks/LTX-2.3-I2AV-TwoStage",
    # 以下仅仅支持首帧图片，不支持尾帧图片
    "Wan-AI/Wan2.1-I2V-14B-480P",
    "Wan-AI/Wan2.2-I2V-A14B",
    "AnisoraV3.2",
    "AnisoraV3.1",
]

ANIMATE_MODELS = [
    "Wan-AI/Wan2.2-Animate-14B"
]

ANISORA_MODELS = [
    "AnisoraV3.2",
    "AnisoraV3.1",
]

LTX_TWO_STAGE_MODEL = "Lightricks/LTX-2.3-I2AV-TwoStage"
WAN_DEFAULT_CFG_SCALE = 5.0
LTX_DEFAULT_FPS = 24
LTX_DEFAULT_CFG_SCALE = 3.0
LTX_DEFAULT_INFERENCE_STEPS = 30
LTX_DURATION_FRAME_MAP = {
    "6秒": 145,
    "8秒": 193,
    "10秒": 241,
}
LTX_RESOLUTION_PRESETS = {
    "横屏 1920×1080": (1920, 1080),
    "竖屏 1080×1920": (1080, 1920),
}

MEMORY_MODE_BALANCED = "均衡模式（推荐）"
MEMORY_MODE_EXTREME = "极限省显存"
MEMORY_MODE_CHOICES = [
    MEMORY_MODE_BALANCED,
    MEMORY_MODE_EXTREME,
]
MEMORY_MODE_INFO = {
    MEMORY_MODE_BALANCED: "速度和显存更均衡，默认使用约 92% 的总显存预算。",
    MEMORY_MODE_EXTREME: "使用更激进的 offload，显存更省，但明显更慢。",
}
DEFAULT_MEMORY_MODE = MEMORY_MODE_BALANCED

ASPECT_RATIOS_14b = {
    "1:1":  (960, 960),
    "4:3":  (1104, 832),
    "3:4":  (832, 1104),
    "3:2":  (1152, 768),
    "2:3":  (768, 1152),
    "16:9": (1280, 720),
    "16:9_low": (832, 480),
    "9:16": (720, 1280),
    "9:16_low": (480, 832),
    "21:9": (1472, 624),
    "9:21": (624, 1472),
    "4:5":  (864, 1072),
    "5:4":  (1072, 864),
}


def get_models_by_mode(mode: str) -> list[str]:
    """根据输入模式返回对应的模型列表"""
    if mode == "vace":
        return VACE_MODELS
    elif mode == "inp":
        return INP_MODELS
    elif mode == "animate":
        return ANIMATE_MODELS
    else:
        return VACE_MODELS  # 默认返回VACE模型


def is_ltx_model(model_id: str | None) -> bool:
    return model_id == LTX_TWO_STAGE_MODEL


def is_animate_model(model_id: str | None) -> bool:
    return model_id in ANIMATE_MODELS


def is_anisora_model(model_id: str | None) -> bool:
    return model_id in ANISORA_MODELS


def get_default_cfg_scale(model_id: str | None) -> float:
    """返回不同模型更稳妥的 CFG 默认值。"""
    if is_ltx_model(model_id):
        return LTX_DEFAULT_CFG_SCALE
    if is_anisora_model(model_id):
        return 1.0
    if is_animate_model(model_id):
        return 1.0
    return WAN_DEFAULT_CFG_SCALE


def get_ltx_duration_label(num_frames) -> str:
    try:
        target_frames = int(num_frames)
    except (TypeError, ValueError):
        return "6秒"
    return min(
        LTX_DURATION_FRAME_MAP,
        key=lambda label: abs(LTX_DURATION_FRAME_MAP[label] - target_frames),
    )


def get_ltx_resolution_label(width, height) -> str:
    try:
        width = int(width)
        height = int(height)
    except (TypeError, ValueError):
        return "横屏 1920×1080"
    return "竖屏 1080×1920" if height > width else "横屏 1920×1080"


def normalize_ltx_generation_params(model_id, fps, width, height, num_frames, tiled=None):
    """将 LTX 参数收敛到官方推荐的固定档位。"""
    if not is_ltx_model(model_id):
        return {
            "fps": int(fps) if fps is not None else 16,
            "width": int(width) if width is not None else 832,
            "height": int(height) if height is not None else 480,
            "num_frames": int(num_frames) if num_frames is not None else 81,
            "tiled": bool(tiled),
        }

    duration_label = get_ltx_duration_label(num_frames)
    resolution_label = get_ltx_resolution_label(width, height)
    normalized_width, normalized_height = LTX_RESOLUTION_PRESETS[resolution_label]
    return {
        "fps": LTX_DEFAULT_FPS,
        "width": normalized_width,
        "height": normalized_height,
        "num_frames": LTX_DURATION_FRAME_MAP[duration_label],
        "tiled": True,
    }
