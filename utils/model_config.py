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
    # 以下仅仅支持首帧图片，不支持尾帧图片
    "Wan-AI/Wan2.1-I2V-14B-480P",
    "Wan-AI/Wan2.2-I2V-A14B",
]

ANIMATE_MODELS = [
    "Wan-AI/Wan2.2-Animate-14B"
]

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

