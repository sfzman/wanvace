"""
模型配置模块
包含模型列表、宽高比配置等
"""
from __future__ import annotations

ANISORA_MODEL = "AnisoraV3.2"
LTX2_TI2VID_HQ_MODEL = "LTX2-TI2Vid-HQ"

# 首尾帧模式模型列表
INP_MODELS = [
    ANISORA_MODEL,
    LTX2_TI2VID_HQ_MODEL,
]

MODEL_BACKENDS = {
    ANISORA_MODEL: "anisora",
    LTX2_TI2VID_HQ_MODEL: "ltx2_ti2vid_hq",
}

MODEL_DEFAULTS = {
    ANISORA_MODEL: {
        "fps": 16,
        "num_inference_steps": 8,
        "cfg_scale": 1.0,
        "sigma_shift": 5.0,
        "motion_score": 2.5,
    },
    LTX2_TI2VID_HQ_MODEL: {
        "fps": 24,
        "num_inference_steps": 15,
        "cfg_scale": 3.0,
        "sigma_shift": 5.0,
        "motion_score": 2.5,
    },
}

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


def get_model_backend(model_id: str) -> str | None:
    return MODEL_BACKENDS.get(model_id)


def get_model_defaults(model_id: str) -> dict:
    return MODEL_DEFAULTS.get(model_id, MODEL_DEFAULTS[ANISORA_MODEL]).copy()


def _parse_aspect_ratio(aspect_ratio: str) -> float:
    raw = aspect_ratio.split("_", 1)[0]
    if ":" in raw:
        left, right = raw.split(":", 1)
        try:
            l_val = float(left)
            r_val = float(right)
            if l_val > 0 and r_val > 0:
                return l_val / r_val
        except (TypeError, ValueError):
            pass

    width, height = ASPECT_RATIOS_14b.get(aspect_ratio, (832, 480))
    return float(width) / float(height)


def _nearest_multiple_64(value: float) -> int:
    return max(64, int(round(value / 64.0) * 64))


def _get_ltx2_approx_1080p_size(aspect_ratio: str) -> tuple[int, int]:
    """
    为 LTX2 返回 1080p 档位下、且宽高均为 64 倍数的近似分辨率。
    """
    ratio = _parse_aspect_ratio(aspect_ratio)
    target_area = 1920 * 1088  # LTX2 1080p 档（高度对齐到 64）
    max_side = 1920
    min_side = 512

    best: tuple[int, int] | None = None
    best_score = None

    for width in range(min_side, max_side + 1, 64):
        height_float = width / ratio
        height = _nearest_multiple_64(height_float)
        if height < min_side or height > max_side:
            continue

        ratio_error = abs((width / height) - ratio) / ratio
        area = width * height
        area_error = abs(area - target_area) / target_area
        score = ratio_error * 3.0 + area_error

        if best_score is None or score < best_score:
            best_score = score
            best = (width, height)

    if best is not None:
        return best

    # 回退：以 16:9 档位作为兜底
    return 1920, 1088


def get_dimensions_for_model(aspect_ratio: str, model_id: str) -> tuple[int, int]:
    """
    返回指定模型在当前宽高比下建议使用的 (width, height)。
    """
    if get_model_backend(model_id) == "ltx2_ti2vid_hq":
        return _get_ltx2_approx_1080p_size(aspect_ratio)
    return ASPECT_RATIOS_14b.get(aspect_ratio, (832, 480))


def get_models_by_mode(mode: str) -> list[str]:
    """根据输入模式返回对应的模型列表"""
    if mode == "inp":
        return INP_MODELS
    else:
        return INP_MODELS
