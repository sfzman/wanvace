"""Application configuration helpers."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = "./outputs"
OUTPUT_DIR_ENV_KEYS = ("WANVACE_OUTPUT_DIR", "OUTPUT_DIR", "SAVE_FOLDER_PATH")
_ENV_LOADED = False


def load_env_file(env_path: str | Path | None = None):
    """Load simple KEY=VALUE pairs from .env without requiring python-dotenv."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    path = Path(env_path) if env_path else REPO_ROOT / ".env"
    if not path.exists():
        _ENV_LOADED = True
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        os.environ.setdefault(key, value)

    _ENV_LOADED = True


def get_output_dir() -> str:
    """Return the output directory configured in .env, falling back to ./outputs."""
    load_env_file()
    for key in OUTPUT_DIR_ENV_KEYS:
        value = os.getenv(key)
        if value:
            return value
    return DEFAULT_OUTPUT_DIR


def get_ltx2_root() -> str:
    """Return LTX-2 repo root path from env, with a local default fallback."""
    load_env_file()
    return os.getenv("LTX2_ROOT", "/home/arkstone/workspace/LTX-2")


def get_ltx2_ti2vid_hq_config() -> dict:
    """Return TI2Vid-HQ pipeline configuration from environment variables."""
    load_env_file()
    streaming_prefetch_count = os.getenv("LTX2_STREAMING_PREFETCH_COUNT", "").strip()
    max_batch_size = os.getenv("LTX2_MAX_BATCH_SIZE", "").strip()
    return {
        "checkpoint_path": os.getenv("LTX2_CHECKPOINT_PATH"),
        "distilled_lora_path": os.getenv("LTX2_DISTILLED_LORA_PATH"),
        "spatial_upsampler_path": os.getenv("LTX2_SPATIAL_UPSAMPLER_PATH"),
        "gemma_root": os.getenv("LTX2_GEMMA_ROOT"),
        "distilled_lora_strength_stage_1": float(os.getenv("LTX2_DISTILLED_LORA_STRENGTH_STAGE_1", "0.25")),
        "distilled_lora_strength_stage_2": float(os.getenv("LTX2_DISTILLED_LORA_STRENGTH_STAGE_2", "0.5")),
        "image_strength": float(os.getenv("LTX2_IMAGE_STRENGTH", "1.0")),
        "image_crf": int(os.getenv("LTX2_IMAGE_CRF", "33")),
        "torch_compile": os.getenv("LTX2_TORCH_COMPILE", "").lower() in {"1", "true", "yes", "on"},
        "quantization": os.getenv("LTX2_QUANTIZATION", "").strip().lower(),
        "streaming_prefetch_count": int(streaming_prefetch_count) if streaming_prefetch_count else None,
        "max_batch_size": int(max_batch_size) if max_batch_size else 1,
    }
