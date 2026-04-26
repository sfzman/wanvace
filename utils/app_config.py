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
