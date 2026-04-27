"""LTX2 阶段 0：依赖探针与接入模式判定。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib
import importlib.metadata as importlib_metadata
from pathlib import Path
import sys
from typing import Any


LTX2_MIN_TRANSFORMERS = (4, 52, 0)
LTX2_MIN_TORCH = (2, 7, 0)
LTX2_MAX_TORCH_EXCLUSIVE = (2, 8, 0)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LTX2_ROOT = Path("/home/arkstone/workspace/LTX-2")


@dataclass
class VersionCheck:
    name: str
    installed: str | None
    required: str
    ok: bool
    note: str = ""


@dataclass
class ProbeResult:
    checked_at: str
    ltx2_root: str
    mode: str
    checks: list[VersionCheck]
    wan_transformers_constraint: str | None
    has_transformers_conflict: bool
    import_smoke_ok: bool
    import_smoke_error: str | None
    recommendation: str


def _parse_version_tuple(raw: str | None) -> tuple[int, int, int] | None:
    if not raw:
        return None

    pieces: list[int] = []
    for part in raw.split("."):
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits == "":
            break
        pieces.append(int(digits))
        if len(pieces) == 3:
            break

    if not pieces:
        return None
    while len(pieces) < 3:
        pieces.append(0)
    return tuple(pieces[:3])


def _version_ge(installed: str | None, minimum: tuple[int, int, int]) -> bool:
    parsed = _parse_version_tuple(installed)
    return parsed is not None and parsed >= minimum


def _version_lt(installed: str | None, maximum: tuple[int, int, int]) -> bool:
    parsed = _parse_version_tuple(installed)
    return parsed is not None and parsed < maximum


def _read_installed_version(dist_name: str) -> str | None:
    try:
        return importlib_metadata.version(dist_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _extract_transformers_constraint(requirements_path: Path) -> str | None:
    if not requirements_path.exists():
        return None

    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("transformers"):
            return stripped
    return None


def _wan_has_transformers_conflict(requirement_line: str | None) -> bool:
    """检查 wanvace 的 transformers 上限是否低于 LTX2 要求（>=4.52）。"""
    if not requirement_line:
        return False

    # 示例：transformers>=4.49.0,<=4.51.3
    chunks = [part.strip() for part in requirement_line.split(",")]
    upper_bound = None
    for chunk in chunks:
        if "<=" in chunk:
            upper_bound = chunk.split("<=", 1)[1].strip()
            break
        if "<" in chunk and "<=" not in chunk:
            upper_bound = chunk.split("<", 1)[1].strip()
            break

    if upper_bound is None:
        return False

    upper_tuple = _parse_version_tuple(upper_bound)
    if upper_tuple is None:
        return False
    return upper_tuple < LTX2_MIN_TRANSFORMERS


def _append_ltx2_paths(ltx2_root: Path) -> list[str]:
    inserted: list[str] = []
    candidates = [
        ltx2_root / "packages" / "ltx-core" / "src",
        ltx2_root / "packages" / "ltx-pipelines" / "src",
    ]
    for path in candidates:
        path_str = str(path.resolve())
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)
            inserted.append(path_str)
    return inserted


def run_probe(ltx2_root: str | Path | None = None, import_smoke: bool = True) -> dict[str, Any]:
    root = Path(ltx2_root) if ltx2_root else DEFAULT_LTX2_ROOT

    torch_ver = _read_installed_version("torch")
    transformers_ver = _read_installed_version("transformers")

    checks = [
        VersionCheck(
            name="torch",
            installed=torch_ver,
            required=">=2.7,<2.8",
            ok=_version_ge(torch_ver, LTX2_MIN_TORCH) and _version_lt(torch_ver, LTX2_MAX_TORCH_EXCLUSIVE),
            note="LTX-2 官方 ltx-core 当前声明 torch~=2.7",
        ),
        VersionCheck(
            name="transformers",
            installed=transformers_ver,
            required=">=4.52",
            ok=_version_ge(transformers_ver, LTX2_MIN_TRANSFORMERS),
            note="LTX-2 官方 ltx-core 当前声明 transformers>=4.52",
        ),
    ]

    req_line = _extract_transformers_constraint(REPO_ROOT / "requirements.txt")
    has_conflict = _wan_has_transformers_conflict(req_line)

    import_ok = False
    import_error: str | None = None

    if import_smoke:
        try:
            _append_ltx2_paths(root)
            importlib.import_module("ltx_pipelines.ti2vid_two_stages_hq")
            importlib.import_module("ltx_pipelines.a2vid_two_stage")
            import_ok = True
        except Exception as exc:  # noqa: BLE001
            import_ok = False
            import_error = f"{type(exc).__name__}: {exc}"

    checks_ok = all(check.ok for check in checks)
    if checks_ok and import_ok and not has_conflict:
        mode = "inprocess"
        recommendation = "当前环境满足 LTX2 直接集成条件，可进入阶段 1（同进程后端接入）。"
    else:
        mode = "subprocess_isolated"
        recommendation = "建议使用独立 Python 环境通过子进程调用 LTX2，避免依赖冲突。"

    result = ProbeResult(
        checked_at=datetime.now(timezone.utc).isoformat(),
        ltx2_root=str(root.resolve()),
        mode=mode,
        checks=checks,
        wan_transformers_constraint=req_line,
        has_transformers_conflict=has_conflict,
        import_smoke_ok=import_ok,
        import_smoke_error=import_error,
        recommendation=recommendation,
    )
    return asdict(result)
