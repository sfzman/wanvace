#!/usr/bin/env python3
"""阶段 0：LTX2 接入可行性探针。

用法：
  python scripts/phase0_ltx2_probe.py
  python scripts/phase0_ltx2_probe.py --ltx2-root /home/arkstone/workspace/LTX-2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.ltx2_phase0 import run_probe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="LTX2 阶段 0 依赖与导入探针")
    parser.add_argument(
        "--ltx2-root",
        type=str,
        default="/home/arkstone/workspace/LTX-2",
        help="LTX-2 仓库根目录路径",
    )
    parser.add_argument(
        "--skip-import-smoke",
        action="store_true",
        help="跳过 ltx_pipelines 导入烟雾测试",
    )
    args = parser.parse_args()

    result = run_probe(ltx2_root=args.ltx2_root, import_smoke=not args.skip_import_smoke)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 阶段 0 期望输出可执行结论，非 inprocess 返回非 0，便于 CI/脚本判断。
    return 0 if result.get("mode") == "inprocess" else 2


if __name__ == "__main__":
    raise SystemExit(main())
