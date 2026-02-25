# Repository Guidelines

## Project Structure & Module Organization
`main.py` is the Gradio entrypoint and UI orchestration layer (VACE, InP, Animate, plus preview tab). Core processing lives in `utils/`:
- `utils/video_processor.py`: model init + inference pipeline.
- `utils/task_queue.py`: JSON-backed queue and worker/subprocess lifecycle.
- `utils/video_utils.py`, `utils/img_utils.py`, `utils/vram_utils.py`: media and VRAM helpers.
- `utils/model_config.py`: model lists and aspect-ratio presets.
- `utils/animate/` and `utils/modules/`: preprocess and model internals.

Runtime artifacts are written to `./outputs/` and `./task_queue/` (both ignored by Git).

## Build, Test, and Development Commands
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/modelscope/DiffSynth-Studio.git
pip install -e DiffSynth-Studio
python main.py
```
- Starts Gradio on `0.0.0.0:7861`.
- Use `python -m py_compile main.py utils/*.py` for a quick syntax check before committing.

## Coding Style & Naming Conventions
- Python style: 4-space indentation, PEP 8 layout, `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep functions focused; avoid mixing UI wiring and heavy processing logic.
- Preserve existing UI language conventions (current interface text is Chinese).
- No formatter/linter is enforced in-repo; keep diffs minimal and consistent with surrounding code.

## Testing Guidelines
- There is currently no formal automated test suite or coverage gate.
- Minimum validation for changes:
  1. Launch `python main.py`.
  2. Smoke test the affected mode(s) (VACE/InP/Animate).
  3. Confirm queue status updates and generated files in `./outputs/`.
- If adding tests, prefer `pytest` with files named `test_*.py` under a new `tests/` directory.

## Commit & Pull Request Guidelines
- Recent history follows lightweight prefixes such as `feat:`, `fix:`, `chore:`, and `opt:`.
- Recommended commit format: `<type>: <short summary>` (Chinese or English is acceptable).
- Keep commits scoped to one logical change; do not include generated videos, model weights, or queue/output artifacts.
- PRs should include: purpose, key file changes, manual verification steps, and UI screenshots/GIFs when interface behavior changes.

## Security & Configuration Tips
- Do not commit local model/data folders (`models/`, `data/`) or temporary media.
- Validate output paths before long runs to avoid writing large artifacts to unintended locations.
