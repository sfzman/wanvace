# Repository Guidelines

## Project Structure & Module Organization
`main.py` is the Gradio entrypoint and UI orchestration layer for VACE, InP, Animate, and the preview tab. Core processing lives in `utils/`:
- `utils/video_processor.py`: model initialization and inference flow
- `utils/task_queue.py`: JSON-backed task queue and worker lifecycle
- `utils/video_utils.py`, `utils/img_utils.py`, `utils/vram_utils.py`: media and VRAM helpers
- `utils/model_config.py`: model lists and aspect-ratio presets
- `utils/animate/` and `utils/modules/`: preprocessors and model internals

Runtime artifacts are written to `./outputs/` and `./task_queue/`. Do not commit generated videos, queue files, model weights, or local data directories.

## Build, Test, and Development Commands
Set up and run locally:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/modelscope/DiffSynth-Studio.git
pip install -e DiffSynth-Studio
python main.py
```

`python main.py` launches Gradio on `0.0.0.0:7861`.

Quick syntax validation before submitting changes:

```bash
python -m py_compile main.py utils/*.py
```

## Coding Style & Naming Conventions
Use 4-space indentation and keep Python code PEP 8-aligned. Prefer `snake_case` for functions and variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Keep heavy processing in `utils/` instead of embedding it in Gradio callbacks. Preserve the existing Chinese UI text style when editing labels, help text, or status messages.

## Testing Guidelines
There is no formal automated test suite yet. Minimum validation:
1. Launch `python main.py`.
2. Smoke test the affected mode (`VACE`, `InP`, or `Animate`).
3. Confirm queue status updates and output files under `./outputs/`.

If you add tests, use `pytest` and place them in `tests/` with names like `test_queue.py`.

## Commit & Pull Request Guidelines
Recent history uses lightweight prefixes such as `feat:`, `fix:`, `chore:`, and `opt:`. Keep commits focused and use a short summary, for example `fix: guard invalid preview thumbnails`.

PRs should include the purpose of the change, key files touched, manual verification steps, and screenshots or GIFs for UI changes.

## Security & Configuration Tips
Do not commit `models/`, `data/`, or temporary media. Validate output paths before long runs so large artifacts stay inside intended directories.
