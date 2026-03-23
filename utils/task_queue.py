"""
任务队列模块
处理后台任务队列的创建、执行和管理

任务在独立的子进程中执行（通过 multiprocessing spawn），好处：
- 子进程拥有独立的 CUDA context，超时后可直接 kill 释放显存
- pipeline 在子进程中持久缓存，同一子进程内多次任务不需重新加载模型
- 主进程保持轻量，不持有任何 GPU 资源
"""
import os
import json
import queue
import threading
import multiprocessing as mp
import uuid
import shutil
import time
from pathlib import Path
from datetime import datetime
from utils.model_config import (
    DEFAULT_MEMORY_MODE,
    LTX_TWO_STAGE_MODEL,
    VACE_MODELS,
    get_default_cfg_scale,
    normalize_ltx_generation_params,
)

# 任务队列配置
TASK_QUEUE_DIR = Path("./task_queue").resolve()
TASK_STATUS_PENDING = "pending"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_DONE = "done"
TASK_STATUS_FAILED = "failed"
MAX_RETRIES = 2
TASK_TIMEOUT = 600  # 10 分钟

_worker_thread = None
_worker_stop_event = threading.Event()

# 子进程状态
_mp_ctx = mp.get_context("spawn")
_worker_proc = None
_task_q = None
_result_q = None
_worker_proc_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 子进程管理
# ---------------------------------------------------------------------------

def _worker_subprocess_fn(task_q, result_q):
    """在子进程中运行。循环接收任务、调用 process_video、回传结果。
    Pipeline 在子进程的全局变量中缓存，多次任务共享同一 pipeline。"""
    import queue as _queue_mod
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    while True:
        try:
            params = task_q.get(timeout=1.0)
        except _queue_mod.Empty:
            continue
        except (EOFError, OSError):
            break  # 队列已关闭

        if params is None:  # 关闭哨兵
            break

        try:
            from utils.video_processor import process_video
            from utils.model_config import VACE_MODELS as _vm
            out_path, msg = process_video(
                params.get("depth_video"),
                params.get("reference_image"),
                params.get("prompt"),
                params.get("negative_prompt"),
                params.get("seed", -1),
                params.get("fps", 16),
                params.get("quality", 8),
                params.get("height", 480),
                params.get("width", 832),
                params.get("num_frames", 81),
                params.get("num_inference_steps", 30),
                params.get("memory_mode", params.get("vram_limit", DEFAULT_MEMORY_MODE)),
                params.get("model_id", _vm[0]),
                params.get("first_frame"),
                params.get("last_frame"),
                params.get("tiled", False),
                params.get("animate_reference_image"),
                params.get("template_video"),
                "",  # 传空以跳过 process_video 中的复制保存
                params.get("cfg_scale", get_default_cfg_scale(params.get("model_id", _vm[0]))),
                params.get("sigma_shift", 5.0),
            )
            result_q.put(("ok", out_path, msg))
        except Exception as e:
            import traceback
            result_q.put(("error", str(e), traceback.format_exc()))


def _start_worker_subprocess():
    """启动（或重新启动）工作子进程。"""
    global _worker_proc, _task_q, _result_q
    kill_worker_subprocess()
    task_q = _mp_ctx.Queue()
    result_q = _mp_ctx.Queue()
    worker_proc = _mp_ctx.Process(
        target=_worker_subprocess_fn,
        args=(task_q, result_q),
        daemon=True,
    )
    worker_proc.start()
    with _worker_proc_lock:
        _task_q = task_q
        _result_q = result_q
        _worker_proc = worker_proc
    print(f"[worker] 工作子进程已启动 (PID: {worker_proc.pid})")


def kill_worker_subprocess():
    """终止工作子进程并清理队列。可被信号处理器安全调用。"""
    global _worker_proc, _task_q, _result_q
    with _worker_proc_lock:
        worker_proc = _worker_proc
        task_q = _task_q
        result_q = _result_q
        _worker_proc = None
        _task_q = None
        _result_q = None

    if worker_proc is not None:
        if worker_proc.is_alive():
            print(f"[worker] 终止工作子进程 (PID: {worker_proc.pid})...")
            worker_proc.terminate()
            worker_proc.join(5)
            if worker_proc.is_alive():
                worker_proc.kill()
                worker_proc.join(2)
    for q in (task_q, result_q):
        if q is not None:
            try:
                q.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# JSON / 文件工具函数（与之前一致）
# ---------------------------------------------------------------------------

def _ensure_task_dirs():
    TASK_QUEUE_DIR.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(file_path: Path, data: dict):
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, file_path)


def _load_json(file_path: Path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _copy_if_exists(src_path: str, dst_dir: Path, new_name: str) -> str:
    if not src_path:
        return None
    try:
        src = Path(src_path)
        if src.exists():
            dst = dst_dir / new_name
            shutil.copy2(str(src), str(dst))
            return str(dst)
    except Exception:
        pass
    return None


def _save_pil_image_if_needed(img_obj, dst_dir: Path, filename: str) -> str:
    if img_obj is None:
        return None
    if isinstance(img_obj, str):
        return _copy_if_exists(img_obj, dst_dir, filename)
    try:
        out_path = dst_dir / filename
        img_obj.save(str(out_path))
        return str(out_path)
    except Exception:
        return None


def _recover_orphan_running_tasks():
    for task_file in TASK_QUEUE_DIR.rglob("task_*.json"):
        data = _load_json(task_file)
        if not data:
            continue
        if data.get("status") == TASK_STATUS_RUNNING:
            data["status"] = TASK_STATUS_PENDING
            _atomic_write_json(task_file, data)


def _iter_pending_tasks():
    tasks = []
    for task_file in TASK_QUEUE_DIR.rglob("task_*.json"):
        data = _load_json(task_file)
        if not data:
            continue
        status = data.get("status")
        retries = int(data.get("retries", 0))
        if status in (TASK_STATUS_PENDING, TASK_STATUS_FAILED) and retries <= MAX_RETRIES:
            tasks.append((task_file, data))
    tasks.sort(key=lambda x: x[1].get("created_at", ""))
    return tasks


# ---------------------------------------------------------------------------
# 主工作线程（在主进程中运行，负责调度任务到子进程）
# ---------------------------------------------------------------------------

def _task_worker_loop():
    _ensure_task_dirs()
    _recover_orphan_running_tasks()

    while not _worker_stop_event.is_set():
        try:
            pending = _iter_pending_tasks()
            if not pending:
                _worker_stop_event.wait(1.0)
                continue

            task_file, task = pending[0]
            task_id = task.get("id")
            print(f"[worker] 准备执行任务: {task_id} -> {task_file}")

            task["status"] = TASK_STATUS_RUNNING
            start_time = datetime.now()
            task["started_at"] = start_time.isoformat()
            _atomic_write_json(task_file, task)

            params = task.get("params", {})
            model_id = params.get("model_id")
            is_ltx_task = model_id == LTX_TWO_STAGE_MODEL

            # 确保子进程存活
            with _worker_proc_lock:
                worker_proc = _worker_proc
            if is_ltx_task and worker_proc is not None and worker_proc.is_alive():
                # LTX 显存占用和碎片更敏感，按 task 隔离子进程以贴近 test.py 的单次执行行为。
                kill_worker_subprocess()
                worker_proc = None
            if worker_proc is None or not worker_proc.is_alive():
                _start_worker_subprocess()

            # 发送任务到子进程
            with _worker_proc_lock:
                task_q = _task_q
                result_q = _result_q
            if task_q is None or result_q is None:
                if _worker_stop_event.is_set():
                    break
                raise RuntimeError("工作子进程队列未初始化")
            task_q.put(params)

            # 轮询等待结果（每秒检查一次超时、子进程状态、停止事件）
            out_path = None
            msg = ""
            result = None
            elapsed = 0.0

            while elapsed < TASK_TIMEOUT and not _worker_stop_event.is_set():
                # 子进程意外退出
                with _worker_proc_lock:
                    worker_proc = _worker_proc
                    result_q = _result_q
                if worker_proc is not None and not worker_proc.is_alive():
                    result = ("error", "工作子进程异常退出", f"Exit code: {worker_proc.exitcode}")
                    break
                if result_q is None:
                    if _worker_stop_event.is_set():
                        break
                    result = ("error", "工作结果队列不可用", "result queue is None")
                    break
                try:
                    result = result_q.get(timeout=1.0)
                    break
                except queue.Empty:
                    elapsed += 1.0

            # --- 处理结果 ---
            if _worker_stop_event.is_set():
                # 主进程正在关闭，任务状态留给下次启动时 recover
                break

            if result is None:
                # 超时
                print(f"[worker] 任务超时: {task_id} (超过 {TASK_TIMEOUT}s)")
                kill_worker_subprocess()
                out_path = None
                msg = f"任务超时（超过 {TASK_TIMEOUT // 60} 分钟），已强制终止子进程"
            elif result[0] == "ok":
                out_path, msg = result[1], result[2]
            else:
                out_path = None
                msg = f"执行异常：{result[1]}\n\n详细错误信息：\n{result[2]}"
                print(f"[worker] 任务异常: {task_id}")
                print(f"[worker] 错误: {result[1]}")
                print(f"[worker] 完整堆栈:\n{result[2]}")

            task["status"] = TASK_STATUS_DONE if out_path else TASK_STATUS_FAILED
            task["finished_at"] = datetime.now().isoformat()

            # 若成功则将视频与任务目录剪切到输出目录
            moved_dir = None
            moved_video = None
            if out_path:
                try:
                    save_folder = params.get("save_folder_path", "./outputs") or "./outputs"
                    save_folder_abs = Path(save_folder).resolve()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    subfolder_name = f"generation_{timestamp}_{task_id}"
                    dest_dir = save_folder_abs / subfolder_name
                    dest_dir.mkdir(parents=True, exist_ok=True)

                    out_src = Path(out_path).resolve()
                    if out_src.exists():
                        moved_video_path = dest_dir / out_src.name
                        shutil.move(str(out_src), str(moved_video_path))
                        moved_video = str(moved_video_path)

                    task_dir = task_file.parent.resolve()
                    if task_dir.exists():
                        dest_task_dir = dest_dir / task_dir.name
                        shutil.move(str(task_dir), str(dest_task_dir))
                        moved_dir = str(dest_task_dir)
                    print(f"[worker] 已剪切到输出目录: 视频={moved_video}, 任务目录={moved_dir}")
                except Exception as move_e:
                    print(f"[worker] 剪切到输出目录失败: {move_e}")
            else:
                print(f"[worker] 任务失败: {task_id}, 错误信息: {msg}")

            if is_ltx_task:
                # 无论成功失败都销毁 LTX 子进程，避免下一个任务复用碎片化 CUDA context。
                kill_worker_subprocess()

            task["result"] = {
                "output_video": out_path,
                "message": msg,
                "moved_video": moved_video,
                "moved_task_dir": moved_dir,
            }

            end_time = datetime.now()
            duration_seconds = (end_time - start_time).total_seconds()
            task["duration_seconds"] = round(duration_seconds, 2)
            print(f"[worker] 任务完成: {task_id}, 状态: {task['status']}, 耗时: {task['duration_seconds']}s")

            task["retries"] = int(task.get("retries", 0)) + (0 if task["status"] == TASK_STATUS_DONE else 1)
            _atomic_write_json(task_file, task)

        except Exception:
            import traceback
            traceback.print_exc()
            _worker_stop_event.wait(1.0)

    # 线程退出时清理子进程
    kill_worker_subprocess()


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------

def start_task_worker():
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        return "任务工作线程已在运行"
    _ensure_task_dirs()
    _worker_stop_event.clear()
    _worker_thread = threading.Thread(target=_task_worker_loop, name="wanvace-task-worker", daemon=True)
    _worker_thread.start()
    return "任务工作线程已启动"


def stop_task_worker():
    _worker_stop_event.set()
    kill_worker_subprocess()
    return "任务工作线程已请求停止"


def enqueue_task(
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
    first_frame,
    last_frame,
    tiled,
    animate_reference_image,
    template_video,
    save_folder_path,
    cfg_scale=None,
    sigma_shift=5.0
):
    """将当前生成请求持久化为任务文件并入队（立即返回）。"""

    if not negative_prompt:
        negative_prompt = "发灰，字幕，过曝，细节模糊不清"
    try:
        normalized_generation = normalize_ltx_generation_params(
            model_id=model_id,
            fps=fps,
            width=width,
            height=height,
            num_frames=num_frames,
            tiled=tiled,
        )

        _ensure_task_dirs()
        task_id = str(uuid.uuid4())
        task_dir = TASK_QUEUE_DIR / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        depth_video_path = _copy_if_exists(depth_video, task_dir, "depth_video.mp4") if depth_video else None
        template_video_path = _copy_if_exists(template_video, task_dir, "template_video.mp4") if template_video else None

        reference_image_path = _save_pil_image_if_needed(reference_image, task_dir, "reference_image.png")
        first_frame_path = _save_pil_image_if_needed(first_frame, task_dir, "first_frame.png")
        last_frame_path = _save_pil_image_if_needed(last_frame, task_dir, "last_frame.png")
        animate_reference_image_path = _save_pil_image_if_needed(animate_reference_image, task_dir, "animate_reference_image.png")

        params = {
            "depth_video": depth_video_path,
            "reference_image": reference_image_path,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": int(seed) if seed is not None else -1,
            "fps": normalized_generation["fps"],
            "quality": int(quality) if quality is not None else 8,
            "height": normalized_generation["height"],
            "width": normalized_generation["width"],
            "num_frames": normalized_generation["num_frames"],
            "num_inference_steps": int(num_inference_steps) if num_inference_steps is not None else 40,
            "memory_mode": memory_mode or DEFAULT_MEMORY_MODE,
            "model_id": model_id,
            "first_frame": first_frame_path,
            "last_frame": last_frame_path,
            "tiled": normalized_generation["tiled"],
            "animate_reference_image": animate_reference_image_path,
            "template_video": template_video_path,
            "save_folder_path": save_folder_path or "./outputs",
            "cfg_scale": float(cfg_scale) if cfg_scale is not None else get_default_cfg_scale(model_id),
            "sigma_shift": float(sigma_shift) if sigma_shift is not None else 5.0,
        }

        task = {
            "id": task_id,
            "created_at": datetime.now().isoformat(),
            "status": TASK_STATUS_PENDING,
            "retries": 0,
            "max_retries": MAX_RETRIES,
            "params": params,
            "result": None,
        }

        task_file = task_dir / f"task_{task_id}.json"
        _atomic_write_json(task_file, task)

        print(f"[enqueue] 新任务已创建: {task_id} 于 {task_dir}")
        start_task_worker()

        return f"✅ 任务已入队：{task_id}\n📁 队列目录：{str(task_dir)}\n⏳ 稍后在后台依次执行，完成后会保存到输出目录。"
    except Exception as e:
        return f"❌ 入队失败：{e}"
