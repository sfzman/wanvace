"""
任务队列模块
处理后台任务队列的创建、执行和管理
"""
import os
import json
import threading
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from utils.video_processor import process_video
from utils.model_config import VACE_MODELS

# 任务队列配置
TASK_QUEUE_DIR = Path("./task_queue").resolve()
TASK_STATUS_PENDING = "pending"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_DONE = "done"
TASK_STATUS_FAILED = "failed"
MAX_RETRIES = 2

_worker_thread = None
_worker_stop_event = threading.Event()


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
    # 如果是路径字符串，复制到任务目录；如果是PIL对象，保存为PNG
    if isinstance(img_obj, str):
        return _copy_if_exists(img_obj, dst_dir, filename)
    try:
        out_path = dst_dir / filename
        img_obj.save(str(out_path))
        return str(out_path)
    except Exception:
        return None


def _recover_orphan_running_tasks():
    # 将上次异常退出留下的running任务重置为pending（递归扫描子目录）
    for task_file in TASK_QUEUE_DIR.rglob("task_*.json"):
        data = _load_json(task_file)
        if not data:
            continue
        if data.get("status") == TASK_STATUS_RUNNING:
            data["status"] = TASK_STATUS_PENDING
            _atomic_write_json(task_file, data)


def _iter_pending_tasks():
    # 返回按创建时间排序的待处理任务（递归扫描子目录）
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
            try:
                # 在工作线程中禁用process_video的复制保存逻辑，改为后续统一剪切
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
                    params.get("vram_limit", 6.0),
                    params.get("model_id", VACE_MODELS[0]),
                    params.get("first_frame"),
                    params.get("last_frame"),
                    params.get("tiled", False),
                    params.get("animate_reference_image"),
                    params.get("template_video"),
                    "",  # 传空以跳过process_video中的复制保存
                    params.get("cfg_scale", 1.0),
                    params.get("sigma_shift", 5.0),
                )
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

                        # 先移动输出视频
                        out_src = Path(out_path).resolve()
                        if out_src.exists():
                            moved_video_path = dest_dir / out_src.name
                            shutil.move(str(out_src), str(moved_video_path))
                            moved_video = str(moved_video_path)

                        # 再移动任务目录
                        task_dir = task_file.parent.resolve()
                        if task_dir.exists():
                            dest_task_dir = dest_dir / task_dir.name
                            shutil.move(str(task_dir), str(dest_task_dir))
                            moved_dir = str(dest_task_dir)
                        print(f"[worker] 已剪切到输出目录: 视频={moved_video}, 任务目录={moved_dir}")
                    except Exception as move_e:
                        print(f"[worker] 剪切到输出目录失败: {move_e}")
                else:
                    # 失败时打印错误信息
                    print(f"[worker] 任务失败: {task_id}, 错误信息: {msg}")
                task["result"] = {
                    "output_video": out_path,
                    "message": msg,
                    "moved_video": moved_video,
                    "moved_task_dir": moved_dir,
                }
                print(f"[worker] 任务完成: {task_id}, 状态: {task['status']}, 输出: {out_path}")
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                task["status"] = TASK_STATUS_FAILED
                task["finished_at"] = datetime.now().isoformat()
                error_msg = f"执行异常：{str(e)}\n\n详细错误信息：\n{error_trace}"
                task["result"] = {"output_video": None, "message": error_msg}
                print(f"[worker] 任务异常: {task_id}")
                print(f"[worker] 错误: {str(e)}")
                print(f"[worker] 完整堆栈:\n{error_trace}")
            finally:
                # 计算处理时长（秒）
                if "started_at" in task:
                    end_time = datetime.now()
                    duration_seconds = (end_time - start_time).total_seconds()
                    task["duration_seconds"] = round(duration_seconds, 2)
                    print(f"[worker] 任务处理时长: {task['duration_seconds']} 秒")
                
                task["retries"] = int(task.get("retries", 0)) + (0 if task["status"] == TASK_STATUS_DONE else 1)
                _atomic_write_json(task_file, task)
        except Exception:
            # 防御性：任何循环异常都不应终止线程
            _worker_stop_event.wait(1.0)


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
    vram_limit,
    model_id,
    first_frame,
    last_frame,
    tiled,
    animate_reference_image,
    template_video,
    save_folder_path,
    cfg_scale=1.0,
    sigma_shift=5.0
):
    """将当前生成请求持久化为任务文件并入队（立即返回）。"""

    if not negative_prompt:
        negative_prompt = "色调艳丽，过曝，细节模糊不清"
    try:
        _ensure_task_dirs()
        task_id = str(uuid.uuid4())
        task_dir = TASK_QUEUE_DIR / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        # 将可能的输入文件复制/保存到任务目录，确保重启后可用
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
            "fps": int(fps) if fps is not None else 16,
            "quality": int(quality) if quality is not None else 8,
            "height": int(height) if height is not None else 480,
            "width": int(width) if width is not None else 832,
            "num_frames": int(num_frames) if num_frames is not None else 81,
            "num_inference_steps": int(num_inference_steps) if num_inference_steps is not None else 40,
            "vram_limit": float(vram_limit) if vram_limit is not None else 6.0,
            "model_id": model_id,
            "first_frame": first_frame_path,
            "last_frame": last_frame_path,
            "tiled": bool(tiled),
            "animate_reference_image": animate_reference_image_path,
            "template_video": template_video_path,
            "save_folder_path": save_folder_path or "./outputs",
            "cfg_scale": float(cfg_scale) if cfg_scale is not None else 1.0,
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

        # 确保后台线程已启动
        print(f"[enqueue] 新任务已创建: {task_id} 于 {task_dir}")
        start_task_worker()

        return f"✅ 任务已入队：{task_id}\n📁 队列目录：{str(task_dir)}\n⏳ 稍后在后台依次执行，完成后会保存到输出目录。"
    except Exception as e:
        return f"❌ 入队失败：{e}"

