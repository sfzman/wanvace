import random
import torch
import gradio as gr
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import time
import os
import gc
import tempfile
import shutil
import json
import threading
import uuid
from pathlib import Path
from datetime import datetime

from utils.video_utils import clean_temp_videos, reencode_video_to_16fps, get_video_info
from utils.img_utils import get_image_info
from utils.vram_utils import clear_vram, get_vram_info
from utils.animate.preprocess.process_pipepline import ProcessPipeline

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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 全局变量存储pipeline和模型选择
pipe:WanVideoPipeline = None
selected_model = "PAI/Wan2.2-VACE-Fun-A14B"  # 默认选择14B模型
input_mode = "vace"  # 默认输入模式：vace（深度视频+参考图片）或 inp（首尾帧）

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
]

ANIMATE_MODELS = [
    "Wan-AI/Wan2.2-Animate-14B"
]

def get_models_by_mode(mode):
    """根据输入模式返回对应的模型列表"""
    if mode == "vace":
        return VACE_MODELS
    elif mode == "inp":
        return INP_MODELS
    elif mode == "animate":
        return ANIMATE_MODELS
    else:
        return VACE_MODELS  # 默认返回VACE模型

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
            task["started_at"] = datetime.now().isoformat()
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
                    params.get("num_inference_steps", 40),
                    params.get("vram_limit", 6.0),
                    params.get("model_id", VACE_MODELS[0]),
                    params.get("first_frame"),
                    params.get("last_frame"),
                    params.get("tiled", True),
                    params.get("animate_reference_image"),
                    params.get("template_video"),
                    "",  # 传空以跳过process_video中的复制保存
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
    save_folder_path
):
    """将当前生成请求持久化为任务文件并入队（立即返回）。"""
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

        return None, f"任务已入队：{task_id}\n队列目录：{str(task_dir)}\n稍后在后台依次执行。"
    except Exception as e:
        return None, f"入队失败：{e}"

def handle_tab_change(evt: gr.SelectData):
    """处理Tab切换事件"""
    # evt.index: 0 = VACE模式, 1 = 首尾帧模式, 2 = Animate模式
    if evt.index == 0:  # VACE模式
        models = VACE_MODELS
        default_model = VACE_MODELS[0]
        mode = "vace"
    elif evt.index == 1:  # 首尾帧模式
        models = INP_MODELS
        default_model = INP_MODELS[0]
        mode = "inp"
    else:  # Animate模式
        models = ANIMATE_MODELS
        default_model = ANIMATE_MODELS[0]
        mode = "animate"
    
    # 更新全局变量
    global input_mode, selected_model
    input_mode = mode
    selected_model = default_model
    
    # 返回新的模型选择列表和默认值
    return gr.Dropdown(choices=models, value=default_model)


def update_dimensions(aspect_ratio):
    """根据选择的宽高比更新高度和宽度"""
    if aspect_ratio in ASPECT_RATIOS_14b:
        width, height = ASPECT_RATIOS_14b[aspect_ratio]
        return height, width
    return 480, 832  # 默认值


def update_size_display(aspect_ratio):
    """更新Dropdown的info显示当前尺寸"""
    if aspect_ratio in ASPECT_RATIOS_14b:
        width, height = ASPECT_RATIOS_14b[aspect_ratio]
        size_text = f"{width} × {height}"
    else:
        size_text = "832 × 480"  # 默认值
    
    info_text = f"选择预设的宽高比，系统会自动计算对应的尺寸\n当前尺寸: {size_text}"
    return gr.Dropdown(info=info_text)


# 已弃用：保存生成结果逻辑改由后台任务线程在任务完成后进行剪切处理


def preprocess_template_video(template_video_path, reference_image_path, width, height, num_frames):
    """预处理模板视频，生成pose和face视频"""
    try:
        # 创建临时目录用于存储预处理结果
        temp_dir = tempfile.mkdtemp(prefix="wanvace_preprocess_")
        
        # 构建模型路径
        ckpt_path = "./models"
        pose2d_checkpoint_path = os.path.join(ckpt_path, 'pose2d/vitpose_h_wholebody.onnx')
        det_checkpoint_path = os.path.join(ckpt_path, 'det/yolov10m.onnx')
        
        # 检查模型文件是否存在
        if not os.path.exists(pose2d_checkpoint_path):
            error_msg = f"姿态检测模型文件不存在: {pose2d_checkpoint_path}"
            print(error_msg)
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None, None, error_msg
        
        if not os.path.exists(det_checkpoint_path):
            error_msg = f"目标检测模型文件不存在: {det_checkpoint_path}"
            print(error_msg)
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None, None, error_msg
        
        print(f"开始预处理模板视频...")
        print(f"模板视频: {template_video_path}")
        print(f"参考图片: {reference_image_path}")
        print(f"目标分辨率: {width}x{height}")
        print(f"输出目录: {temp_dir}")
        
        # 创建ProcessPipeline实例
        process_pipeline = ProcessPipeline(
            det_checkpoint_path=det_checkpoint_path,
            pose2d_checkpoint_path=pose2d_checkpoint_path,
            sam_checkpoint_path=None,  # 不使用SAM
            flux_kontext_path=None     # 不使用FLUX
        )
        
        # 调用预处理
        process_pipeline(
            video_path=template_video_path,
            refer_image_path=reference_image_path,
            output_path=temp_dir,
            resolution_area=[width, height],
            fps=30,  # 使用默认FPS
            iterations=3,
            k=7,
            w_len=1,
            h_len=1,
            retarget_flag=True,  # 启用姿态重定向
            use_flux=False,
            replace_flag=False
        )
        
        # 查找生成的pose和face视频
        pose_video_path = os.path.join(temp_dir, "src_pose.mp4")
        face_video_path = os.path.join(temp_dir, "src_face.mp4")
        reference_image_path = os.path.join(temp_dir, "src_ref.png")
        
        if not os.path.exists(pose_video_path):
            error_msg = "预处理未生成pose视频文件"
            print(error_msg)
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None, None, error_msg
        
        if not os.path.exists(face_video_path):
            print("警告: 未找到face视频文件，将只使用pose视频")
            face_video_path = None

        if not os.path.exists(reference_image_path):
            print("警告: 未找到reference image文件，将只使用视频")
            reference_image_path = None
        
        print(f"预处理成功完成")
        print(f"Pose视频: {pose_video_path}")
        print(f"Face视频: {face_video_path}")
        print(f"Reference Image: {reference_image_path}")
        return pose_video_path, face_video_path, reference_image_path, temp_dir
        
    except Exception as e:
        error_msg = f"预处理过程中出现错误: {str(e)}"
        print(error_msg)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None, None, error_msg


def initialize_pipeline(model_id="PAI/Wan2.2-VACE-Fun-A14B", vram_limit=6.0):
    """初始化WanVideoPipeline"""
    global pipe, selected_model, input_mode
    
    # 如果模型改变了，需要重新初始化
    if pipe is not None and selected_model != model_id:
        print(f"模型从 {selected_model} 切换到 {model_id}，重新初始化...")
        del pipe
        pipe = None
        torch.cuda.empty_cache()
        gc.collect()
    
    if pipe is None:
        print(f"正在初始化模型: {model_id}")
        selected_model = model_id
        
        # 根据模型类型设置输入模式
        if "InP" in model_id or "I2V" in model_id:
            input_mode = "inp"
        elif "Animate" in model_id:
            input_mode = "animate"
        else:
            input_mode = "vace"
        
        if model_id == "PAI/Wan2.2-VACE-Fun-A14B":
            # 14B VACE模型配置
            pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(model_id="PAI/Wan2.2-VACE-Fun-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
                    ModelConfig(model_id="PAI/Wan2.2-VACE-Fun-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
                    ModelConfig(model_id="PAI/Wan2.2-VACE-Fun-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
                    ModelConfig(model_id="PAI/Wan2.2-VACE-Fun-A14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
                ],
            )
        elif model_id == "Wan-AI/Wan2.1-VACE-1.3B":
            # 1.3B VACE模型配置
            pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
                    ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
                    ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
                ],
            )
        elif model_id == "PAI/Wan2.2-Fun-A14B-InP":
            # 14B InP模型配置
            pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-InP", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
                    ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-InP", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
                    ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-InP", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
                    ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-InP", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
                ],
            )
        elif model_id == "PAI/Wan2.1-Fun-V1.1-1.3B-InP":
            # 1.3B InP模型配置
            pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
                    ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
                    ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
                    ModelConfig(model_id="PAI/Wan2.1-Fun-1.3B-InP", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
                ],
            )
        elif model_id == "Wan-AI/Wan2.1-I2V-14B-480P":
            print(f"正在初始化480P I2V模型: {model_id}")
            # 480P I2V模型配置
            pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
                    ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
                    ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
                    ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
                ],
            )
        elif model_id == "Wan-AI/Wan2.2-Animate-14B":
            # 14B Animate模型配置（与test.py保持一致）
            pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
                    ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
                    ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
                    ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
                ],
            )
        
        # 将GB转换为字节
        num_persistent_param_in_dit = int(vram_limit * 1024**3)
        pipe.enable_vram_management(num_persistent_param_in_dit=num_persistent_param_in_dit)
        
        print(f"Pipeline初始化完成，使用模型: {model_id}")
        print(f"输入模式: {input_mode}")
        print(f"显存限制: {vram_limit}GB")
    return f"Pipeline初始化完成！使用模型: {model_id} (模式: {input_mode})"


def process_video(
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
    first_frame=None,
    last_frame=None,
    tiled=True,
    animate_reference_image=None,
    template_video=None,
    save_folder_path="./outputs"
):
    """处理视频生成"""
    try:
        # 根据模型类型判断输入模式
        is_inp_mode = ("InP" in model_id) or ("I2V" in model_id)
        is_animate_mode = "Animate" in model_id
        
        if is_inp_mode:
            # InP模式：需要首帧，尾帧可选
            has_first_frame = first_frame is not None
            if not has_first_frame:
                return None, "错误：首尾帧模式需要上传首帧图片"
        elif is_animate_mode:
            # Animate模式：需要参考图片和模板视频
            has_reference_image = animate_reference_image is not None
            has_template_video = template_video is not None and template_video != ""
            if not has_reference_image:
                return None, "错误：Animate模式需要上传参考图片"
            if not has_template_video:
                return None, "错误：Animate模式需要上传模板视频"
        else:
            # VACE模式：需要深度视频或参考图片
            has_depth_video = depth_video is not None and depth_video != ""
            has_reference_image = reference_image is not None
            if not has_depth_video and not has_reference_image:
                return None, "错误：请至少上传深度视频或参考图片中的一种"
        
        if seed < 0:
            seed = random.randint(1, 2**32 - 1)
        
        # 自动初始化pipeline（如果还没有初始化）
        if pipe is None:
            status = initialize_pipeline(model_id, vram_limit)
            if "完成" not in status:
                return None, f"model_id初始化失败：{status}"
        
        vace_video = None
        vace_reference_image = None
        animate_reference_img = None
        template_video_data = None

        if not prompt:
            prompt = "两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。"
        
        if not negative_prompt:
            negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        
        if is_inp_mode:
            # InP模式：处理首尾帧
            if has_first_frame:
                print(f"调试信息：首帧图片 = {first_frame}, 类型 = {type(first_frame)}")
                if isinstance(first_frame, str):
                    first_frame_img = Image.open(first_frame).resize((width, height)).convert("RGB")
                else:
                    first_frame_img = first_frame.resize((width, height)).convert("RGB")
                
                # 如果有尾帧，也处理
                last_frame_img = None
                if last_frame is not None:
                    print(f"调试信息：尾帧图片 = {last_frame}, 类型 = {type(last_frame)}")
                    if isinstance(last_frame, str):
                        last_frame_img = Image.open(last_frame).resize((width, height)).convert("RGB")
                    else:
                        last_frame_img = last_frame.resize((width, height)).convert("RGB")
                
                # 调用InP模型的pipeline
                video = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    input_image=first_frame_img,
                    end_image=last_frame_img,
                    seed=seed,
                    tiled=True,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    cfg_scale=6.0,
                    sigma_shift=6.0,
                )
            else:
                return None, "首尾帧模式需要上传首帧图片"
        elif is_animate_mode:
            # Animate模式：处理参考图片和模板视频
            if has_reference_image:
                print(f"调试信息：Animate参考图片 = {animate_reference_image}, 类型 = {type(animate_reference_image)}")
                if isinstance(animate_reference_image, str):
                    animate_reference_img = Image.open(animate_reference_image).resize((width, height)).convert("RGB")
                else:
                    animate_reference_img = animate_reference_image.resize((width, height)).convert("RGB")
            
            if has_template_video:
                # 预处理模板视频
                temp_template_path = template_video
                print(f"调试信息：模板视频路径 = {temp_template_path}, 类型 = {type(temp_template_path)}")
                print(f"目标尺寸: {width}x{height}")
                
                # 保存参考图片到临时文件用于预处理
                temp_ref_path = None
                if isinstance(animate_reference_image, str):
                    temp_ref_path = animate_reference_image
                else:
                    temp_ref_path = tempfile.mktemp(suffix=".png")
                    animate_reference_image.save(temp_ref_path)
                
                # 调用预处理函数
                pose_video_path, face_video_path, reference_image_path, temp_dir = preprocess_template_video(
                    temp_template_path, temp_ref_path, width, height, num_frames
                )
                animate_reference_img = Image.open(reference_image_path).resize((width, height)).convert("RGB")
                
                if pose_video_path is None:
                    return None, f"模板视频预处理失败：{face_video_path}"  # face_video_path此时包含错误信息
                
                try:
                    # 加载预处理后的pose视频
                    pose_video_data = VideoData(pose_video_path, height=height, width=width).raw_data()[:num_frames-4]
                    print(f"成功加载pose视频数据，帧数: {len(pose_video_data)}")
                    
                    # 如果有face视频，也加载
                    face_video_data = None
                    if face_video_path and os.path.exists(face_video_path):
                        face_video_data = VideoData(face_video_path).raw_data()[:num_frames-4]
                        print(f"成功加载face视频数据，帧数: {len(face_video_data)}")
                    
                except Exception as e:
                    print(f"预处理视频加载失败: {e}")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return None, f"预处理视频加载失败：{str(e)}"
            
            # 调用Animate模型的pipeline（与test.py保持一致）
            video = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                input_image=animate_reference_img,
                animate_pose_video=pose_video_data,
                animate_face_video=face_video_data,
                seed=seed,
                tiled=True,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                cfg_scale=6.0,
                sigma_shift=6.0,
            )
            
            # 清理临时文件
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            # VACE模式：处理深度视频和参考图片
            if has_depth_video:
                # Gradio Video组件返回的是文件路径字符串
                temp_video_path = depth_video
                print(f"调试信息：深度视频路径 = {temp_video_path}, 类型 = {type(temp_video_path)}")
                print(f"目标尺寸: {width}x{height}")
                # 把目标视频宽高和帧率转换为16fps
                temp_video_path = reencode_video_to_16fps(temp_video_path, num_frames, width, height)
                
                # 创建VideoData时指定目标尺寸，让系统自动调整
                try:
                    vace_video = VideoData(temp_video_path, height=height, width=width)
                    print(f"成功创建VideoData，尺寸: {width}x{height}")
                except Exception as e:
                    print(f"VideoData创建失败: {e}")
                    return None, f"深度视频处理失败：{str(e)}\n请确保视频尺寸与目标尺寸兼容"
            
            if has_reference_image:
                print(f"调试信息：参考图片 = {reference_image}, 类型 = {type(reference_image)}")
                if isinstance(reference_image, str):
                    vace_reference_image = Image.open(reference_image).resize((width, height)).convert("RGB")
                else:
                    vace_reference_image = reference_image.resize((width, height)).convert("RGB")
        
            video = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                vace_video=vace_video,
                vace_reference_image=vace_reference_image,
                seed=seed,
                tiled=tiled,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                cfg_scale=6.0,
                sigma_shift=6.0,
                vace_scale=0.8,
            )
        
        timestamp = int(time.time())
        output_path = f"output_video_{seed}_{timestamp}.mp4"
        save_video(video, output_path, fps=fps, quality=quality)

        print("cleaning temp videos...")
        clean_temp_videos()
        print("clearing vram...")
        clear_vram(pipe)
        
        # 不再在此处保存/复制；统一由后台线程在任务成功后剪切
        return output_path, f"视频生成成功！已保存为 {output_path}"
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return None, f"生成过程中出现错误：{str(e)}\n\n详细错误信息：\n{error_trace}"

def create_interface():
    """创建Gradio界面"""
    with gr.Blocks(title="WanVACE 视频生成器", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎬 WanVACE 视频生成器")
        gr.Markdown("使用Wan2.1-VACE-14B模型生成高质量视频")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📤 输入设置")
                
                with gr.Tabs() as input_tabs:
                    with gr.TabItem("🎬 VACE模式 (深度视频+参考图片)", id="vace_tab"):
                        depth_video = gr.Video(
                            label="深度视频 (Depth Video)",
                            height=200,
                        )
                        
                        video_info = gr.Textbox(
                            label="视频信息",
                            value="未上传视频",
                            interactive=False,
                            info="显示上传视频的原始信息"
                        )
                        
                        reference_image = gr.Image(
                            label="参考图片 (Reference Image)",
                            height=200,
                            type="pil",
                        )
                        
                        reference_image_info = gr.Textbox(
                            label="参考图片信息",
                            value="未上传图片",
                            interactive=False,
                            info="显示参考图片的尺寸、格式等信息"
                        )
                    
                    with gr.TabItem("🖼️ 首尾帧模式 (首帧+尾帧)", id="inp_tab"):
                        first_frame = gr.Image(
                            label="首帧图片 (First Frame)",
                            height=200,
                            type="pil",
                        )
                        
                        first_frame_info = gr.Textbox(
                            label="首帧图片信息",
                            value="未上传图片",
                            interactive=False,
                            info="显示首帧图片的尺寸、格式等信息"
                        )
                        
                        last_frame = gr.Image(
                            label="尾帧图片 (Last Frame)",
                            height=200,
                            type="pil"
                        )
                        
                        last_frame_info = gr.Textbox(
                            label="尾帧图片信息",
                            value="未上传图片",
                            interactive=False,
                            info="显示尾帧图片的尺寸、格式等信息"
                        )
                    
                    with gr.TabItem("🎭 Animate模式 (参考图片+模板视频)", id="animate_tab"):
                        animate_reference_image = gr.Image(
                            label="参考图片 (Reference Image)",
                            height=200,
                            type="pil",
                        )
                        
                        animate_reference_image_info = gr.Textbox(
                            label="参考图片信息",
                            value="未上传图片",
                            interactive=False,
                            info="显示参考图片的尺寸、格式等信息"
                        )
                        
                        template_video = gr.Video(
                            label="模板视频 (Template Video)",
                            height=200,
                        )
                        
                        template_video_info = gr.Textbox(
                            label="模板视频信息",
                            value="未上传视频",
                            interactive=False,
                            info="显示模板视频的原始信息"
                        )
                

                
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ 参数设置")
                
                # 模型选择
                model_id = gr.Dropdown(
                    label="选择模型",
                    choices=VACE_MODELS,
                    value="PAI/Wan2.2-VACE-Fun-A14B",
                    info="模型会根据选择的输入模式自动更新"
                )
                
                prompt = gr.Textbox(
                    label="正面提示词",
                    placeholder="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
                    lines=3
                )
                
                negative_prompt = gr.Textbox(
                    label="负面提示词",
                    placeholder="色调艳丽，过曝，静态，细节模糊不清...",
                    lines=4
                )
                
                with gr.Row():
                    seed = gr.Number(
                        label="随机种子",
                        value=-1,
                        minimum=-1,
                    )
                    fps = gr.Slider(
                        label="FPS",
                        value=16,
                        minimum=1,
                        maximum=60,
                        step=1
                    )
                
                # 视频尺寸设置
                gr.Markdown("### 📐 视频尺寸设置")
                with gr.Tabs() as size_tabs:
                    with gr.TabItem("📏 预设宽高比", id="aspect_ratio_tab"):
                        aspect_ratio = gr.Dropdown(
                            label="选择宽高比",
                            choices=list(ASPECT_RATIOS_14b.keys()),
                            value="16:9_low",
                            info="选择预设的宽高比，系统会自动计算对应的尺寸\n当前尺寸: 832 × 480"
                        )
                    
                    with gr.TabItem("🔧 手动设置", id="manual_size_tab"):
                        with gr.Row():
                            width = gr.Number(
                                label="视频宽度",
                                value=832,
                                minimum=256,
                                maximum=1280,
                                step=64,
                                info="视频宽度（像素）"
                            )
                            height = gr.Number(
                                label="视频高度",
                                value=480,
                                minimum=256,
                                maximum=1280,
                                step=64,
                                info="视频高度（像素）"
                            )
                
                # 显存管理
                with gr.Row():
                    clear_vram_btn = gr.Button("🧹 释放显存", variant="secondary", size="sm")
                    refresh_vram_btn = gr.Button("🔄 刷新显存信息", variant="secondary", size="sm")
                
                vram_info = gr.Textbox(
                    label="显存状态",
                    value="点击刷新按钮查看显存信息",
                    interactive=False,
                    info="显示当前显存使用情况"
                )
                
                quality = gr.Slider(
                    label="质量",
                    value=8,
                    minimum=1,
                    maximum=10,
                    step=1,
                    info="1=最低质量，10=最高质量"
                )
                
                # 新增的高级参数设置
                gr.Markdown("### 🔧 高级参数设置")
                
                with gr.Row():
                    num_frames = gr.Number(
                        label="视频帧数",
                        value=81,
                        minimum=16,
                        maximum=256,
                        step=1,
                        info="视频总帧数，建议16-256之间"
                    )
                    num_inference_steps = gr.Number(
                        label="推理步数",
                        value=40,
                        minimum=10,
                        maximum=100,
                        step=1,
                        info="推理步数，步数越多质量越高但速度越慢"
                    )
                
                with gr.Row():
                    vram_limit = gr.Slider(
                        label="显存占用量限制",
                        value=6.0,
                        minimum=0.0,
                        maximum=100.0,
                        step=1,
                        info="显存占用量限制（GB），影响显存使用和性能"
                    )
                    tiled_checkbox = gr.Checkbox(
                        label="Tiled VAE Decode", 
                        value=True, 
                        info="禁用可能导致VAE错误，但可提高性能"
                    )
                
                # 视频保存设置
                gr.Markdown("### 💾 视频保存设置")
                save_folder_path = gr.Textbox(
                    label="视频保存地址",
                    value="./outputs",
                    placeholder="./outputs 或 /path/to/save/folder",
                    info="支持相对路径和绝对路径，每次生成会创建时间戳子文件夹"
                )
                
                generate_btn = gr.Button("🎬 生成视频", variant="primary", size="lg")
        
        with gr.Row():
            gr.Markdown("## 📹 输出结果")
        
        with gr.Row():
            output_video = gr.Video(label="生成的视频")
            output_status = gr.Textbox(label="生成状态", interactive=False)
        
        # Tab切换时更新模型选择
        input_tabs.select(
            fn=handle_tab_change,
            outputs=[model_id]
        )
        
        # 宽高比选择变化时自动更新尺寸和显示
        aspect_ratio.change(
            fn=update_dimensions,
            inputs=[aspect_ratio],
            outputs=[height, width]
        )
        
        # 宽高比选择变化时更新Dropdown的info显示
        aspect_ratio.change(
            fn=update_size_display,
            inputs=[aspect_ratio],
            outputs=[aspect_ratio]
        )
        
        # 视频上传时自动更新视频信息
        depth_video.change(
            fn=get_video_info,
            inputs=[depth_video],
            outputs=[video_info]
        )
        
        # 图片上传时自动更新图片信息
        reference_image.change(
            fn=get_image_info,
            inputs=[reference_image],
            outputs=[reference_image_info]
        )
        
        first_frame.change(
            fn=get_image_info,
            inputs=[first_frame],
            outputs=[first_frame_info]
        )
        
        last_frame.change(
            fn=get_image_info,
            inputs=[last_frame],
            outputs=[last_frame_info]
        )
        
        # Animate模式的事件处理
        animate_reference_image.change(
            fn=get_image_info,
            inputs=[animate_reference_image],
            outputs=[animate_reference_image_info]
        )
        
        template_video.change(
            fn=get_video_info,
            inputs=[template_video],
            outputs=[template_video_info]
        )
        
        # 显存管理按钮事件
        clear_vram_btn.click(
            fn=clear_vram,
            outputs=[vram_info]
        )
        
        refresh_vram_btn.click(
            fn=get_vram_info,
            outputs=[vram_info]
        )
        
        generate_btn.click(
            fn=enqueue_task,
            inputs=[
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
                tiled_checkbox,
                animate_reference_image,
                template_video,
                save_folder_path
            ],
            outputs=[output_video, output_status]
        )
        
        gr.Markdown("## 📚 使用说明")
        gr.Markdown("""
        1. **选择输入模式**：点击"VACE模式"、"首尾帧模式"或"Animate模式"标签页
        2. **选择模型**：模型会根据选择的标签页自动更新，选择适合的模型（14B质量更高，1.3B速度更快）
        3. **上传文件**：根据选择的模式上传相应的文件
        4. **设置参数**：调整提示词、种子、FPS、质量、视频尺寸和高级参数
        5. **设置保存地址**：指定视频保存文件夹（支持相对路径和绝对路径）
        6. **生成视频**：点击"生成视频"按钮开始处理（首次使用会自动初始化模型）
        
        **标签页与模型对应关系**：
        - **🎬 VACE模式标签页**：显示VACE模型
          - **PAI/Wan2.2-VACE-Fun-A14B**：高质量VACE模型，生成效果更好，但需要更多显存和计算时间
          - **Wan-AI/Wan2.1-VACE-1.3B**：轻量级VACE模型，生成速度更快，显存需求更少，适合快速测试
        - **🖼️ 首尾帧模式标签页**：显示InP模型
          - **PAI/Wan2.2-Fun-A14B-InP**：高质量首尾帧模型，14B参数
          - **PAI/Wan2.1-Fun-V1.1-1.3B-InP**：轻量级首尾帧模型，1.3B参数
        - **🎭 Animate模式标签页**：显示Animate模型
          - **Wan-AI/Wan2.2-Animate-14B**：高质量Animate模型，14B参数，用于基于参考图片和模板视频生成动画
        
        **输入模式详细说明**：
        - **VACE模式**：上传深度视频和/或参考图片
          - 可以单独使用深度视频或参考图片
          - 也可以同时使用两者获得更好的效果
          - 深度视频提供运动信息，参考图片提供视觉风格
        - **首尾帧模式**：上传首帧图片（必需）和尾帧图片（可选）
          - 首帧图片是必需的，用于定义视频的起始状态
          - 尾帧图片是可选的，如果提供会生成从首帧到尾帧的过渡视频
          - 如果不提供尾帧，则只使用首帧生成视频
        - **Animate模式**：上传参考图片（必需）和模板视频（必需）
          - 参考图片是必需的，用于定义要动画化的主体外观
          - 模板视频是必需的，用于提供动画的运动模式和时序
          - 系统会先将模板视频预处理成pose和face视频，然后将参考图片的外观应用到预处理后的视频上
          - 预处理过程可能需要几分钟时间，请耐心等待
        
        **智能模型切换**：
        - 切换标签页时，模型选择会自动更新为对应模式的模型
        - VACE模式标签页只显示VACE模型（Wan-AI/Wan2.1-VACE-*）
        - 首尾帧模式标签页只显示InP模型（PAI/Wan2.1-Fun-V1.1-*-InP）
        - Animate模式标签页只显示Animate模型（Wan-AI/Wan2.2-Animate-14B）
        - 系统会自动选择每个模式的默认模型（14B版本）
        
        **视频尺寸设置**：
        - **预设宽高比标签页**：选择预设的宽高比，系统会自动计算对应的尺寸
          - 支持多种常用宽高比：1:1、4:3、16:9、9:16等
          - 在Dropdown的info中实时显示当前选择的尺寸（如"当前尺寸: 832 × 480"）
          - 选择宽高比后会自动更新手动设置中的数值
        - **手动设置标签页**：直接输入宽度和高度数值
          - 高度和宽度范围：256-1280像素
          - 建议使用64的倍数以获得最佳性能
          - 默认尺寸：832x480（16:9_low，适合大多数显示器）
        - 两个标签页的参数是互斥的，选择预设宽高比时会自动更新手动设置的值
        
        **注意事项**：
        - VACE模式：至少需要上传深度视频或参考图片中的一种
        - 首尾帧模式：必须上传首帧图片，尾帧图片可选
        - Animate模式：必须上传参考图片和模板视频
        - 首次生成时会自动初始化模型，请耐心等待
        - Animate模式的预处理过程需要额外时间（通常1-3分钟），请耐心等待
        - 视频生成需要较长时间，请耐心等待
        - 建议使用较小的视频文件以提高处理速度
        - 较大的视频尺寸会增加处理时间和显存需求
        - 深度视频尺寸应与目标尺寸兼容，系统会自动调整
        - 模板视频尺寸应与目标尺寸兼容，系统会自动调整
        - 如果出现尺寸错误，请尝试使用与原始视频相近的宽高比
        - 如果出现VAE解码错误，请尝试禁用"Tiled VAE Decode"选项
        - Animate模式需要预处理模型文件，请确保models目录下有相应的模型文件
        
        **高级参数说明**：
        - **视频帧数**：控制生成视频的长度，帧数越多视频越长
        - **推理步数**：控制生成质量，步数越多质量越高但速度越慢
        - **显存占用量限制**：控制显存使用，数值越大显存占用越多但性能越好（0-100GB）
        - **Tiled VAE Decode**：启用分块VAE解码，可提高性能但可能导致VAE错误
        
        **视频保存功能**：
        - **保存地址设置**：在"视频保存设置"中指定保存文件夹
        - **支持路径类型**：相对路径（如"./outputs"）和绝对路径（如"/home/user/videos"）
        - **自动子文件夹**：每次生成会创建带时间戳的子文件夹（如"generation_20241201_143022"）
        - **文件保存**：自动保存生成视频和所有输入文件（图片、视频）
        - **参数记录**：生成JSON文件记录所有生成参数，便于复现和调试
        - **文件夹结构**：
          ```
          保存文件夹/
          └── generation_20241201_143022/
              ├── output_video_12345_1701234567.mp4  # 生成的视频
              ├── reference_image.jpg                # 参考图片
              ├── template_video.mp4                 # 模板视频
              └── generation_params.json             # 生成参数
          ```
        
        **显存管理**：
        - 调试中断后点击"释放显存"按钮清理显存
        - 使用"刷新显存信息"查看当前显存使用情况
        - 显存不足时建议先释放显存再重新生成
        - 可以通过调整显存占用量限制来平衡显存使用和性能（0-100GB滑块控制）
        - 切换模型时会自动清理显存并重新初始化
        
        **使用提示**：
        - 根据您的需求选择合适的标签页
        - VACE模式标签页适合有深度视频或参考图片的场景
        - 首尾帧模式标签页适合有起始和结束图片的场景
        - Animate模式标签页适合有参考图片和模板视频的场景，可以生成基于模板运动的动画
        - 系统会根据标签页自动显示相应的输入界面和模型选项
        - 切换标签页时会自动更新模型选择，无需手动调整
        """)
    
    return demo

if __name__ == "__main__":
    # 启动后台任务线程
    try:
        print("启动任务工作线程")
        start_task_worker()
        print("任务工作线程已启动")
    except Exception as e:
        print(f"启动任务工作线程失败：{e}")
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        show_error=True
    )
