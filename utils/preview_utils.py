"""
视频预览工具模块
用于扫描、加载和显示已生成的视频及其参数
"""
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from PIL import Image
import subprocess
import tempfile


def scan_generation_directories(output_dir: str = "./outputs") -> List[Dict]:
    """
    扫描输出目录，查找所有已完成的生成任务
    
    Args:
        output_dir: 输出目录路径
        
    Returns:
        任务列表，每个任务包含路径、时间戳、任务ID等信息
    """
    output_path = Path(output_dir).resolve()
    if not output_path.exists():
        return []
    
    tasks = []
    
    # 扫描所有generation_开头的文件夹
    for gen_dir in sorted(output_path.glob("generation_*"), reverse=True):
        if not gen_dir.is_dir():
            continue
        
        # 查找任务JSON文件（可能在子目录中）
        task_json_files = list(gen_dir.rglob("task_*.json"))
        
        if not task_json_files:
            # 如果没有找到任务JSON，尝试查找视频文件
            video_files = list(gen_dir.glob("*.mp4"))
            if video_files:
                # 创建基本信息
                task_info = {
                    "generation_dir": str(gen_dir),
                    "task_id": gen_dir.name.split("_")[-1] if "_" in gen_dir.name else "unknown",
                    "created_at": _extract_timestamp_from_dirname(gen_dir.name),
                    "video_path": str(video_files[0]),
                    "has_task_json": False,
                    "task_data": None,
                }
                tasks.append(task_info)
            continue
        
        # 读取任务JSON
        for task_json_file in task_json_files:
            try:
                with open(task_json_file, 'r', encoding='utf-8') as f:
                    task_data = json.load(f)
                
                # 查找视频文件
                video_path = None
                result = task_data.get("result", {})
                if result:
                    moved_video = result.get("moved_video")
                    if moved_video and os.path.exists(moved_video):
                        video_path = moved_video
                    else:
                        # 尝试在任务目录中查找视频
                        task_dir = task_json_file.parent
                        video_files = list(task_dir.glob("*.mp4"))
                        if video_files:
                            video_path = str(video_files[0])
                
                # 如果任务目录中有视频，也查找
                if not video_path:
                    task_dir = task_json_file.parent
                    video_files = list(task_dir.glob("*.mp4"))
                    if video_files:
                        video_path = str(video_files[0])
                
                # 在generation目录中查找视频
                if not video_path:
                    video_files = list(gen_dir.glob("*.mp4"))
                    if video_files:
                        video_path = str(video_files[0])
                
                task_info = {
                    "generation_dir": str(gen_dir),
                    "task_id": task_data.get("id", "unknown"),
                    "created_at": task_data.get("created_at", _extract_timestamp_from_dirname(gen_dir.name)),
                    "started_at": task_data.get("started_at"),
                    "finished_at": task_data.get("finished_at"),
                    "duration_seconds": task_data.get("duration_seconds"),
                    "status": task_data.get("status", "unknown"),
                    "video_path": video_path,
                    "has_task_json": True,
                    "task_data": task_data,
                }
                tasks.append(task_info)
            except Exception as e:
                print(f"读取任务JSON失败 {task_json_file}: {e}")
                continue
    
    return tasks


def _extract_timestamp_from_dirname(dirname: str) -> str:
    """从目录名中提取时间戳"""
    # generation_20251112_081343_xxx 格式
    parts = dirname.split("_")
    if len(parts) >= 3:
        try:
            date_part = parts[1]  # 20251112
            time_part = parts[2]  # 081343
            return f"{date_part}_{time_part}"
        except:
            pass
    return "unknown"


def get_task_params_summary(task_data: Dict) -> str:
    """
    格式化任务参数为可读的字符串
    
    Args:
        task_data: 任务数据字典
        
    Returns:
        格式化的参数字符串
    """
    if not task_data:
        return "无参数信息"
    
    params = task_data.get("params", {})
    if not params:
        return "无参数信息"
    
    lines = []
    lines.append("### 生成参数")
    lines.append(f"- **模型**: {params.get('model_id', 'N/A')}")
    lines.append(f"- **尺寸**: {params.get('width', 'N/A')} × {params.get('height', 'N/A')}")
    lines.append(f"- **帧数**: {params.get('num_frames', 'N/A')}")
    lines.append(f"- **推理步数**: {params.get('num_inference_steps', 'N/A')}")
    lines.append(f"- **FPS**: {params.get('fps', 'N/A')}")
    lines.append(f"- **质量**: {params.get('quality', 'N/A')}")
    lines.append(f"- **种子**: {params.get('seed', 'N/A')}")
    lines.append(f"- **显存限制**: {params.get('vram_limit', 'N/A')} GB")
    
    if params.get('prompt'):
        lines.append(f"\n### 提示词")
        lines.append(f"**正面**: {params.get('prompt', 'N/A')}")
        if params.get('negative_prompt'):
            lines.append(f"**负面**: {params.get('negative_prompt', 'N/A')}")
    
    # 时间信息
    if task_data.get("duration_seconds"):
        duration = task_data.get("duration_seconds")
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        lines.append(f"\n### 处理信息")
        lines.append(f"- **处理时长**: {minutes}分{seconds}秒 ({duration:.2f}秒)")
    
    if task_data.get("created_at"):
        lines.append(f"- **创建时间**: {task_data.get('created_at')}")
    if task_data.get("finished_at"):
        lines.append(f"- **完成时间**: {task_data.get('finished_at')}")
    
    return "\n".join(lines)


def extract_video_thumbnail(video_path: str, output_path: Optional[str] = None) -> Optional[str]:
    """
    从视频中提取第一帧作为缩略图
    
    Args:
        video_path: 视频文件路径
        output_path: 输出缩略图路径，如果为None则自动生成
        
    Returns:
        缩略图文件路径，如果失败返回None
    """
    if not video_path or not os.path.exists(video_path):
        return None
    
    try:
        if output_path is None:
            # 在视频同目录下生成缩略图
            video_file = Path(video_path)
            output_path = str(video_file.parent / f"{video_file.stem}_thumb.jpg")
        
        output_path_obj = Path(output_path)
        
        # 检查缩略图是否已存在，且视频文件未被修改
        if output_path_obj.exists():
            video_mtime = os.path.getmtime(video_path)
            thumb_mtime = os.path.getmtime(output_path)
            # 如果缩略图比视频新或时间相近（1秒内），直接使用缓存
            if thumb_mtime >= video_mtime - 1:
                return output_path
        
        # 使用ffmpeg提取第一帧
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", "scale=320:-1",  # 缩放到宽度320，保持宽高比
            "-frames:v", "1",
            "-y",  # 覆盖已存在的文件
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10
        )
        
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path
        else:
            # 如果ffmpeg失败，尝试使用PIL（需要opencv或其他库）
            # 这里先返回None，后续可以改进
            return None
    except Exception as e:
        print(f"提取缩略图失败 {video_path}: {e}")
        return None


def refresh_preview_list(output_dir: str = "./outputs") -> Tuple[List[Tuple], Optional[int], List[Dict]]:
    """
    刷新预览列表，返回任务缩略图列表和默认选中的任务索引
    
    Args:
        output_dir: 输出目录路径
        
    Returns:
        (任务缩略图列表, 默认选中任务的索引, 任务数据列表)
        每个任务是 (图片路径, 标题) 元组，用于Gradio Gallery组件
    """
    tasks = scan_generation_directories(output_dir)
    
    if not tasks:
        return [], None, []
    
    gallery_items = []
    for i, task in enumerate(tasks):
        video_path = task.get("video_path")
        generation_dir = task.get("generation_dir", "")
        task_id = task.get("task_id", "unknown")
        
        # 获取文件夹名称作为标题
        folder_name = Path(generation_dir).name if generation_dir else f"unknown_{task_id[:8]}"
        
        # 生成缩略图
        thumbnail_path = None
        if video_path and os.path.exists(video_path):
            thumbnail_path = extract_video_thumbnail(video_path)
            # 转换为绝对路径
            if thumbnail_path:
                thumbnail_path = str(Path(thumbnail_path).resolve())
        
        # 如果没有缩略图，使用占位符
        if not thumbnail_path:
            # 创建一个简单的占位符图片
            placeholder = Image.new('RGB', (320, 180), color=(50, 50, 50))
            placeholder_path = str(Path(output_dir) / f"placeholder_{task_id}.jpg")
            try:
                placeholder.save(placeholder_path)
                thumbnail_path = str(Path(placeholder_path).resolve())
            except:
                thumbnail_path = None
        
        # 构建标题 - 直接使用文件夹名称
        caption = folder_name
        
        # Gallery组件需要 (图片路径, 标题) 元组格式
        if thumbnail_path:
            gallery_items.append((thumbnail_path, caption))
        else:
            # 如果还是没有缩略图，创建一个默认的
            default_placeholder = Image.new('RGB', (320, 180), color=(100, 100, 100))
            default_path = str(Path(tempfile.gettempdir()) / f"default_thumb_{task_id}.jpg")
            try:
                default_placeholder.save(default_path)
                gallery_items.append((str(Path(default_path).resolve()), caption))
            except:
                # 最后的备选方案：只使用标题
                gallery_items.append((None, caption))
    
    # 返回缩略图列表、默认索引以及任务列表，供前端缓存
    return gallery_items, 0 if gallery_items else None, tasks


def load_task_preview(
    selected_index: int,
    output_dir: str = "./outputs",
    cached_tasks: Optional[List[Dict]] = None,
) -> Tuple[Optional[str], str, str]:
    """
    加载选中任务的预览信息
    
    Args:
        selected_index: 选中的任务索引（从Gallery组件的事件中获取）
        output_dir: 输出目录路径
        
    Returns:
        (视频路径, 参数摘要, 任务详情JSON)
    """
    tasks = cached_tasks if cached_tasks is not None else scan_generation_directories(output_dir)
    
    if not tasks or selected_index is None or selected_index >= len(tasks):
        return None, "未找到任务", "{}"
    
    task = tasks[selected_index]
    video_path = task.get("video_path")
    
    # 检查视频文件是否存在
    if video_path and not os.path.exists(video_path):
        video_path = None
    
    # 获取参数摘要
    task_data = task.get("task_data", {})
    params_summary = get_task_params_summary(task_data) if task_data else "无参数信息"
    
    # 返回任务详情JSON（格式化）
    task_json = json.dumps(task_data, ensure_ascii=False, indent=2) if task_data else "{}"
    
    return video_path, params_summary, task_json


def delete_task_files(task: Dict) -> Tuple[bool, str]:
    """
    删除任务对应的生成目录及相关文件
    
    Args:
        task: 任务信息字典
        
    Returns:
        (是否成功, 提示信息)
    """
    if not task:
        return False, "未找到任务信息"
    
    generation_dir = task.get("generation_dir")
    if not generation_dir:
        return False, "任务缺少生成目录信息"
    
    generation_path = Path(generation_dir)
    video_path = task.get("video_path")
    
    try:
        # 优先删除生成目录
        if generation_path.exists():
            shutil.rmtree(generation_path)
        else:
            return False, f"目录不存在：{generation_path}"
        
        # 如果视频文件在目录外，单独删除
        if video_path:
            video_path_obj = Path(video_path)
            try:
                # Python 3.9+: Path.is_relative_to
                if not video_path_obj.is_relative_to(generation_path):
                    if video_path_obj.exists():
                        video_path_obj.unlink()
            except ValueError:
                # Python <3.9 fallback
                try:
                    video_path_obj.relative_to(generation_path)
                except ValueError:
                    if video_path_obj.exists():
                        video_path_obj.unlink()
        
        return True, f"已删除任务目录：{generation_path.name}"
    except Exception as e:
        return False, f"删除任务失败：{e}"

