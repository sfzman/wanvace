"""LTX2 视频处理模块（阶段 1+2：TI2Vid-HQ / A2Vid）。"""
from __future__ import annotations

import gc
import importlib
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from typing import Any

import torch
from PIL import Image

from utils.app_config import get_ltx2_root, get_ltx2_ti2vid_hq_config
from utils.model_config import LTX2_A2VID_MODEL, LTX2_TI2VID_HQ_MODEL
from utils.video_utils import clean_temp_videos

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

_ltx2_pipe: Any = None
_ltx2_pipe_cache_key: tuple | None = None


def _append_ltx2_python_paths(ltx2_root: str | Path) -> None:
    root = Path(ltx2_root).expanduser().resolve()
    path_candidates = [
        root / "packages" / "ltx-core" / "src",
        root / "packages" / "ltx-pipelines" / "src",
    ]
    for candidate in path_candidates:
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _validate_ltx2_config() -> tuple[dict, str | None]:
    config = get_ltx2_ti2vid_hq_config()

    required_file_fields = [
        "checkpoint_path",
        "distilled_lora_path",
        "spatial_upsampler_path",
    ]
    required_dir_fields = ["gemma_root"]

    missing = []
    invalid = []

    for field in required_file_fields:
        value = config.get(field)
        if not value:
            missing.append(field)
            continue
        if not Path(value).expanduser().exists():
            invalid.append(f"{field}: {value}")

    for field in required_dir_fields:
        value = config.get(field)
        if not value:
            missing.append(field)
            continue
        path = Path(value).expanduser()
        if not path.exists() or not path.is_dir():
            invalid.append(f"{field}: {value}")

    if missing or invalid:
        parts = []
        if missing:
            parts.append(f"缺少环境变量: {', '.join(missing)}")
        if invalid:
            parts.append(f"路径不存在或类型错误: {', '.join(invalid)}")
        hint = (
            "请在 .env 中配置 LTX2_CHECKPOINT_PATH, LTX2_DISTILLED_LORA_PATH, "
            "LTX2_SPATIAL_UPSAMPLER_PATH, LTX2_GEMMA_ROOT"
        )
        return config, f"LTX2 配置无效：{'；'.join(parts)}。{hint}"

    return config, None


def _build_cache_key(model_id: str, config: dict) -> tuple:
    return (
        model_id,
        config.get("checkpoint_path"),
        config.get("distilled_lora_path"),
        config.get("spatial_upsampler_path"),
        config.get("gemma_root"),
        config.get("distilled_lora_strength_stage_1"),
        config.get("distilled_lora_strength_stage_2"),
        config.get("torch_compile"),
        config.get("quantization"),
    )


def _build_ltx2_quantization_policy(config: dict):
    quantization = str(config.get("quantization") or "").lower()
    if not quantization:
        return None

    quantization_mod = importlib.import_module("ltx_core.quantization")
    if quantization in {"fp8-cast", "fp8_cast"}:
        return quantization_mod.QuantizationPolicy.fp8_cast()
    if quantization in {"fp8-scaled-mm", "fp8_scaled_mm"}:
        return quantization_mod.QuantizationPolicy.fp8_scaled_mm()
    raise ValueError(f"不支持的 LTX2_QUANTIZATION: {quantization}")


def clear_ltx2_pipeline() -> None:
    global _ltx2_pipe, _ltx2_pipe_cache_key
    if _ltx2_pipe is not None:
        del _ltx2_pipe
        _ltx2_pipe = None
        _ltx2_pipe_cache_key = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def initialize_ltx2_pipeline(model_id: str = LTX2_TI2VID_HQ_MODEL, vram_limit: float = 0.0) -> str:  # noqa: ARG001
    """按模型初始化 LTX2 pipeline。"""
    global _ltx2_pipe, _ltx2_pipe_cache_key

    if model_id not in {LTX2_TI2VID_HQ_MODEL, LTX2_A2VID_MODEL}:
        return f"不支持的 LTX2 模型：{model_id}"

    config, config_error = _validate_ltx2_config()
    if config_error:
        return config_error

    _append_ltx2_python_paths(get_ltx2_root())

    cache_key = _build_cache_key(model_id, config)
    if _ltx2_pipe is not None and _ltx2_pipe_cache_key != cache_key:
        clear_ltx2_pipeline()

    if _ltx2_pipe is None:
        try:
            loader_mod = importlib.import_module("ltx_core.loader")

            distilled_lora = [
                loader_mod.LoraPathStrengthAndSDOps(
                    path=str(Path(config["distilled_lora_path"]).expanduser().resolve()),
                    strength=1.0,
                    sd_ops=loader_mod.LTXV_LORA_COMFY_RENAMING_MAP,
                )
            ]

            common_kwargs = dict(
                checkpoint_path=str(Path(config["checkpoint_path"]).expanduser().resolve()),
                distilled_lora=distilled_lora,
                spatial_upsampler_path=str(Path(config["spatial_upsampler_path"]).expanduser().resolve()),
                gemma_root=str(Path(config["gemma_root"]).expanduser().resolve()),
                loras=tuple(),
                quantization=_build_ltx2_quantization_policy(config),
                torch_compile=bool(config.get("torch_compile", False)),
            )

            if model_id == LTX2_TI2VID_HQ_MODEL:
                pipeline_mod = importlib.import_module("ltx_pipelines.ti2vid_two_stages_hq")
                _ltx2_pipe = pipeline_mod.TI2VidTwoStagesHQPipeline(
                    distilled_lora_strength_stage_1=float(config.get("distilled_lora_strength_stage_1", 0.25)),
                    distilled_lora_strength_stage_2=float(config.get("distilled_lora_strength_stage_2", 0.5)),
                    **common_kwargs,
                )
            else:
                pipeline_mod = importlib.import_module("ltx_pipelines.a2vid_two_stage")
                _ltx2_pipe = pipeline_mod.A2VidPipelineTwoStage(**common_kwargs)

            _ltx2_pipe_cache_key = cache_key
        except Exception as exc:  # noqa: BLE001
            clear_ltx2_pipeline()
            return f"LTX2 pipeline 初始化失败：{type(exc).__name__}: {exc}"

    mode_text = "首帧生成视频" if model_id == LTX2_TI2VID_HQ_MODEL else "音频+首帧生成视频"
    return f"Pipeline初始化完成！使用模型: {model_id} (模式: {mode_text})"


def _ensure_valid_size_and_frames(width: int, height: int, num_frames: int) -> str | None:
    if width % 64 != 0 or height % 64 != 0:
        return f"错误：LTX2 模型要求宽高是 64 的倍数，当前为 {width}x{height}"
    if num_frames < 1 or (num_frames - 1) % 8 != 0:
        return f"错误：LTX2 模型要求帧数满足 8k+1，当前为 {num_frames}"
    return None


def _materialize_image_input(frame: Any, tag: str, tmp_paths: list[Path]) -> str | None:
    if frame is None:
        return None

    if isinstance(frame, str):
        path = Path(frame)
        if path.exists():
            return str(path.resolve())
        return None

    if isinstance(frame, Image.Image):
        tmp_file = Path(tempfile.gettempdir()) / f"ltx2_{tag}_{int(time.time() * 1000)}.png"
        frame.save(tmp_file)
        tmp_paths.append(tmp_file)
        return str(tmp_file.resolve())

    return None


def _ensure_ltx2_audio_input(
    audio_file: Path,
    tmp_paths: list[Path],
    target_duration: float,
    front_pad_ratio: float = 50.0,
) -> tuple[Path, list[str]]:
    """
    LTX2 A2Vid 的音频 VAE 输入固定为双声道，且必须与视频窗口匹配。
    音频短于视频时按滑块比例补静音；长于视频时直接报错。
    """
    try:
        import av
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"无法导入 PyAV（{exc}）") from exc

    channels = None
    duration = None
    try:
        container = av.open(str(audio_file))
        try:
            audio_stream = next(stream for stream in container.streams if stream.type == "audio")
            channels = int(audio_stream.channels) if audio_stream.channels is not None else None
            if audio_stream.duration is not None and audio_stream.time_base is not None:
                duration = float(audio_stream.duration * audio_stream.time_base)
            elif container.duration is not None:
                duration = float(container.duration / 1_000_000)
        finally:
            container.close()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"读取音频元信息失败（{exc}）") from exc

    target_duration = float(target_duration)
    if target_duration <= 0:
        raise RuntimeError("视频时长无效，无法处理音频窗口")

    if duration is None:
        selected_duration = target_duration
    else:
        selected_duration = max(0.0, duration)

    if selected_duration <= 0:
        raise RuntimeError("音频为空，请检查音频文件长度")

    tolerance = 0.05
    if selected_duration > target_duration + tolerance:
        raise RuntimeError(
            f"音频时长 {selected_duration:.2f} 秒大于视频时长 {target_duration:.2f} 秒。"
            "请增大视频时长，或换用不长于视频的音频。"
        )

    tmp_audio = Path(tempfile.gettempdir()) / f"ltx2_audio_stereo_{int(time.time() * 1000)}.wav"
    pad_duration = max(0.0, target_duration - selected_duration)
    front_pad_ratio = max(0.0, min(100.0, float(front_pad_ratio or 0.0))) / 100.0
    front_pad = pad_duration * front_pad_ratio
    back_pad = pad_duration - front_pad

    filter_parts = []
    if front_pad > 0:
        filter_parts.append(f"adelay={round(front_pad * 1000)}:all=1")
    # 多留 0.05 秒给音频 VAE 取整，pipeline 仍按视频时长读取。
    filter_parts.append(f"apad=whole_dur={target_duration + 0.05:.3f}")
    filter_parts.append(f"atrim=0:{target_duration + 0.05:.3f}")

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-t",
        f"{selected_duration:.3f}",
        "-i",
        str(audio_file),
        "-vn",
        "-af",
        ",".join(filter_parts),
    ]
    cmd.extend([
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        str(tmp_audio),
    ])

    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        error_text = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"ffmpeg 转换失败（{error_text or '未知错误'}）")

    if not tmp_audio.exists():
        raise RuntimeError("ffmpeg 转换后未生成输出文件")

    tmp_paths.append(tmp_audio)
    notices = []
    if channels == 1:
        notices.append("检测到单声道输入，已自动转换为双声道。")
    elif channels and channels > 2:
        notices.append(f"检测到 {channels} 声道输入，已自动转换为双声道。")
    elif channels != 2:
        notices.append("检测到非常规声道配置，已自动转换为双声道。")

    if pad_duration > tolerance:
        notices.append(
            f"音频时长 {selected_duration:.2f} 秒短于视频时长，"
            f"已补前置静音 {front_pad:.2f} 秒、后置静音 {back_pad:.2f} 秒。"
        )

    return tmp_audio.resolve(), notices


def _is_cuda_oom(exc: Exception) -> bool:
    text = str(exc).lower()
    return isinstance(exc, torch.OutOfMemoryError) or "cuda out of memory" in text


def _build_ltx2_tiling_config(video_vae_mod, profile: str):
    if profile == "none":
        return None
    if profile == "default":
        return video_vae_mod.TilingConfig.default()
    if profile == "aggressive":
        return video_vae_mod.TilingConfig(
            spatial_config=video_vae_mod.SpatialTilingConfig(tile_size_in_pixels=384, tile_overlap_in_pixels=64),
            temporal_config=video_vae_mod.TemporalTilingConfig(tile_size_in_frames=32, tile_overlap_in_frames=8),
        )
    if profile == "extreme":
        return video_vae_mod.TilingConfig(
            spatial_config=video_vae_mod.SpatialTilingConfig(tile_size_in_pixels=256, tile_overlap_in_pixels=64),
            temporal_config=video_vae_mod.TemporalTilingConfig(tile_size_in_frames=16, tile_overlap_in_frames=8),
        )
    return video_vae_mod.TilingConfig.default()


def _run_with_tiling_retry(run_once, width: int, height: int, num_frames: int, tiled: bool) -> None:
    # 1080p + 24fps + 5s 对显存压力很大，默认在 LTX2 路径启用更激进 tiled decode。
    use_auto_tiling = (int(width) * int(height) >= 1280 * 720) or int(num_frames) > 97
    initial_profile = "aggressive" if (tiled or use_auto_tiling) else "none"

    try:
        run_once(initial_profile)
    except Exception as first_exc:  # noqa: BLE001
        if not _is_cuda_oom(first_exc):
            raise
        if initial_profile == "extreme":
            raise

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        run_once("extreme")


def _build_images(args_mod, first_frame_path: str, last_frame_path: str | None, num_frames: int, image_strength: float, image_crf: int):
    images = [args_mod.ImageConditioningInput(first_frame_path, 0, image_strength, image_crf)]
    if last_frame_path:
        images.append(args_mod.ImageConditioningInput(last_frame_path, int(num_frames) - 1, image_strength, image_crf))
    return images


def _format_ltx2_exception(exc: Exception, traceback_text: str) -> tuple[None, str]:
    if _is_cuda_oom(exc):
        suggestion = (
            "建议：缩短视频时长或降低 FPS（例如 16），并保持/启用 Tiled VAE。"
            "当前实现已自动尝试更激进的 tiled decode 兜底。"
        )
        return None, (
            f"LTX2 生成过程中出现显存不足：{exc}\n\n"
            f"{suggestion}\n\n详细错误信息：\n{traceback_text}"
        )
    return None, f"LTX2 生成过程中出现错误：{exc}\n\n详细错误信息：\n{traceback_text}"


def _get_ltx2_runtime_options(config: dict) -> dict:
    return {
        "streaming_prefetch_count": config.get("streaming_prefetch_count"),
        "max_batch_size": max(1, int(config.get("max_batch_size") or 1)),
    }


def process_ltx2_video(
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
    model_id=LTX2_TI2VID_HQ_MODEL,
    first_frame=None,
    last_frame=None,
    audio_path=None,
    tiled=False,
    save_folder_path="./outputs",
    cfg_scale=1.0,
    sigma_shift=5.0,
    audio_front_pad_ratio=50.0,
):
    """使用 LTX2 TI2Vid-HQ 处理首帧（可选尾帧）视频生成。"""
    del quality, vram_limit, save_folder_path, sigma_shift, audio_path, audio_front_pad_ratio

    if first_frame is None:
        return None, "错误：LTX2 模式需要上传首帧图片"

    size_error = _ensure_valid_size_and_frames(int(width), int(height), int(num_frames))
    if size_error:
        return None, size_error

    if seed < 0:
        seed = random.randint(1, 2**32 - 1)

    init_status = initialize_ltx2_pipeline(model_id=model_id)
    if "完成" not in init_status:
        return None, init_status

    if not prompt:
        prompt = "cinematic scene, detailed, high quality"

    _append_ltx2_python_paths(get_ltx2_root())

    tmp_paths: list[Path] = []
    try:
        first_frame_path = _materialize_image_input(first_frame, "first", tmp_paths)
        last_frame_path = _materialize_image_input(last_frame, "last", tmp_paths)
        if not first_frame_path:
            return None, "错误：首帧图片路径无效"

        args_mod = importlib.import_module("ltx_pipelines.utils.args")
        constants_mod = importlib.import_module("ltx_pipelines.utils.constants")
        guiders_mod = importlib.import_module("ltx_core.components.guiders")
        media_io_mod = importlib.import_module("ltx_pipelines.utils.media_io")
        video_vae_mod = importlib.import_module("ltx_core.model.video_vae")

        hq_params = constants_mod.LTX_2_3_HQ_PARAMS
        if not negative_prompt:
            negative_prompt = constants_mod.DEFAULT_NEGATIVE_PROMPT

        config = get_ltx2_ti2vid_hq_config()
        runtime_options = _get_ltx2_runtime_options(config)
        image_strength = float(config.get("image_strength", 1.0))
        image_crf = int(config.get("image_crf", 33))
        resolved_cfg_scale = float(cfg_scale) if cfg_scale is not None else float(hq_params.video_guider_params.cfg_scale)

        images = _build_images(args_mod, first_frame_path, last_frame_path, int(num_frames), image_strength, image_crf)

        video_defaults = hq_params.video_guider_params
        audio_defaults = hq_params.audio_guider_params
        video_guider_params = guiders_mod.MultiModalGuiderParams(
            cfg_scale=resolved_cfg_scale,
            stg_scale=video_defaults.stg_scale,
            rescale_scale=video_defaults.rescale_scale,
            modality_scale=video_defaults.modality_scale,
            skip_step=video_defaults.skip_step,
            stg_blocks=list(video_defaults.stg_blocks),
        )
        audio_guider_params = guiders_mod.MultiModalGuiderParams(
            cfg_scale=audio_defaults.cfg_scale,
            stg_scale=audio_defaults.stg_scale,
            rescale_scale=audio_defaults.rescale_scale,
            modality_scale=audio_defaults.modality_scale,
            skip_step=audio_defaults.skip_step,
            stg_blocks=list(audio_defaults.stg_blocks),
        )

        output_path = f"output_video_{seed}_{int(time.time())}.mp4"

        def _run_once(tiling_profile: str):
            tiling_config = _build_ltx2_tiling_config(video_vae_mod, tiling_profile)
            video_chunks_number = video_vae_mod.get_video_chunks_number(int(num_frames), tiling_config)
            with torch.no_grad():
                video, audio = _ltx2_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=int(seed),
                    height=int(height),
                    width=int(width),
                    num_frames=int(num_frames),
                    frame_rate=float(fps),
                    num_inference_steps=int(num_inference_steps),
                    video_guider_params=video_guider_params,
                    audio_guider_params=audio_guider_params,
                    images=images,
                    tiling_config=tiling_config,
                    **runtime_options,
                )
                media_io_mod.encode_video(
                    video=video,
                    fps=int(fps),
                    audio=audio,
                    output_path=output_path,
                    video_chunks_number=video_chunks_number,
                )

        _run_with_tiling_retry(_run_once, int(width), int(height), int(num_frames), bool(tiled))
        clean_temp_videos()
        return output_path, f"视频生成成功！已保存为 {output_path}"
    except Exception as exc:  # noqa: BLE001
        import traceback

        return _format_ltx2_exception(exc, traceback.format_exc())
    finally:
        for tmp_path in tmp_paths:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def process_ltx2_a2vid(
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
    model_id=LTX2_A2VID_MODEL,
    first_frame=None,
    last_frame=None,
    audio_path=None,
    tiled=False,
    save_folder_path="./outputs",
    cfg_scale=1.0,
    sigma_shift=5.0,
    audio_front_pad_ratio=50.0,
):
    """使用 LTX2 A2Vid 处理音频+首帧（可选尾帧）视频生成。"""
    del quality, vram_limit, save_folder_path, sigma_shift

    if first_frame is None:
        return None, "错误：LTX2-A2Vid 需要上传首帧图片"
    if not audio_path:
        return None, "错误：LTX2-A2Vid 需要上传音频文件"

    audio_file = Path(audio_path)
    if not audio_file.exists():
        return None, f"错误：音频文件不存在 {audio_path}"

    size_error = _ensure_valid_size_and_frames(int(width), int(height), int(num_frames))
    if size_error:
        return None, size_error

    if seed < 0:
        seed = random.randint(1, 2**32 - 1)

    if not prompt:
        prompt = "cinematic scene, detailed, high quality"

    _append_ltx2_python_paths(get_ltx2_root())

    tmp_paths: list[Path] = []
    try:
        first_frame_path = _materialize_image_input(first_frame, "first", tmp_paths)
        last_frame_path = _materialize_image_input(last_frame, "last", tmp_paths)
        if not first_frame_path:
            return None, "错误：首帧图片路径无效"

        video_duration = float(num_frames) / float(fps)
        prepared_audio_file, audio_notices = _ensure_ltx2_audio_input(
            audio_file=audio_file,
            tmp_paths=tmp_paths,
            target_duration=video_duration,
            front_pad_ratio=float(audio_front_pad_ratio or 0.0),
        )

        init_status = initialize_ltx2_pipeline(model_id=model_id)
        if "完成" not in init_status:
            return None, init_status

        args_mod = importlib.import_module("ltx_pipelines.utils.args")
        constants_mod = importlib.import_module("ltx_pipelines.utils.constants")
        guiders_mod = importlib.import_module("ltx_core.components.guiders")
        media_io_mod = importlib.import_module("ltx_pipelines.utils.media_io")
        video_vae_mod = importlib.import_module("ltx_core.model.video_vae")

        params = constants_mod.LTX_2_3_PARAMS
        if not negative_prompt:
            negative_prompt = constants_mod.DEFAULT_NEGATIVE_PROMPT

        config = get_ltx2_ti2vid_hq_config()
        runtime_options = _get_ltx2_runtime_options(config)
        image_strength = float(config.get("image_strength", 1.0))
        image_crf = int(config.get("image_crf", 33))
        resolved_cfg_scale = float(cfg_scale) if cfg_scale is not None else float(params.video_guider_params.cfg_scale)

        images = _build_images(args_mod, first_frame_path, last_frame_path, int(num_frames), image_strength, image_crf)

        video_defaults = params.video_guider_params
        video_guider_params = guiders_mod.MultiModalGuiderParams(
            cfg_scale=resolved_cfg_scale,
            stg_scale=video_defaults.stg_scale,
            rescale_scale=video_defaults.rescale_scale,
            modality_scale=video_defaults.modality_scale,
            skip_step=video_defaults.skip_step,
            stg_blocks=list(video_defaults.stg_blocks),
        )

        output_path = f"output_video_{seed}_{int(time.time())}.mp4"

        def _run_once(tiling_profile: str):
            tiling_config = _build_ltx2_tiling_config(video_vae_mod, tiling_profile)
            video_chunks_number = video_vae_mod.get_video_chunks_number(int(num_frames), tiling_config)
            with torch.no_grad():
                video, audio = _ltx2_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=int(seed),
                    height=int(height),
                    width=int(width),
                    num_frames=int(num_frames),
                    frame_rate=float(fps),
                    num_inference_steps=int(num_inference_steps),
                    video_guider_params=video_guider_params,
                    images=images,
                    audio_path=str(prepared_audio_file),
                    audio_start_time=0.0,
                    audio_max_duration=video_duration,
                    tiling_config=tiling_config,
                    **runtime_options,
                )
                media_io_mod.encode_video(
                    video=video,
                    fps=int(fps),
                    audio=audio,
                    output_path=output_path,
                    video_chunks_number=video_chunks_number,
                )

        _run_with_tiling_retry(_run_once, int(width), int(height), int(num_frames), bool(tiled))
        clean_temp_videos()
        success_msg = f"视频生成成功！已保存为 {output_path}"
        if audio_notices:
            success_msg = f"{success_msg}\n注：{' '.join(audio_notices)}"
        return output_path, success_msg
    except Exception as exc:  # noqa: BLE001
        import traceback

        return _format_ltx2_exception(exc, traceback.format_exc())
    finally:
        for tmp_path in tmp_paths:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
