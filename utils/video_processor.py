"""
视频处理模块
包含视频生成、pipeline初始化、模板视频预处理等核心逻辑
"""
import random
import torch
import gc
import os
import shutil
import tempfile
import time
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

from utils.video_utils import clean_temp_videos, reencode_video_to_16fps
from utils.vram_utils import clear_vram
from utils.animate.preprocess.process_pipepline import ProcessPipeline
from utils.model_config import (
    VACE_MODELS, INP_MODELS, ANIMATE_MODELS
)

# 全局变量存储pipeline和模型选择
pipe: WanVideoPipeline = None
selected_model = "PAI/Wan2.2-VACE-Fun-A14B"  # 默认选择14B模型
input_mode = "vace"  # 默认输入模式：vace（深度视频+参考图片）或 inp（首尾帧）
last_used_model = None  # 记录上一次处理时使用的模型，用于判断是否需要清理显存


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
        if 'temp_dir' in locals():
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
        if model_id in INP_MODELS:
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
        elif model_id == "Wan-AI/Wan2.2-I2V-A14B":
            print(f"正在初始化Wan2.2 I2V模型: {model_id}")
            # Wan2.2 I2V模型配置
            pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
                    ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
                    ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
                    ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
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
        elif model_id == "AnisoraV3.2":
            # 14B Animate模型配置（与test.py保持一致）
            pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(path=[
                        "/home/arkstone/workspace/anisora-models/V3.2/high_noise_model/model_part1.safetensors",
                        "/home/arkstone/workspace/anisora-models/V3.2/high_noise_model/model_part2.safetensors",
                    ], offload_device="cpu"),
                    ModelConfig(path=[
                        "/home/arkstone/workspace/anisora-models/V3.2/low_noise_model/model_part1.safetensors",
                        "/home/arkstone/workspace/anisora-models/V3.2/low_noise_model/model_part2.safetensors",
                    ], offload_device="cpu"),
                    ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
                    ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
                    ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
                ],
            )
        elif model_id == "AnisoraV3.1":
            # 14B Animate模型配置（与test.py保持一致）
            pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(path=[
                        "/home/arkstone/workspace/anisora-models/V3.1/model_part1.safetensors",
                        "/home/arkstone/workspace/anisora-models/V3.1/model_part2.safetensors",
                    ], offload_device="cpu"),
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
    tiled=False,
    animate_reference_image=None,
    template_video=None,
    save_folder_path="./outputs",
    cfg_scale=1.0,
    sigma_shift=5.0
):
    """处理视频生成"""
    global pipe, last_used_model
    try:
        # 根据模型类型判断输入模式
        is_inp_mode = model_id in INP_MODELS
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
                    tiled=tiled,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    cfg_scale=cfg_scale,
                    sigma_shift=sigma_shift,
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
                tiled=tiled,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                cfg_scale=cfg_scale,
                sigma_shift=sigma_shift,
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
                cfg_scale=cfg_scale,
                sigma_shift=sigma_shift,
                vace_scale=0.8,
            )
        
        timestamp = int(time.time())
        output_path = f"output_video_{seed}_{timestamp}.mp4"
        save_video(video, output_path, fps=fps, quality=quality)

        print("cleaning temp videos...")
        clean_temp_videos()
        
        # 只有当模型切换时才清理显存
        if last_used_model is not None and last_used_model != model_id:
            print(f"检测到模型切换（{last_used_model} -> {model_id}），清理显存...")
            clear_vram(pipe)
        else:
            print("使用相同模型，跳过显存清理")
        
        # 更新上一次使用的模型
        last_used_model = model_id
        
        # 不再在此处保存/复制；统一由后台线程在任务成功后剪切
        return output_path, f"视频生成成功！已保存为 {output_path}"
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return None, f"生成过程中出现错误：{str(e)}\n\n详细错误信息：\n{error_trace}"

