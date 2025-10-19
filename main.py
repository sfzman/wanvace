import random
import torch
import gradio as gr
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import time
import os
import gc

from video_utils import clean_temp_videos, reencode_video_to_16fps, get_video_info
from img_utils import get_image_info
from vram_utils import clear_vram, get_vram_info

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
    "Wan-AI/Wan2.1-VACE-1.3B"
]

INP_MODELS = [
    "PAI/Wan2.2-Fun-A14B-InP",
    "PAI/Wan2.1-Fun-V1.1-1.3B-InP"
]

def get_models_by_mode(mode):
    """根据输入模式返回对应的模型列表"""
    if mode == "vace":
        return VACE_MODELS
    elif mode == "inp":
        return INP_MODELS
    else:
        return VACE_MODELS  # 默认返回VACE模型

def handle_tab_change(evt: gr.SelectData):
    """处理Tab切换事件"""
    # evt.index: 0 = VACE模式, 1 = 首尾帧模式
    if evt.index == 0:  # VACE模式
        models = VACE_MODELS
        default_model = VACE_MODELS[0]
        mode = "vace"
    else:  # 首尾帧模式
        models = INP_MODELS
        default_model = INP_MODELS[0]
        mode = "inp"
    
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
        if "InP" in model_id:
            input_mode = "inp"
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
    tiled=True
):
    """处理视频生成"""
    try:
        # 根据模型类型判断输入模式
        is_inp_mode = "InP" in model_id
        
        if is_inp_mode:
            # InP模式：需要首帧，尾帧可选
            has_first_frame = first_frame is not None
            if not has_first_frame:
                return None, "错误：首尾帧模式需要上传首帧图片"
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
                
                # 宽高比选择
                aspect_ratio = gr.Dropdown(
                    label="宽高比",
                    choices=list(ASPECT_RATIOS_14b.keys()),
                    value="16:9_low",
                    info="选择预设的宽高比，或手动设置尺寸"
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
        
        # 宽高比选择变化时自动更新尺寸
        aspect_ratio.change(
            fn=update_dimensions,
            inputs=[aspect_ratio],
            outputs=[height, width]
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
            fn=process_video,
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
                tiled_checkbox
            ],
            outputs=[output_video, output_status]
        )
        
        gr.Markdown("## 📚 使用说明")
        gr.Markdown("""
        1. **选择输入模式**：点击"VACE模式"或"首尾帧模式"标签页
        2. **选择模型**：模型会根据选择的标签页自动更新，选择适合的模型（14B质量更高，1.3B速度更快）
        3. **上传文件**：根据选择的模式上传相应的文件
        4. **设置参数**：调整提示词、种子、FPS、质量、视频尺寸和高级参数
        5. **生成视频**：点击"生成视频"按钮开始处理（首次使用会自动初始化模型）
        
        **标签页与模型对应关系**：
        - **🎬 VACE模式标签页**：显示VACE模型
          - **PAI/Wan2.2-VACE-Fun-A14B**：高质量VACE模型，生成效果更好，但需要更多显存和计算时间
          - **Wan-AI/Wan2.1-VACE-1.3B**：轻量级VACE模型，生成速度更快，显存需求更少，适合快速测试
        - **🖼️ 首尾帧模式标签页**：显示InP模型
          - **PAI/Wan2.2-Fun-A14B-InP**：高质量首尾帧模型，14B参数
          - **PAI/Wan2.1-Fun-V1.1-1.3B-InP**：轻量级首尾帧模型，1.3B参数
        
        **输入模式详细说明**：
        - **VACE模式**：上传深度视频和/或参考图片
          - 可以单独使用深度视频或参考图片
          - 也可以同时使用两者获得更好的效果
          - 深度视频提供运动信息，参考图片提供视觉风格
        - **首尾帧模式**：上传首帧图片（必需）和尾帧图片（可选）
          - 首帧图片是必需的，用于定义视频的起始状态
          - 尾帧图片是可选的，如果提供会生成从首帧到尾帧的过渡视频
          - 如果不提供尾帧，则只使用首帧生成视频
        
        **智能模型切换**：
        - 切换标签页时，模型选择会自动更新为对应模式的模型
        - VACE模式标签页只显示VACE模型（Wan-AI/Wan2.1-VACE-*）
        - 首尾帧模式标签页只显示InP模型（PAI/Wan2.1-Fun-V1.1-*-InP）
        - 系统会自动选择每个模式的默认模型（14B版本）
        
        **视频尺寸**：
        - 选择预设宽高比或手动设置尺寸
        - 支持多种常用宽高比：1:1、4:3、16:9、9:16等
        - 高度和宽度范围：256-1280像素
        - 建议使用64的倍数以获得最佳性能
        - 默认尺寸：832x480（16:9_low，适合大多数显示器）
        
        **注意事项**：
        - VACE模式：至少需要上传深度视频或参考图片中的一种
        - 首尾帧模式：必须上传首帧图片，尾帧图片可选
        - 首次生成时会自动初始化模型，请耐心等待
        - 视频生成需要较长时间，请耐心等待
        - 建议使用较小的视频文件以提高处理速度
        - 较大的视频尺寸会增加处理时间和显存需求
        - 深度视频尺寸应与目标尺寸兼容，系统会自动调整
        - 如果出现尺寸错误，请尝试使用与原始视频相近的宽高比
        - 如果出现VAE解码错误，请尝试禁用"Tiled VAE Decode"选项
        
        **高级参数说明**：
        - **视频帧数**：控制生成视频的长度，帧数越多视频越长
        - **推理步数**：控制生成质量，步数越多质量越高但速度越慢
        - **显存占用量限制**：控制显存使用，数值越大显存占用越多但性能越好（0-100GB）
        - **Tiled VAE Decode**：启用分块VAE解码，可提高性能但可能导致VAE错误
        
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
        - 系统会根据标签页自动显示相应的输入界面和模型选项
        - 切换标签页时会自动更新模型选择，无需手动调整
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        show_error=True
    )
