import os
import gradio as gr

from utils.video_utils import get_video_info
from utils.img_utils import get_image_info
from utils.vram_utils import clear_vram, get_vram_info
from utils.preview_utils import refresh_preview_list, load_task_preview
from utils.model_config import (
    VACE_MODELS, INP_MODELS, ANIMATE_MODELS, 
    ASPECT_RATIOS_14b, get_models_by_mode
)
from utils.task_queue import enqueue_task, start_task_worker

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def handle_tab_change(evt: gr.SelectData):
    """处理Tab切换事件"""
    # evt.index: 0 = VACE模式, 1 = 首尾帧模式, 2 = Animate模式
    if evt.index == 0:  # VACE模式
        models = VACE_MODELS
        default_model = VACE_MODELS[0]
    elif evt.index == 1:  # 首尾帧模式
        models = INP_MODELS
        default_model = INP_MODELS[0]
    else:  # Animate模式
        models = ANIMATE_MODELS
        default_model = ANIMATE_MODELS[0]
    
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


def create_preview_tab():
    """创建视频预览标签页"""
    with gr.Column():
        gr.Markdown("## 📹 视频预览")
        gr.Markdown("预览已生成的视频及其参数信息 - 点击缩略图选择任务")
        
        with gr.Row():
            preview_output_dir = gr.Textbox(
                label="输出目录",
                value="./outputs",
                placeholder="./outputs 或 /path/to/output/folder",
                info="指定视频输出目录路径",
                scale=3
            )
            refresh_btn = gr.Button("🔄 刷新列表", variant="primary", scale=1)
        
        # 初始化任务缩略图列表
        initial_gallery, initial_idx = refresh_preview_list("./outputs")
        
        task_gallery = gr.Gallery(
            label="任务缩略图（点击缩略图选择要预览的任务）",
            value=initial_gallery,
            show_label=True,
            elem_id="task_gallery",
            columns=4,
            rows=2,
            height="auto",
            allow_preview=True,
            interactive=True
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                preview_video = gr.Video(
                    label="生成的视频",
                    height=400
                )
            
            with gr.Column(scale=1):
                params_summary = gr.Markdown(
                    value="点击缩略图后显示参数信息",
                    label="参数摘要"
                )
        
        with gr.Row():
            with gr.Column():
                task_json_display = gr.Code(
                    label="任务详情 (JSON)",
                    language="json",
                    lines=15,
                    interactive=False
                )
        
        # 刷新列表事件
        def refresh_list(output_dir):
            gallery_items, default_idx = refresh_preview_list(output_dir)
            return gr.Gallery(value=gallery_items)
        
        refresh_btn.click(
            fn=refresh_list,
            inputs=[preview_output_dir],
            outputs=[task_gallery]
        )
        
        preview_output_dir.submit(
            fn=refresh_list,
            inputs=[preview_output_dir],
            outputs=[task_gallery]
        )
        
        # 加载预览事件 - Gallery组件返回选中的索引
        def load_preview(evt: gr.SelectData, output_dir):
            if evt is None or evt.index is None:
                return None, "请先刷新列表并选择任务", "{}"
            
            selected_index = evt.index
            video_path, params_summary_text, task_json = load_task_preview(selected_index, output_dir)
            return video_path, params_summary_text, task_json
        
        task_gallery.select(
            fn=load_preview,
            inputs=[preview_output_dir],
            outputs=[preview_video, params_summary, task_json_display]
        )


def create_interface():
    """创建Gradio界面"""
    with gr.Blocks(title="WanVACE 视频生成器", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎬 WanVACE 视频生成器")
        gr.Markdown("使用Wan2.1-VACE-14B模型生成高质量视频")
        
        # 主Tabs：视频生成和视频预览
        with gr.Tabs() as main_tabs:
            with gr.TabItem("🎬 视频生成", id="generate_tab"):
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
                                    value="16:9",
                                    info="选择预设的宽高比，系统会自动计算对应的尺寸\n当前尺寸: 1280 × 720"
                                )
                            
                            with gr.TabItem("🔧 手动设置", id="manual_size_tab"):
                                with gr.Row():
                                    width = gr.Number(
                                        label="视频宽度",
                                        value=1280,
                                        minimum=256,
                                        maximum=1280,
                                        step=64,
                                        info="视频宽度（像素）"
                                    )
                                    height = gr.Number(
                                        label="视频高度",
                                        value=720,
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
                            value=10,
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
                                info="视频总帧数，建议时长（秒）*FPS+1"
                            )
                            num_inference_steps = gr.Number(
                                label="推理步数",
                                value=15,
                                minimum=1,
                                maximum=100,
                                step=1,
                                info="推理步数，步数越多质量越高但速度越慢"
                            )
                        
                        with gr.Row():
                            vram_limit = gr.Slider(
                                label="显存占用量限制",
                                value=48.0,
                                minimum=0.0,
                                maximum=200.0,
                                step=1,
                                info="显存占用量限制（GB），影响显存使用和性能"
                            )
                            tiled_checkbox = gr.Checkbox(
                                label="Tiled VAE Decode", 
                                value=False, 
                                info="是否启用 VAE 分块推理。设置为 `True` 时可显著减少 VAE 编解码阶段的显存占用，会产生少许误差，以及少量推理时间延长。"
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
                        
                        output_status = gr.Textbox(
                            label="任务状态",
                            value="点击生成视频按钮提交任务到后台队列",
                            interactive=False,
                            lines=3,
                            info="任务提交后会显示任务ID和队列信息，视频生成完成后会保存到指定的输出目录"
                        )
        
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
                    outputs=[output_status]
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
            
            with gr.TabItem("📹 视频预览", id="preview_tab"):
                create_preview_tab()
    
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
