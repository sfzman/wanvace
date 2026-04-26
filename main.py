import os
import sys
import signal
import atexit
import threading
import gradio as gr

from utils.img_utils import get_image_info
from utils.vram_utils import clear_vram, get_vram_info
from utils.preview_utils import refresh_preview_list, load_task_preview, delete_task_files
from utils.model_config import INP_MODELS, ASPECT_RATIOS_14b
from utils.task_queue import enqueue_task, start_task_worker, stop_task_worker, kill_worker_subprocess
from utils.app_config import get_output_dir

os.environ["TOKENIZERS_PARALLELISM"] = "false"

GENERATE_SHORTCUT_JS = """
() => {
  if (window.__wanvaceGenerateShortcutBound) {
    return;
  }
  window.__wanvaceGenerateShortcutBound = true;
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      const button = document.querySelector("#generate_video_btn button");
      if (button && !button.disabled) {
        button.click();
      }
    }
  });
}
"""


# ---------------------------------------------------------------------------
# 优雅退出：信号处理 + atexit
# ---------------------------------------------------------------------------

_shutdown_lock = threading.Lock()
_shutdown_done = False


def _graceful_shutdown(reason="unknown"):
    """清理子进程和 CUDA 资源。可被信号处理器和 atexit 安全调用（只执行一次）。"""
    global _shutdown_done
    with _shutdown_lock:
        if _shutdown_done:
            return
        _shutdown_done = True
    print(f"\n[shutdown] 正在清理资源 (原因: {reason})...")
    try:
        stop_task_worker()
    except Exception:
        pass
    try:
        kill_worker_subprocess()
    except Exception:
        pass
    try:
        clear_vram()
    except Exception:
        pass
    print("[shutdown] 清理完成。")


def _signal_handler(signum, frame):
    sig_name = signal.Signals(signum).name
    _graceful_shutdown(reason=sig_name)
    # 启动定时器：如果 10 秒后进程仍未退出，强制终止
    def _force_exit():
        print("[shutdown] 清理超时，强制退出。")
        os._exit(1)
    t = threading.Timer(10.0, _force_exit)
    t.daemon = True
    t.start()
    sys.exit(0)


# 注册信号处理器
for _sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(_sig, _signal_handler)
if hasattr(signal, "SIGHUP"):
    signal.signal(signal.SIGHUP, _signal_handler)

# atexit 兜底（正常退出时也清理）
atexit.register(_graceful_shutdown, reason="atexit")


def handle_clear_vram():
    """释放显存：先终止持有 GPU 的工作子进程，再清理残余缓存。"""
    kill_worker_subprocess()
    return clear_vram()


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
    default_output_dir = get_output_dir()
    with gr.Column():
        gr.Markdown("## 📹 视频预览")
        gr.Markdown("预览已生成的视频及其参数信息 - 点击缩略图选择任务")

        DEFAULT_PARAM_SUMMARY = "点击缩略图后显示参数信息"

        with gr.Row():
            preview_output_dir = gr.Textbox(
                label="输出目录",
                value=default_output_dir,
                placeholder="./outputs 或 /path/to/output/folder",
                info="指定视频输出目录路径",
                scale=3
            )
            refresh_btn = gr.Button("🔄 刷新列表", variant="primary", scale=1)

        # 初始化任务缩略图列表
        initial_gallery, initial_idx, initial_tasks = refresh_preview_list(default_output_dir)
        preview_tasks_state = gr.State(initial_tasks)
        selected_task_index_state = gr.State(None)

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
                    value=DEFAULT_PARAM_SUMMARY,
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

        with gr.Row():
            delete_btn = gr.Button("🗑️ 删除选中任务", variant="stop", scale=1)
            action_status = gr.Textbox(
                label="操作状态",
                value="",
                interactive=False
            )

        # 刷新列表事件
        def refresh_list(output_dir):
            gallery_items, _, tasks = refresh_preview_list(output_dir)
            # 返回更新后的Gallery内容以及任务缓存
            return gr.update(value=gallery_items), tasks

        refresh_btn.click(
            fn=refresh_list,
            inputs=[preview_output_dir],
            outputs=[task_gallery, preview_tasks_state]
        )

        preview_output_dir.submit(
            fn=refresh_list,
            inputs=[preview_output_dir],
            outputs=[task_gallery, preview_tasks_state]
        )

        # 加载预览事件 - Gallery组件返回选中的索引
        def load_preview(evt: gr.SelectData, tasks, output_dir):
            if evt is None or evt.index is None:
                return None, DEFAULT_PARAM_SUMMARY, "{}", None
            if not tasks:
                return None, "当前没有可预览的任务，请先刷新列表", "{}", None

            selected_index = evt.index
            if selected_index >= len(tasks):
                return None, "任务索引已过期，请刷新列表后重试", "{}", None

            video_path, params_summary_text, task_json = load_task_preview(
                selected_index,
                output_dir,
                cached_tasks=tasks,
            )
            return video_path, params_summary_text, task_json, selected_index

        task_gallery.select(
            fn=load_preview,
            inputs=[preview_tasks_state, preview_output_dir],
            outputs=[preview_video, params_summary, task_json_display, selected_task_index_state]
        )

        def delete_selected_task(tasks, selected_index, output_dir):
            if not tasks:
                return (
                    gr.update(),
                    tasks,
                    None,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    "当前没有任务可以删除，请先刷新列表"
                )
            if selected_index is None or selected_index >= len(tasks):
                return (
                    gr.update(),
                    tasks,
                    None,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    "请选择一个要删除的任务"
                )

            task = tasks[selected_index]
            success, message = delete_task_files(task)

            # 无论成功与否，刷新列表保持状态一致
            gallery_items, _, new_tasks = refresh_preview_list(output_dir)

            return (
                gr.update(value=gallery_items),
                new_tasks,
                None if success else selected_index,
                gr.update(value=None) if success else gr.update(),
                gr.update(value=DEFAULT_PARAM_SUMMARY) if success else gr.update(),
                gr.update(value="{}") if success else gr.update(),
                message
            )

        delete_btn.click(
            fn=delete_selected_task,
            inputs=[preview_tasks_state, selected_task_index_state, preview_output_dir],
            outputs=[
                task_gallery,
                preview_tasks_state,
                selected_task_index_state,
                preview_video,
                params_summary,
                task_json_display,
                action_status,
            ]
        )


def create_interface():
    """创建Gradio界面"""
    with gr.Blocks(title="首尾帧视频生成器", theme=gr.themes.Soft(), js=GENERATE_SHORTCUT_JS) as demo:
        gr.Markdown("# 🎬 首尾帧视频生成器")
        gr.Markdown("使用 AnisoraV3.2 模型基于首帧/尾帧生成视频")

        # 主Tabs：视频生成和视频预览
        with gr.Tabs() as main_tabs:
            with gr.TabItem("🎬 视频生成", id="generate_tab"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## 📤 输入设置")

                        first_frame = gr.Image(
                            label="首帧图片 (First Frame)",
                            height=260,
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
                            height=260,
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

                        generate_btn = gr.Button(
                            "🎬 生成视频 (Ctrl+Enter)",
                            variant="primary",
                            size="lg",
                            elem_id="generate_video_btn"
                        )

                        # 模型选择
                        model_id = gr.Dropdown(
                            label="选择模型",
                            choices=INP_MODELS,
                            value=INP_MODELS[0],
                            info="当前仅保留 AnisoraV3.2"
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

                        video_duration = gr.Slider(
                            label="视频时长（秒）",
                            value=5,
                            minimum=5,
                            maximum=10,
                            step=1,
                            info="FPS 固定为 16，帧数自动计算为 FPS × 视频时长 + 1"
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

                        with gr.Row():
                            tiled_checkbox = gr.Checkbox(
                                label="Tiled VAE Decode",
                                value=False,
                                info="是否启用 VAE 分块推理。设置为 `True` 时可显著减少 VAE 编解码阶段的显存占用，会产生少许误差，以及少量推理时间延长。"
                            )

                        output_status = gr.Textbox(
                            label="任务状态",
                            value="点击生成视频按钮提交任务到后台队列",
                            interactive=False,
                            lines=3,
                            info="任务提交后会显示任务ID和队列信息，视频生成完成后会保存到指定的输出目录"
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

                # 图片上传时自动更新图片信息
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
                    fn=handle_clear_vram,
                    outputs=[vram_info]
                )

                refresh_vram_btn.click(
                    fn=get_vram_info,
                    outputs=[vram_info]
                )

                generate_btn.click(
                    fn=enqueue_task,
                    inputs=[
                        prompt,
                        negative_prompt,
                        height,
                        width,
                        video_duration,
                        model_id,
                        first_frame,
                        last_frame,
                        tiled_checkbox,
                    ],
                    outputs=[output_status]
                )

                gr.Markdown("## 📚 使用说明")
                gr.Markdown("""
        1. **上传首帧图片**：首帧图片是必需的，用于定义视频的起始状态
        2. **上传尾帧图片（可选）**：提供尾帧时生成从首帧到尾帧的过渡视频；不提供则只基于首帧生成
        3. **选择模型**：当前仅保留 **AnisoraV3.2**
        4. **设置参数**：调整提示词、视频时长、视频尺寸和高级参数
        5. **保存地址**：从 `.env` 读取（推荐 `WANVACE_OUTPUT_DIR=./outputs`）
        6. **生成视频**：点击"生成视频"按钮提交到后台队列

        **输入要求**：
        - 首帧图片：必需
        - 尾帧图片：可选
        - 视频尺寸建议使用 64 的倍数；较大的尺寸会增加处理时间和显存需求

        **高级参数说明**：
        - **视频时长**：可选 5-10 秒；FPS 固定为 16，帧数自动计算为 FPS×时长+1
        - **推理步数**：固定为 8 步
        - **Tiled VAE Decode**：启用分块VAE解码，可降低 VAE 编解码阶段的显存占用

        **视频保存功能**：
        - 每次生成会在保存目录中创建带时间戳的子文件夹
        - 系统会保存生成视频、输入图片和任务参数 JSON，便于复现和调试

        **显存管理**：
        - 调试中断后点击"释放显存"按钮清理显存
        - 使用"刷新显存信息"查看当前显存使用情况
        - 显存不足时建议先释放显存再重新生成
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
