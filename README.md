# 首尾帧视频生成器使用说明

这是一个基于 Gradio 的首尾帧视频生成 Web 界面。当前视频生成模块仅保留首尾帧模式，并且模型选项仅保留 **AnisoraV3.2**。

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/modelscope/DiffSynth-Studio.git
pip install -e DiffSynth-Studio
python main.py
```

默认启动地址：`0.0.0.0:7861`。

## 界面功能

### 输入设置
- **首帧图片**：必需，用于定义视频起始画面。
- **尾帧图片**：可选，用于定义视频结束画面；不上传时只基于首帧生成。
- **图片信息**：上传后自动显示尺寸、格式等信息。

### 参数设置
- **模型**：当前仅支持 `AnisoraV3.2`。
- **正面提示词 / 负面提示词**：控制生成内容与规避内容。
- **随机种子**：`-1` 表示随机种子。
- **FPS / 质量 / 视频帧数 / 推理步数**：控制输出帧率、编码质量、视频长度和生成质量。
- **视频尺寸**：支持预设宽高比，也支持手动输入宽高；建议使用 64 的倍数。
- **显存占用量限制 / Tiled VAE Decode**：用于平衡显存占用和生成性能。
- **保存地址**：每次生成会创建时间戳子目录，保存视频、输入图片和任务参数 JSON。

## 操作流程

1. 上传首帧图片。
2. 可选上传尾帧图片。
3. 填写提示词并调整生成参数。
4. 选择输出目录。
5. 点击“生成视频”提交到后台队列。
6. 在“视频预览”标签页刷新并查看生成结果。

## 文件结构

```text
wanvace/
├── main.py                  # Gradio 界面入口
├── utils/
│   ├── video_processor.py   # AnisoraV3.2 首尾帧生成逻辑
│   ├── task_queue.py        # 后台队列与子进程调度
│   ├── model_config.py      # 模型列表与宽高比配置
│   └── ...
├── requirements.txt
└── README.md
```

运行产物会写入 `./outputs/` 和 `./task_queue/`，这两个目录不应提交到 Git。

## 验证

```bash
python -m py_compile main.py utils/*.py
python main.py
```

启动后至少验证：上传首帧、可选尾帧、任务入队、输出目录中生成视频和参数文件。
