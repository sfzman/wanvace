# 首尾帧视频生成器使用说明

这是一个基于 Gradio 的首尾帧视频生成 Web 界面。当前支持 **AnisoraV3.2**、**LTX2-TI2Vid-HQ** 和 **LTX2-A2Vid** 三个模型后端。

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/modelscope/DiffSynth-Studio.git
pip install -e DiffSynth-Studio
python main.py
```

默认启动地址：`0.0.0.0:7861`。

## LTX2 阶段 0 探针

在正式接入 LTX2 前，可先运行依赖与导入探针，判断是否可以同进程集成：

```bash
python scripts/phase0_ltx2_probe.py --ltx2-root /home/arkstone/workspace/LTX-2
```

输出 `mode` 含义：
- `inprocess`：当前环境可直接进入 LTX2 后端接入。
- `subprocess_isolated`：建议使用独立 Python 环境通过子进程调用 LTX2。

## LTX2 环境变量

启用 LTX2 系列模型（`LTX2-TI2Vid-HQ` / `LTX2-A2Vid`）需要在 `.env` 配置以下路径：

```bash
LTX2_ROOT=/home/arkstone/workspace/LTX-2
LTX2_CHECKPOINT_PATH=/path/to/ltx2_checkpoint.safetensors
LTX2_DISTILLED_LORA_PATH=/path/to/distilled_lora.safetensors
LTX2_SPATIAL_UPSAMPLER_PATH=/path/to/spatial_upsampler.safetensors
LTX2_GEMMA_ROOT=/path/to/gemma_root
```

更多可选项见 `.env.example`，包括 `LTX2_QUANTIZATION`、`LTX2_STREAMING_PREFETCH_COUNT`、
`LTX2_MAX_BATCH_SIZE` 和 `LTX2_TORCH_COMPILE` 等性能/显存开关。

## 界面功能

### 输入设置
- **首帧图片**：必需，用于定义视频起始画面。
- **尾帧图片**：可选，用于定义视频结束画面；不上传时只基于首帧生成。
- **图片信息**：上传后自动显示尺寸、格式等信息。

### 参数设置
- **模型**：支持 `AnisoraV3.2`、`LTX2-TI2Vid-HQ` 和 `LTX2-A2Vid`。
- **正面提示词 / 负面提示词**：控制生成内容与规避内容。
- **随机种子**：`-1` 表示随机种子。
- **Motion Score (Anisora)**：放在正面提示词下方，仅 Anisora 生效；会自动追加到正向提示词模板。
- **高级参数块（默认折叠）**：统一放置 `FPS / 推理步数 / CFG Scale / Sigma Shift / 显存限制 / Tiled VAE`。
- **默认 FPS**：按模型切换（`AnisoraV3.2=16`，`LTX2-TI2Vid-HQ=24`，`LTX2-A2Vid=24`；帧数按 `FPS × 时长 + 1` 自动计算）。
- **模型默认值**：切换模型时自动刷新（例如 TI2Vid 默认推理步数 15、A2Vid 默认推理步数 30、CFG 3.0）。
- **Anisora 提示词追加**：会自动在正向提示词后追加 `aesthetic score: 5.0. motion score: X.X. There is no text in the video.`，其中 `X.X` 由 Motion Score 滑块控制（2.0~5.0，步长 0.5）。
- **A2Vid 静音补齐**：音频短于视频时，可在音频上传区域下方用前置静音比例滑块分配前后静音；音频长于视频会直接入队失败。
- **Tiled VAE 默认值**：LTX2 模型默认开启，降低显存峰值。
- **音频输入**：`LTX2-A2Vid` 模型需要上传音频文件。
- **视频尺寸**：支持预设宽高比，也支持手动输入宽高；LTX2 模型要求宽高必须是 64 的倍数，并会将预设比例自动映射到 1080p 档位近似尺寸。
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
