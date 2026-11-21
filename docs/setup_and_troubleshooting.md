# Windows 11 + RTX 4070 环境搭建与排错日志

本文档记录在原生 Windows 11 + NVIDIA RTX 4070 Laptop GPU（8GB VRAM，驱动 595.79 / CUDA 13.2）上从零搭建 **HuggingFace LeRobot (v0.4.4)** 与 **PushT 仿真评测环境** 过程中遇到的真实问题及解决方案。

---

## 1. 环境版本清单

- **OS**: Windows 11 (x86_64)
- **GPU**: NVIDIA GeForce RTX 4070 Laptop GPU (8188 MiB)
- **Python**: `3.10.22`（通过 `uv` 创建独立 `.venv`，与系统自带 Python 3.13 隔离）
- **核心依赖**：
  - `torch==2.10.0+cu126`, `torchvision==0.25.0+cu126`
  - `lerobot==0.4.4`
  - `gym-pusht==0.1.8`, `gymnasium==1.4.0`, `pymunk==6.11.1`, `pygame==2.6.1`
  - `imageio==2.38.0`, `imageio-ffmpeg==0.8.1`, `matplotlib==3.10.9`, `pillow==12.3.0`

---

## 2. 实际踩坑与排错记录

### 问题 1：系统默认 Python 3.13 与具身仿真生态兼容性风险
- **现象**：宿主机仅安装了 Python 3.13.2，无 Conda；部分机器人仿真与音视频解码包在 Python 3.13 上缺乏 Windows 预编译 Wheel。
- **解决**：使用 `uv` 在项目内秒级拉取独立 Python 3.10.22 并建立 `.venv`：
  ```powershell
  python -m pip install --user uv
  uv venv .venv --python 3.10
  ```
- **效果**：107 个依赖包全部命中预编译 Wheel，0 编译报错，总安装耗时 < 2 分钟。

### 问题 2：LeRobot v0.4.4 加载 `lerobot/diffusion_pusht` 报错缺少 `policy_preprocessor.json`
- **现象**：
  ```text
  WARNING:root:Unexpected key(s) when loading model: ['normalize_inputs.buffer_observation_image.mean', ...]
  FileNotFoundError: Could not find 'policy_preprocessor.json' on the HuggingFace Hub at 'lerobot/diffusion_pusht'
  ```
- **根因**：`lerobot/diffusion_pusht` 为 v0.3 时代上传的权重，其输入/输出归一化统计量直接嵌入在 `model.safetensors` 的 buffer 键中；而 LeRobot v0.4 将归一化逻辑抽离为独立的 `PolicyProcessorPipeline`（需读取 `policy_preprocessor.json`）。
- **解决**：在 `scripts/eval_pusht.py` 中新增 `_extract_legacy_stats()` 桥接函数：当检测到缺少 `policy_preprocessor.json` 时，自动通过 `safetensors.torch.load_file` 读取 `model.safetensors` 中的 `mean/std/min/max` 张量，构造标准 `dataset_stats` 字典传入 `make_pre_post_processors(policy.config, dataset_stats=legacy_stats)`。
- **验证**：20 轮 PushT 评测顺利跑通，成功率达到 **90.0%（18/20）**，与官方基准一致。

### 问题 3：Windows 下 `torchcodec` 不可用提示
- **现象**：加载 `LeRobotDataset("lerobot/pusht")` 时提示 `'torchcodec' is not available in your platform, falling back to 'pyav' as a default decoder`。
- **解决**：`lerobot[pusht]` 已自动安装 `av==15.1.0`（PyAV），且在数据分析脚本中直接读取底层 `dataset.hf_dataset` 的 Parquet 数值列（`action`、`observation.state`），完全绕开逐帧视频解码瓶颈，1,347 帧解析与绘图耗时不到 5 秒。

### 问题 4：HuggingFace 网络连接波动
- **解决**：脚本内均实现了自动降级逻辑——直连失败时自动设置 `HF_ENDPOINT=https://hf-mirror.com` 重试，避免长时间卡死。
