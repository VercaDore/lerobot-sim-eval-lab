# Setup & Troubleshooting Notes (Windows 11 + RTX 4070)

Environment: Windows 11 x64, RTX 4070 Laptop GPU (8GB, driver 595.79 / CUDA 13.2), Python 3.10.22, `torch==2.10.0+cu126`, `lerobot==0.4.4`, `gym-pusht==0.1.8`.

## 1. System Python 3.13 -> Isolated Python 3.10

System default Python was 3.13.2 (missing prebuilt wheels for parts of the robotics/video stack on Windows). Used `uv` to create a standalone Python 3.10 venv:

```powershell
python -m pip install --user uv
uv venv .venv --python 3.10
uv pip install --python .\.venv\Scripts\python.exe `
  --extra-index-url https://download.pytorch.org/whl/cu126 `
  --index-strategy unsafe-best-match `
  -r requirements.txt
```

## 2. `FileNotFoundError: Could not find 'policy_preprocessor.json'`

Error when loading `lerobot/diffusion_pusht` with `lerobot==0.4.4`:

```text
WARNING:root:Unexpected key(s) when loading model: ['normalize_inputs.buffer_observation_image.mean', ...]
FileNotFoundError: Could not find 'policy_preprocessor.json' on the HuggingFace Hub at 'lerobot/diffusion_pusht'
```

Cause: `lerobot/diffusion_pusht` is a v0.3 checkpoint whose normalization buffers are stored inside `model.safetensors`, whereas v0.4 expects `policy_preprocessor.json`.

Fix: Catch `FileNotFoundError`, load `model.safetensors` via `safetensors.torch.load_file`, extract `mean`/`std`/`min`/`max` buffers into a `dataset_stats` dict, and pass it to `make_pre_post_processors(policy.config, dataset_stats=legacy_stats)` (see `_extract_legacy_stats` in `scripts/eval_pusht.py`).

## 3. `torchcodec` unavailable on Windows

Warning on `LeRobotDataset("lerobot/pusht")`:

```text
WARNING:root:'torchcodec' is not available in your platform, falling back to 'pyav' as a default decoder
```

Fix: `av` (PyAV) is installed automatically as a fallback. For action/state plotting in `scripts/visualize_dataset.py`, reading `dataset.hf_dataset["action"]` and `dataset.hf_dataset["observation.state"]` directly from Parquet avoids decoding video frames altogether.

## 4. HuggingFace Hub timeouts

Set mirror fallback in environment or let scripts fall back automatically:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```
