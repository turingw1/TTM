# A100 Demo Guide

This guide is for reproducing a `Time-to-Move` demo on an A100 80GB server.

Assumptions:
- working directory: `~/workspace/Zhengwei`
- large-file / cache directory: `/cache/Zhengwei`
- network has no proxy
- GitHub access may need the `githubfast` rewrite
- single-GPU demo target: `Wan 2.2 I2V 14B`

The recommended first demo is `Wan`, because this repository explicitly recommends it and it gives the strongest controllability / quality trade-off.

## 0. One-time shell setup

Run these in a fresh shell before cloning or installing from GitHub:

```bash
cd ~/workspace/Zhengwei

export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=url.https://githubfast.com/.insteadOf
export GIT_CONFIG_VALUE_0=https://github.com/
```

Create cache directories on the large disk:

```bash
mkdir -p /cache/Zhengwei/{hf,torch,xdg,ttm_outputs,wheels}
```

Set cache-related environment variables:

```bash
export HF_HOME=/cache/Zhengwei/hf
export HUGGINGFACE_HUB_CACHE=/cache/Zhengwei/hf/hub
export TRANSFORMERS_CACHE=/cache/Zhengwei/hf/transformers
export TORCH_HOME=/cache/Zhengwei/torch
export XDG_CACHE_HOME=/cache/Zhengwei/xdg
export CUDA_VISIBLE_DEVICES=0
```

Optional sanity checks:

```bash
nvidia-smi
python3 --version
```

## 1. Clone the repo

```bash
cd ~/workspace/Zhengwei
git clone https://github.com/turingw1/TTM.git
cd TTM
git remote -v
```

Expected `origin`:

```text
https://github.com/turingw1/TTM.git
```

## 2. Create the Python environment

Use `venv` unless your server already has a managed conda setup you prefer.

```bash
cd ~/workspace/Zhengwei/TTM
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 3. Install PyTorch for CUDA

Install PyTorch first. Use the wheel index matching your CUDA runtime. For a typical CUDA 12.1 server:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch torchvision torchaudio
```

Verify CUDA is visible:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("bf16:", torch.cuda.is_bf16_supported())
PY
```

## 4. Install TTM runtime dependencies

The repo depends on `diffusers`, `transformers`, and common video / image packages.
For the Wan path, installing `diffusers` from GitHub is safer than relying on an older PyPI release.

```bash
cd ~/workspace/Zhengwei/TTM

pip install \
  accelerate \
  transformers \
  sentencepiece \
  safetensors \
  ftfy \
  opencv-python \
  imageio \
  imageio-ffmpeg \
  pillow \
  scipy \
  numpy

pip install "git+https://github.com/huggingface/diffusers.git"
```

Quick import check:

```bash
python - <<'PY'
import torch
from diffusers.utils import load_image
from pipelines.wan_pipeline import WanImageToVideoTTMPipeline
print("Imports OK")
print("CUDA available:", torch.cuda.is_available())
PY
```

## 5. Run the recommended Wan demo

This is the cleanest first reproduction target on an A100 80GB GPU.

### 5.1 Monkey cut-and-drag demo

```bash
cd ~/workspace/Zhengwei/TTM
source .venv/bin/activate

export HF_HOME=/cache/Zhengwei/hf
export HUGGINGFACE_HUB_CACHE=/cache/Zhengwei/hf/hub
export TRANSFORMERS_CACHE=/cache/Zhengwei/hf/transformers
export TORCH_HOME=/cache/Zhengwei/torch
export XDG_CACHE_HOME=/cache/Zhengwei/xdg
export CUDA_VISIBLE_DEVICES=0

python run_wan.py \
  --input-path ./examples/cutdrag_wan_Monkey \
  --output-path /cache/Zhengwei/ttm_outputs/wan_monkey.mp4 \
  --tweak-index 3 \
  --tstrong-index 7 \
  --num-inference-steps 50 \
  --num-frames 81 \
  --guidance-scale 3.5
```

Expected output:

```text
/cache/Zhengwei/ttm_outputs/wan_monkey.mp4
```

### 5.2 Camera-control demo

```bash
python run_wan.py \
  --input-path ./examples/camcontrol_ConcertStage \
  --output-path /cache/Zhengwei/ttm_outputs/wan_concert_stage.mp4 \
  --tweak-index 2 \
  --tstrong-index 5 \
  --num-inference-steps 50 \
  --num-frames 81 \
  --guidance-scale 3.5
```

### 5.3 Notes on the two key hyperparameters

- `tweak-index`
  - when background denoising starts
  - too low: scene deformation / duplication / unintended camera motion
  - too high: background stays too static
- `tstrong-index`
  - when masked-region overwriting stops
  - too low: object may drift from the target path
  - too high: object may become rigid or over-constrained

Repository defaults and good starting points:

- cut-and-drag: `tweak-index=3`, `tstrong-index=7`
- camera control: `tweak-index=2`, `tstrong-index=5`

## 6. Alternative backbones

Use these only after the Wan demo succeeds.

### 6.1 CogVideoX demo

```bash
cd ~/workspace/Zhengwei/TTM
source .venv/bin/activate

python run_cog.py \
  --input-path ./examples/cutdrag_cog_Monkey \
  --output-path /cache/Zhengwei/ttm_outputs/cog_monkey.mp4 \
  --tweak-index 4 \
  --tstrong-index 9 \
  --num-inference-steps 50 \
  --num-frames 49 \
  --guidance-scale 6.0
```

### 6.2 Stable Video Diffusion demo

```bash
cd ~/workspace/Zhengwei/TTM
source .venv/bin/activate

python run_svd.py \
  --input-path ./examples/cutdrag_svd_Fish \
  --output-path /cache/Zhengwei/ttm_outputs/svd_fish.mp4 \
  --tweak-index 16 \
  --tstrong-index 21 \
  --num-inference-steps 50 \
  --num-frames 21 \
  --motion_bucket_id 17
```

## 7. If GitHub access is unstable

Re-export the GitHub rewrite before `git clone` or any `pip install git+https://github.com/...` command:

```bash
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=url.https://githubfast.com/.insteadOf
export GIT_CONFIG_VALUE_0=https://github.com/
```

Then rerun the failed command in the same shell.

## 8. If the first run is slow

This is expected.

The first successful run may spend most of its time on:
- downloading the model from Hugging Face
- building local caches under `/cache/Zhengwei`
- initializing transformer / VAE weights

Subsequent runs should be much faster as long as the cache variables still point to `/cache/Zhengwei`.

## 9. Minimal end-to-end command block

If you just want the shortest reproducible path:

```bash
cd ~/workspace/Zhengwei

export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=url.https://githubfast.com/.insteadOf
export GIT_CONFIG_VALUE_0=https://github.com/

mkdir -p /cache/Zhengwei/{hf,torch,xdg,ttm_outputs}

git clone https://github.com/turingw1/TTM.git
cd TTM

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch torchvision torchaudio

pip install \
  accelerate transformers sentencepiece safetensors ftfy \
  opencv-python imageio imageio-ffmpeg pillow scipy numpy

pip install "git+https://github.com/huggingface/diffusers.git"

export HF_HOME=/cache/Zhengwei/hf
export HUGGINGFACE_HUB_CACHE=/cache/Zhengwei/hf/hub
export TRANSFORMERS_CACHE=/cache/Zhengwei/hf/transformers
export TORCH_HOME=/cache/Zhengwei/torch
export XDG_CACHE_HOME=/cache/Zhengwei/xdg
export CUDA_VISIBLE_DEVICES=0

python run_wan.py \
  --input-path ./examples/cutdrag_wan_Monkey \
  --output-path /cache/Zhengwei/ttm_outputs/wan_monkey.mp4 \
  --tweak-index 3 \
  --tstrong-index 7 \
  --num-inference-steps 50 \
  --num-frames 81 \
  --guidance-scale 3.5
```

## 10. Recommended result files to check

After a successful run, check:

```bash
ls -lh /cache/Zhengwei/ttm_outputs
```

Typical target files:
- `wan_monkey.mp4`
- `wan_concert_stage.mp4`
- `cog_monkey.mp4`
- `svd_fish.mp4`
