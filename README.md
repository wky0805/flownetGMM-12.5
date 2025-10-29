# LiteFlowNet3 (CUDA 12.4 ready)

This repository provides a PyTorch implementation of **LiteFlowNet3** that
targets modern NVIDIA GPUs (RTX 40-series and newer) using CUDA 12.4 or
higher. The project includes ready-to-run training and inference scripts
and can be easily integrated into existing optical flow workflows.

## Features

- Lightweight yet expressive LiteFlowNet3 architecture implemented in
  idiomatic PyTorch
- Runtime guard to ensure the CUDA toolkit is at least version 12.4 when
  running on GPU
- Training pipeline with automatic mixed precision (AMP) support
- Dataset abstraction for `.flo` ground-truth files paired with RGB
  frames
- Command-line entry points for both training (`liteflownet3/train.py`)
  and inference (`liteflownet3/infer.py`)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy pillow
```

The dedicated PyTorch index URL installs wheels compiled against CUDA 12.4.
If you prefer conda or system packages, ensure that the CUDA runtime
version satisfies the 12.4 requirement.

## Data preparation

Create a text file where each line contains the relative paths to two RGB
images and a `.flo` optical flow file, separated by spaces. Paths are
resolved relative to the list file location.

Example (`train.txt`):

```
images/frame_0001.png images/frame_0002.png flow/frame_0001.flo
images/frame_0002.png images/frame_0003.png flow/frame_0002.flo
```

## Training

```bash
python -m liteflownet3.train train.txt --epochs 300 --batch-size 6 \
  --checkpoint-dir runs/checkpoints
```

Useful flags:

- `--no-amp` disables mixed precision training.
- `--clip-grad` controls gradient norm clipping (`0` disables it).

Checkpoints are written after every epoch inside the specified directory.

## Inference

```bash
python -m liteflownet3.infer image1.png image2.png runs/checkpoints/epoch_0300.pt \
  --output flow.npy
```

The output is saved as a NumPy array with shape `(H, W, 2)`.

## License

This project is released under the MIT license.

