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

## Crowd analysis post-processing

The repository includes a reference implementation of the paper-style
post-processing pipeline in `liteflownet3/utils/crowd_analysis.py`. It expects
two ordered directories:

1. A folder containing the sampled video frames as PNG files (for example,
   `frame_0011.png`, `frame_0021.png`, …).
2. A folder containing the corresponding optical-flow arrays saved by the
   inference script as `.npy` files (`frame_0011.npy`, etc.).

Each directory is scanned in lexicographical order, so ensure that filenames are
zero-padded consistently.

### CLI workflow (English)

Once the frames and flows are ready, run:

```bash
python -m liteflownet3.utils.crowd_analysis \
  path/to/frames \
  path/to/flows \
  path/to/output \
  --magnitude-threshold 10 \
  --magnitude-diff-threshold 1 \
  --min-area 300
```

- `path/to/frames`: Input frame directory. The script reads files in sorted
  order and uses the original RGB pixels for visualisation.
- `path/to/flows`: Directory containing `.npy` flow tensors produced by
  `liteflownet3.infer`. The filenames must correspond 1:1 with the frames.
- `path/to/output`: Destination folder for annotated PNGs and optional
  metadata.
- `--magnitude-threshold`: Minimum motion magnitude (pixels per frame interval)
  to classify a pixel as moving.
- `--magnitude-diff-threshold`: Extra guard that only keeps pixels whose motion
  magnitude changes by at least the supplied value between updates.
- `--min-area`: Small connected components under this area (in pixels) are
  discarded.
- `--skip-metadata`: When provided, the script will not write the JSON summary.

The command writes annotated PNGs into `path/to/output` (one per input frame)
and, by default, a `crowd_metadata.json` file that records the bounding boxes,
centroids, and ID history of every tracked group.

### 命令行工作流（中文）

准备好帧目录与光流目录后，可运行：

```bash
python -m liteflownet3.utils.crowd_analysis \
  /路径/到/帧文件夹 \
  /路径/到/光流文件夹 \
  /路径/到/输出文件夹 \
  --magnitude-threshold 10 \
  --magnitude-diff-threshold 1 \
  --min-area 300
```

- `/路径/到/帧文件夹`：按帧号排序的 PNG 图片目录，脚本会读取原始像素用于绘制标注。
- `/路径/到/光流文件夹`：与帧一一对应的 `.npy` 光流张量，通常由 `liteflownet3.infer`
  保存。文件名需与帧一致（如 `frame_0011.npy` 对应 `frame_0011.png`）。
- `/路径/到/输出文件夹`：输出目录，脚本会在此写入带标注的 PNG 和 JSON 报告。
- `--magnitude-threshold`：像素被视为“移动”所需的最小运动模长（单位：像素/帧间隔）。
- `--magnitude-diff-threshold`：额外的模长变化阈值，仅保留与上一状态相比变化幅度足够大的像素。
- `--min-area`：最小保留面积，小于该像素数的连通域会被忽略。
- `--skip-metadata`：如果只需要可视化而不需要 JSON 汇总，可加此参数跳过写入。

默认情况下，脚本会在输出目录生成逐帧的 PNG 标注图，并写入 `crowd_metadata.json`
以记录每个群体的框、质心、面积、平均光流和 ID 合并历史；如不需要可使用
`--skip-metadata` 关闭。

## License

This project is released under the MIT license.

