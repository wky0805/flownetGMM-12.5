"""Utilities for motion-based crowd extraction and grouping.

This module provides a reference implementation of the post-processing
described in the README/previous discussion: given a sequence of optical-flow
arrays produced by :mod:`liteflownet3.infer`, it filters the motion vectors
according to magnitude thresholds, groups the remaining pixels into connected
components, and tracks those components across frames while handling merges.

Workflow overview:

1. Compute the magnitude of every optical-flow vector and derive a binary mask
   of “moving” pixels using the supplied thresholds.
2. Extract connected components from the mask to identify candidate crowds
   while filtering out small noisy blobs.
3. Track and merge components across frames via bounding-box overlap and
   centroid proximity, resulting in stable crowd IDs.
4. Render per-frame overlays and optional metadata summaries for downstream
   analysis or visual inspection.

The implementation intentionally uses only the Python standard library and
NumPy so it can run in the default project environment.

中文说明：本模块给出了论文中“基于运动的群体提取与合并”后处理的参考实现。
它读取 :mod:`liteflownet3.infer` 生成的光流，按照阈值筛选移动像素，提取
连通域并在跨帧之间跟踪/合并群体 ID，从而复现论文描述的警民碰撞场景的
分析流程，整个过程只依赖标准库与 NumPy，便于在默认环境中直接运行。

流程拆解（中文）：

1. 计算每个光流向量的模长，根据阈值生成“移动像素”二值掩码。
2. 在掩码上做连通域分析，过滤掉面积较小的噪声区域，得到候选人群。
3. 通过框重叠度和质心距离跨帧关联、合并候选人群，维持稳定的群体 ID。
4. 输出逐帧的可视化叠加图，并按需生成 JSON 元数据以便进一步分析。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ArrayLike = np.ndarray


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_flow(path: Path) -> ArrayLike:
    flow = np.load(path)
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(
            f"Expected optical flow with shape (H, W, 2), got {flow.shape} from {path}"
        )
    return flow


def _load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _compute_motion_mask(
    flow: ArrayLike,
    magnitude_threshold: float,
    magnitude_diff_threshold: float,
    previous_magnitude: Optional[ArrayLike],
) -> Tuple[ArrayLike, ArrayLike]:
    """Return the magnitude and a boolean mask of moving pixels.

    Parameters
    ----------
    flow:
        Optical flow array of shape ``(H, W, 2)``.
    magnitude_threshold:
        Minimum magnitude (in pixels per frame interval) for a pixel to be
        considered moving.
    magnitude_diff_threshold:
        Minimum absolute difference between the current magnitude and the
        reference magnitude.  This mimics the *offset difference threshold*
        described in the paper.
    previous_magnitude:
        Magnitude from the previous analysed step.  When ``None`` every pixel is
        compared against zero, which effectively keeps all pixels above
        ``magnitude_threshold``.

    Returns
    -------
    magnitude, mask
        The magnitude array and the boolean mask of pixels that satisfy both
        thresholds.
    """

    magnitude = np.linalg.norm(flow, axis=2)
    reference = previous_magnitude if previous_magnitude is not None else 0.0
    diff = np.abs(magnitude - reference)
    mask = (magnitude >= magnitude_threshold) & (diff >= magnitude_diff_threshold)
    return magnitude, mask


def _connected_components(mask: ArrayLike) -> List[ArrayLike]:
    """Return connected components of the binary mask.

    Components are returned as boolean masks.  A simple flood-fill is used to
    avoid dependencies on external libraries such as SciPy.
    """

    mask = mask.astype(bool)
    if not mask.any():
        return []

    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: List[ArrayLike] = []
    offsets = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            queue: List[Tuple[int, int]] = [(y, x)]
            component_mask = np.zeros_like(mask, dtype=bool)
            visited[y, x] = True
            component_mask[y, x] = True
            while queue:
                cy, cx = queue.pop()
                for dy, dx in offsets:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        component_mask[ny, nx] = True
                        queue.append((ny, nx))
            components.append(component_mask)
    return components


def _mask_bbox(mask: ArrayLike) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    return x_min, y_min, x_max + 1, y_max + 1


def _mask_area(mask: ArrayLike) -> int:
    return int(mask.sum())


def _mask_centroid(mask: ArrayLike) -> Tuple[float, float]:
    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter_area / float(area_a + area_b - inter_area)


@dataclass
class CrowdGroup:
    group_id: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    area: int
    mean_flow: Tuple[float, float]
    parents: List[int] = field(default_factory=list)
    frames_active: int = 0
    frames_since_update: int = 0

    def update(
        self,
        bbox: Tuple[int, int, int, int],
        centroid: Tuple[float, float],
        area: int,
        mean_flow: Tuple[float, float],
        merged_from: Optional[Iterable[int]] = None,
    ) -> None:
        self.bbox = bbox
        self.centroid = centroid
        self.area = area
        self.mean_flow = mean_flow
        if merged_from:
            for parent in merged_from:
                if parent not in self.parents and parent != self.group_id:
                    self.parents.append(parent)
        self.frames_active += 1
        self.frames_since_update = 0


class CrowdTracker:
    """Track crowd components over time and merge overlapping groups."""

    def __init__(
        self,
        min_area: int = 300,
        iou_merge_threshold: float = 0.2,
        centroid_merge_distance: float = 40.0,
        max_inactive: int = 2,
    ) -> None:
        self.min_area = min_area
        self.iou_merge_threshold = iou_merge_threshold
        self.centroid_merge_distance = centroid_merge_distance
        self.max_inactive = max_inactive
        self.groups: Dict[int, CrowdGroup] = {}
        self.next_id = 1

    def _create_group(
        self,
        bbox: Tuple[int, int, int, int],
        centroid: Tuple[float, float],
        area: int,
        mean_flow: Tuple[float, float],
    ) -> CrowdGroup:
        group = CrowdGroup(
            group_id=self.next_id,
            bbox=bbox,
            centroid=centroid,
            area=area,
            mean_flow=mean_flow,
        )
        group.frames_active = 1
        group.frames_since_update = 0
        self.groups[group.group_id] = group
        self.next_id += 1
        return group

    def _distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def update(self, components: List[ArrayLike], flow: ArrayLike) -> List[CrowdGroup]:
        """Update tracked groups given components from the current frame."""

        updated_ids: List[int] = []
        component_infos: List[Tuple[int, Tuple[int, int, int, int], Tuple[float, float], Tuple[float, float]]]
        for comp_mask in components:
            area = _mask_area(comp_mask)
            if area < self.min_area:
                continue
            bbox = _mask_bbox(comp_mask)
            centroid = _mask_centroid(comp_mask)
            mean_flow = tuple(np.mean(flow[comp_mask], axis=0).tolist())
            component_infos.append((area, bbox, centroid, mean_flow))

        # Sort by area descending to prioritise larger groups when matching.
        component_infos.sort(key=lambda item: item[0], reverse=True)

        for area, bbox, centroid, mean_flow in component_infos:
            matched_groups: List[int] = []
            for group_id, group in self.groups.items():
                iou = _bbox_iou(group.bbox, bbox)
                distance = self._distance(group.centroid, centroid)
                if iou >= self.iou_merge_threshold or distance <= self.centroid_merge_distance:
                    matched_groups.append(group_id)

            if not matched_groups:
                group = self._create_group(bbox, centroid, area, mean_flow)
                updated_ids.append(group.group_id)
                continue

            survivor_id = min(matched_groups)
            survivor = self.groups[survivor_id]
            merged_from = [gid for gid in matched_groups if gid != survivor_id]

            # Union of the survivor bbox, matched bboxes and the current component bbox.
            xs1 = [bbox[0]] + [self.groups[gid].bbox[0] for gid in matched_groups]
            ys1 = [bbox[1]] + [self.groups[gid].bbox[1] for gid in matched_groups]
            xs2 = [bbox[2]] + [self.groups[gid].bbox[2] for gid in matched_groups]
            ys2 = [bbox[3]] + [self.groups[gid].bbox[3] for gid in matched_groups]
            merged_bbox = (min(xs1), min(ys1), max(xs2), max(ys2))

            # Compute a centroid that is biased towards the current component but keeps
            # continuity with tracked groups.
            centroids = [centroid] + [self.groups[gid].centroid for gid in matched_groups]
            merged_centroid = tuple(np.mean(np.array(centroids), axis=0).tolist())

            total_area = int(area + sum(self.groups[gid].area for gid in merged_from))
            flows = [mean_flow] + [self.groups[gid].mean_flow for gid in matched_groups]
            merged_flow = tuple(np.mean(np.array(flows), axis=0).tolist())

            survivor.update(merged_bbox, merged_centroid, total_area, merged_flow, merged_from=merged_from)
            updated_ids.append(survivor_id)

            for gid in merged_from:
                self.groups.pop(gid, None)

        # Age unmatched groups and prune inactive ones.
        for group_id, group in list(self.groups.items()):
            if group_id not in updated_ids:
                group.frames_since_update += 1
                if group.frames_since_update > self.max_inactive:
                    self.groups.pop(group_id)

        return list(self.groups.values())


def _draw_groups(image: Image.Image, groups: Sequence[CrowdGroup]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 127, 0),
        (127, 0, 255),
    ]
    for idx, group in enumerate(groups):
        color = palette[idx % len(palette)]
        x1, y1, x2, y2 = group.bbox
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        label = f"#{group.group_id}"
        draw.text((x1 + 3, y1 + 3), label, fill=color, font=font)
    return output


def analyse_sequence(
    frame_paths: Sequence[Path],
    flow_paths: Sequence[Path],
    magnitude_threshold: float,
    magnitude_diff_threshold: float,
    min_area: int,
    output_dir: Path,
    dump_metadata: bool = True,
) -> None:
    """Analyse a sequence of frames with corresponding optical flow arrays."""

    if len(flow_paths) != len(frame_paths):
        raise ValueError(
            "The number of flow arrays must match the number of frames."  # pragma: no cover - sanity check
        )

    _ensure_dir(output_dir)

    tracker = CrowdTracker(min_area=min_area)
    previous_magnitude: Optional[ArrayLike] = None
    metadata: List[Dict[str, object]] = []

    for idx, (frame_path, flow_path) in enumerate(zip(frame_paths, flow_paths)):
        frame = _load_image(frame_path)
        flow = _load_flow(flow_path)
        magnitude, mask = _compute_motion_mask(
            flow,
            magnitude_threshold=magnitude_threshold,
            magnitude_diff_threshold=magnitude_diff_threshold,
            previous_magnitude=previous_magnitude,
        )
        previous_magnitude = magnitude

        components = _connected_components(mask)
        groups = tracker.update(components, flow)

        visualised = _draw_groups(frame, groups)
        output_path = output_dir / f"frame_{idx:05d}.png"
        visualised.save(output_path)

        if dump_metadata:
            metadata.append(
                {
                    "frame": idx,
                    "frame_path": str(frame_path),
                    "flow_path": str(flow_path),
                    "groups": [
                        {
                            "id": group.group_id,
                            "bbox": group.bbox,
                            "centroid": group.centroid,
                            "area": group.area,
                            "mean_flow": group.mean_flow,
                            "parents": group.parents,
                            "frames_active": group.frames_active,
                        }
                        for group in groups
                    ],
                }
            )

    if dump_metadata:
        with (output_dir / "crowd_metadata.json").open("w", encoding="utf8") as fh:
            json.dump(metadata, fh, indent=2)


def _glob_sorted(directory: Path, suffix: str) -> List[Path]:
    paths = sorted(directory.glob(f"*.{suffix}"))
    if not paths:
        raise FileNotFoundError(f"No *.{suffix} files found in {directory}")
    return paths


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crowd motion extraction from optical flow results")
    parser.add_argument("frames", type=Path, help="Directory containing the sampled video frames")
    parser.add_argument("flows", type=Path, help="Directory containing .npy optical flow arrays")
    parser.add_argument("output", type=Path, help="Directory to write annotated frames and metadata")
    parser.add_argument(
        "--magnitude-threshold",
        type=float,
        default=10.0,
        help="Minimum motion magnitude for a pixel to be considered moving",
    )
    parser.add_argument(
        "--magnitude-diff-threshold",
        type=float,
        default=1.0,
        help="Minimum change in magnitude between updates",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=300,
        help="Minimum connected-component area to keep",
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Do not write the JSON metadata file",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    frame_paths = _glob_sorted(args.frames, "png")
    flow_paths = _glob_sorted(args.flows, "npy")
    if len(frame_paths) != len(flow_paths):
        raise ValueError(
            "Number of frame images and flow arrays does not match. "
            f"Found {len(frame_paths)} frames and {len(flow_paths)} flows"
        )
    analyse_sequence(
        frame_paths,
        flow_paths,
        magnitude_threshold=args.magnitude_threshold,
        magnitude_diff_threshold=args.magnitude_diff_threshold,
        min_area=args.min_area,
        output_dir=args.output,
        dump_metadata=not args.skip_metadata,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
