from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .utils import ensure_dir, scale_from_1080


class Visualizer:
    def draw_prompts(self, image_bgr: np.ndarray, prompts: list[dict], out_path: str | Path) -> None:
        vis = image_bgr.copy()
        _, _, scale = scale_from_1080(image_bgr.shape[1], image_bgr.shape[0])
        thickness = max(1, int(round(scale)))
        point_radius = max(4, int(round(4 * scale)))
        marker_size = max(8, int(round(8 * scale)))
        for p in prompts:
            x1, y1, x2, y2 = p["box"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 180, 0), thickness)
            for x, y in p["positive_points"]:
                cv2.circle(vis, (x, y), point_radius, (0, 255, 0), -1)
            for x, y in p["negative_points"]:
                cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, marker_size, thickness)
        ensure_dir(Path(out_path).parent)
        cv2.imwrite(str(out_path), vis)

    def draw_sam_results(self, image_bgr: np.ndarray, results: list[dict], out_path: str | Path) -> None:
        vis = image_bgr.copy()
        overlay = np.zeros_like(image_bgr)
        _, _, scale = scale_from_1080(image_bgr.shape[1], image_bgr.shape[0])
        thickness = max(1, int(round(scale)))
        rng = np.random.default_rng(20260807)
        for item in results:
            path = item.get("mask_path")
            if not path or not Path(path).exists():
                continue
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            if mask.shape[:2] != image_bgr.shape[:2]:
                mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            m = mask > 0
            color = rng.integers(40, 230, size=3, dtype=np.uint8)
            overlay[m] = color
            box = item.get("sam_bbox")
            if box:
                x1, y1, x2, y2 = box
                cv2.rectangle(vis, (x1, y1), (x2, y2), tuple(int(x) for x in color), thickness)
        mixed = cv2.addWeighted(vis, 0.72, overlay, 0.28, 0)
        ensure_dir(Path(out_path).parent)
        cv2.imwrite(str(out_path), mixed)
