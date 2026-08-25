from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import logging

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, obj) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logger(run_dir: str | Path) -> logging.Logger:
    logger = logging.getLogger(f"dexined_sam_cv_{Path(run_dir).name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(Path(run_dir) / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def scale_from_1080(width: int, height: int) -> tuple[float, float, float]:
    sx = width / 1920.0
    sy = height / 1080.0
    return sx, sy, min(sx, sy)


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def bbox_area(box: list[int] | tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(a, b) -> float:
    """Intersection divided by the smaller box area."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    den = min(bbox_area(a), bbox_area(b))
    return inter / den if den > 0 else 0.0


def center_distance(a, b) -> float:
    ax = (a[0] + a[2]) * 0.5
    ay = (a[1] + a[3]) * 0.5
    bx = (b[0] + b[2]) * 0.5
    by = (b[1] + b[3]) * 0.5
    return float(np.hypot(ax - bx, ay - by))


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    u8 = mask.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    grad = cv2.morphologyEx(u8, cv2.MORPH_GRADIENT, kernel)
    return grad > 0


def edge_alignment_score(mask: np.ndarray, edge_binary: np.ndarray, sigma: float) -> float:
    boundary = mask_boundary(mask)
    if not boundary.any():
        return 0.0
    edge_u8 = (edge_binary > 0).astype(np.uint8)
    # distanceTransform gives distance to zero pixels, so make edge pixels zero.
    dist = cv2.distanceTransform(1 - edge_u8, cv2.DIST_L2, 3)
    d = dist[boundary]
    sigma = max(float(sigma), 1e-6)
    return float(np.exp(-(d * d) / (2.0 * sigma * sigma)).mean())
