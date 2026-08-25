from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import AppConfig
from .utils import ensure_dir, save_json


class ImageProcessor:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def run(self, source: str | Path, out_dir: str | Path) -> dict:
        out_dir = ensure_dir(out_dir)
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read input image: {source}")

        src_h, src_w = image.shape[:2]
        dst_w, dst_h = self.cfg.image.target_size
        interpolation = cv2.INTER_AREA if (dst_w < src_w or dst_h < src_h) else cv2.INTER_CUBIC
        resized = cv2.resize(image, (dst_w, dst_h), interpolation=interpolation)
        resized_path = out_dir / "resized.png"
        cv2.imwrite(str(resized_path), resized)

        if self.cfg.image.preprocess_enabled:
            sam_input = self._preprocess(resized)
            sam_input_path = out_dir / "preprocessed.png"
            cv2.imwrite(str(sam_input_path), sam_input)
        else:
            sam_input = resized.copy()
            sam_input_path = resized_path

        report = {
            "source": str(source),
            "source_size": [src_w, src_h],
            "target_size": [dst_w, dst_h],
            "preprocess_enabled": bool(self.cfg.image.preprocess_enabled),
            "resized_path": str(resized_path),
            "sam_input_path": str(sam_input_path),
        }
        save_json(out_dir / "image_report.json", report)
        return {**report, "resized_bgr": resized, "sam_input_bgr": sam_input}

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        c = self.cfg.image
        d = max(1, int(c.bilateral_d))
        if d % 2 == 0:
            d += 1
        filtered = cv2.bilateralFilter(
            image,
            d,
            float(c.bilateral_sigma_color),
            float(c.bilateral_sigma_space),
        )

        hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        saturation = np.clip(
            saturation.astype(np.float32) * float(c.saturation_gain) + float(c.saturation_bias),
            0,
            255,
        ).astype(np.uint8)
        grid = tuple(int(x) for x in c.clahe_tile_grid_size)
        clahe = cv2.createCLAHE(clipLimit=float(c.clahe_clip_limit), tileGridSize=grid)
        value = clahe.apply(value)
        enhanced = cv2.merge([hue, saturation, value])
        return cv2.cvtColor(enhanced, cv2.COLOR_HSV2BGR)
