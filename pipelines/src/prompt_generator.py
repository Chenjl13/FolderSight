from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import AppConfig
from .utils import bbox_iou, containment, ensure_dir, save_json, scale_from_1080


class PromptGenerator:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def generate_candidates(self, image_bgr: np.ndarray, out_dir: str | Path) -> dict:
        out_dir = ensure_dir(out_dir)
        h, w = image_bgr.shape[:2]
        sx, sy, s = scale_from_1080(w, h)
        c = self.cfg.opencv_candidates

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([c.hue_low, c.saturation_low, c.value_low], dtype=np.uint8)
        upper = np.array([c.hue_high, c.saturation_high, c.value_high], dtype=np.uint8)
        yellow_mask = cv2.inRange(hsv, lower, upper)

        kernel_size = max(1, int(round(c.morph_kernel_1080 * s)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if c.morph_open_iterations > 0:
            yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel, iterations=int(c.morph_open_iterations))
        if c.morph_close_iterations > 0:
            yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel, iterations=int(c.morph_close_iterations))
        if c.morph_dilate_iterations > 0:
            yellow_mask = cv2.dilate(yellow_mask, kernel, iterations=int(c.morph_dilate_iterations))

        min_area = max(1, int(round(c.min_area_1080 * sx * sy)))
        max_area = max(min_area + 1, int(round(c.max_area_1080 * sx * sy)))
        min_w = max(1, int(round(c.min_width_1080 * sx)))
        min_h = max(1, int(round(c.min_height_1080 * sy)))
        max_w = max(min_w + 1, int(round(c.max_width_1080 * sx)))
        max_h = max(min_h + 1, int(round(c.max_height_1080 * sy)))
        taskbar_y = int(round(h * c.taskbar_y_ratio))

        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            x, y, bw, bh = cv2.boundingRect(contour)
            if area < min_area or area > max_area:
                continue
            if bw < min_w or bh < min_h or bw > max_w or bh > max_h:
                continue
            ratio = bw / max(bh, 1)
            if ratio < c.min_aspect_ratio or ratio > c.max_aspect_ratio:
                continue
            if y >= taskbar_y:
                continue
            yellow_pixels = int(np.count_nonzero(yellow_mask[y:y + bh, x:x + bw]))
            yellow_pixel_ratio = yellow_pixels / max(bw * bh, 1)
            if yellow_pixel_ratio < c.min_yellow_pixel_ratio or yellow_pixel_ratio > c.max_yellow_pixel_ratio:
                continue

            local = np.zeros((bh, bw), dtype=np.uint8)
            shifted = contour - np.array([[[x, y]]], dtype=contour.dtype)
            cv2.drawContours(local, [shifted], -1, 255, cv2.FILLED)
            px, py, dist = self._positive_from_mask(local, x, y)
            score = area + dist * max(bw, bh)
            raw.append({
                "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
                "positive_point": [int(px), int(py)],
                "contour_area": area,
                "yellow_pixels": yellow_pixels,
                "yellow_pixel_ratio": float(yellow_pixel_ratio),
                "width": int(bw),
                "height": int(bh),
                "distance_to_boundary": float(dist),
                "proposal_score": float(score),
            })

        candidates = self._deduplicate_candidates(raw)
        max_candidates = int(c.max_candidates)
        if max_candidates > 0:
            candidates = candidates[:max_candidates]
        for idx, item in enumerate(candidates, 1):
            item["candidate_id"] = idx

        mask_path = out_dir / "yellow_mask.png"
        vis_path = out_dir / "candidates.png"
        json_path = out_dir / "candidates.json"
        cv2.imwrite(str(mask_path), yellow_mask)
        self._draw_candidates(image_bgr, candidates, vis_path)
        save_json(json_path, candidates)
        return {
            "yellow_mask": yellow_mask,
            "candidates": candidates,
            "yellow_mask_path": str(mask_path),
            "candidate_visualization_path": str(vis_path),
            "candidates_json_path": str(json_path),
        }

    def generate_prompts(
        self,
        candidates: list[dict],
        image_shape: tuple[int, int],
        edge_probability: np.ndarray,
        yellow_mask: np.ndarray,
        mode: str,
        out_dir: str | Path,
    ) -> list[dict]:
        out_dir = ensure_dir(out_dir)
        h, w = image_shape[:2]
        prompts = []
        for item in candidates:
            box = self._expand_box(item["bbox"], w, h)
            negatives = []
            negative_diagnostics = []
            if mode == "dexined":
                negatives, negative_diagnostics = self._negative_points(
                    item["bbox"],
                    item["positive_point"],
                    box,
                    edge_probability,
                    yellow_mask,
                )
            prompts.append({
                "prompt_id": len(prompts) + 1,
                "source": f"opencv_yellow_{mode}",
                "candidate_id": item["candidate_id"],
                "component_bbox": item["bbox"],
                "box": box,
                "positive_points": [item["positive_point"]],
                "negative_points": negatives,
                "negative_diagnostics": negative_diagnostics,
                "proposal_score": float(item["proposal_score"]),
            })

        max_prompts = int(self.cfg.prompts.max_prompts)
        if max_prompts > 0:
            prompts = prompts[:max_prompts]
            for idx, item in enumerate(prompts, 1):
                item["prompt_id"] = idx
        save_json(out_dir / "prompts.json", prompts)
        return prompts

    def _positive_from_mask(self, local_mask: np.ndarray, x: int, y: int) -> tuple[int, int, float]:
        mask = (local_mask > 0).astype(np.uint8)
        if not np.any(mask):
            return x + local_mask.shape[1] // 2, y + local_mask.shape[0] // 2, 0.0

        c = self.cfg.prompts
        bh, bw = mask.shape[:2]
        margin_x = int(round(bw * c.positive_search_margin_ratio))
        margin_y = int(round(bh * c.positive_search_margin_ratio))
        work = mask.copy()
        if margin_x > 0 and bw > margin_x * 2:
            work[:, :margin_x] = 0
            work[:, bw - margin_x:] = 0
        if margin_y > 0 and bh > margin_y * 2:
            work[:margin_y, :] = 0
            work[bh - margin_y:, :] = 0
        if not np.any(work):
            work = mask

        dist = cv2.distanceTransform(work, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist)
        if max_val <= 0:
            return x + bw // 2, y + bh // 2, 0.0
        return x + int(max_loc[0]), y + int(max_loc[1]), float(max_val)

    def _expand_box(self, box: list[int], w: int, h: int) -> list[int]:
        _, _, s = scale_from_1080(w, h)
        c = self.cfg.prompts
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        extra = max(int(round(c.min_box_expand_1080 * s)), int(round(max(bw, bh) * c.box_expand_ratio)))
        return [
            int(max(0, x1 - extra)),
            int(max(0, y1 - extra)),
            int(min(w - 1, x2 + extra)),
            int(min(h - 1, y2 + extra)),
        ]

    def _negative_points(
        self,
        candidate_box: list[int],
        positive: list[int],
        sam_box: list[int],
        edge_probability: np.ndarray,
        yellow_mask: np.ndarray,
    ) -> tuple[list[list[int]], list[dict]]:
        h, w = edge_probability.shape[:2]
        _, _, s = scale_from_1080(w, h)
        c = self.cfg.prompts
        x1, y1, x2, y2 = candidate_box
        bx1, by1, bx2, by2 = sam_box
        px, py = positive
        threshold = float(c.negative_edge_threshold)
        outside = max(1, int(round(c.negative_outside_offset_1080 * s)))
        safe_margin = max(1, int(round(c.negative_safe_margin_1080 * s)))
        search_distance = max(outside, int(round(c.negative_search_distance_1080 * s)))
        background_dist = cv2.distanceTransform((yellow_mask == 0).astype(np.uint8), cv2.DIST_L2, 5)

        negatives = []
        diagnostics = []
        directions = [
            ("left", -1.0, 0.0),
            ("right", 1.0, 0.0),
            ("top", 0.0, -1.0),
            ("bottom", 0.0, 1.0),
        ]
        if int(c.negative_directions) >= 8:
            inv = 1.0 / np.sqrt(2.0)
            directions.extend([
                ("top_left", -inv, -inv),
                ("top_right", inv, -inv),
                ("bottom_left", -inv, inv),
                ("bottom_right", inv, inv),
            ])

        for name, dx, dy in directions:
            edge_point = self._find_edge_along_vector(
                edge_probability,
                px,
                py,
                x1,
                y1,
                x2,
                y2,
                dx,
                dy,
                threshold,
            )
            if edge_point is None:
                diagnostics.append({
                    "direction": name,
                    "source": "none",
                    "edge_found": False,
                    "edge_point": None,
                    "negative_point": None,
                })
                continue
            point = self._background_after_edge(
                edge_point,
                dx,
                dy,
                sam_box,
                yellow_mask,
                background_dist,
                safe_margin,
                outside,
                search_distance,
            )
            if point is not None:
                negatives.append(point)
                diagnostics.append({
                    "direction": name,
                    "source": "dexined",
                    "edge_found": True,
                    "edge_point": [int(edge_point[0]), int(edge_point[1])],
                    "negative_point": point,
                })
            else:
                diagnostics.append({
                    "direction": name,
                    "source": "none",
                    "edge_found": True,
                    "edge_point": [int(edge_point[0]), int(edge_point[1])],
                    "negative_point": None,
                })
        return negatives, diagnostics

    def _background_after_edge(
        self,
        edge_point: tuple[int, int],
        dx: float,
        dy: float,
        sam_box: list[int],
        yellow_mask: np.ndarray,
        background_dist: np.ndarray,
        safe_margin: int,
        outside: int,
        search_distance: int,
    ) -> list[int] | None:
        bx1, by1, bx2, by2 = sam_box
        ex, ey = edge_point
        for delta in range(outside, search_distance + 1):
            x = int(round(ex + dx * delta))
            y = int(round(ey + dy * delta))
            if x < bx1 or x > bx2 or y < by1 or y > by2:
                break
            if self._is_safe_background(x, y, yellow_mask, background_dist, safe_margin):
                return [int(x), int(y)]
        return None

    @staticmethod
    def _find_edge_along_vector(
        edge_probability: np.ndarray,
        px: int,
        py: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        dx: float,
        dy: float,
        threshold: float,
    ) -> tuple[int, int] | None:
        h, w = edge_probability.shape[:2]
        px = int(np.clip(px, 0, w - 1))
        py = int(np.clip(py, 0, h - 1))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        max_steps = int(np.ceil(np.hypot(x2 - x1, y2 - y1))) + 1
        last = None
        for step in range(max_steps + 1):
            x = int(round(px + dx * step))
            y = int(round(py + dy * step))
            if x < x1 or x > x2 or y < y1 or y > y2:
                break
            if last == (x, y):
                continue
            last = (x, y)
            if edge_probability[y, x] >= threshold:
                return x, y
        return None

    @staticmethod
    def _is_safe_background(
        x: int,
        y: int,
        yellow_mask: np.ndarray,
        background_dist: np.ndarray,
        safe_margin: int,
    ) -> bool:
        h, w = yellow_mask.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return False
        return yellow_mask[y, x] == 0 and background_dist[y, x] >= safe_margin

    def _find_edge(
        self,
        edge_probability: np.ndarray,
        px: int,
        py: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        direction: str,
        threshold: float,
        band: int,
    ) -> int | None:
        h, w = edge_probability.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        px, py = int(np.clip(px, x1, x2)), int(np.clip(py, y1, y2))

        if direction in ("left", "right"):
            ys = slice(max(y1, py - band), min(y2 + 1, py + band + 1))
            xs = range(px, x1 - 1, -1) if direction == "left" else range(px, x2 + 1)
            for x in xs:
                if np.max(edge_probability[ys, x]) >= threshold:
                    return int(x)
        else:
            xs = slice(max(x1, px - band), min(x2 + 1, px + band + 1))
            ys = range(py, y1 - 1, -1) if direction == "top" else range(py, y2 + 1)
            for y in ys:
                if np.max(edge_probability[y, xs]) >= threshold:
                    return int(y)
        return None

    def _deduplicate_candidates(self, items: list[dict]) -> list[dict]:
        c = self.cfg.opencv_candidates
        ordered = sorted(items, key=lambda x: x["proposal_score"], reverse=True)
        kept = []
        for item in ordered:
            duplicate = False
            for old in kept:
                if bbox_iou(item["bbox"], old["bbox"]) >= c.nms_iou:
                    duplicate = True
                    break
                if containment(item["bbox"], old["bbox"]) >= c.containment_ratio:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(item)
        return kept

    def _draw_candidates(self, image_bgr: np.ndarray, candidates: list[dict], out_path: str | Path) -> None:
        h, w = image_bgr.shape[:2]
        _, _, s = scale_from_1080(w, h)
        thickness = max(1, int(round(self.cfg.opencv_candidates.draw_thickness_1080 * s)))
        vis = image_bgr.copy()
        for item in candidates:
            x1, y1, x2, y2 = item["bbox"]
            px, py = item["positive_point"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), thickness)
            cv2.circle(vis, (px, py), max(3, thickness + 2), (0, 255, 0), -1)
        ensure_dir(Path(out_path).parent)
        cv2.imwrite(str(out_path), vis)
