from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.utils import bbox_iou, center_distance, ensure_dir, save_json, scale_from_1080


def odd_size(value: float) -> int:
    size = max(1, int(round(value)))
    return size if size % 2 == 1 else size + 1


def scaled_range(values_1080: range, scale: float) -> list[int]:
    return sorted({max(1, int(round(value * scale))) for value in values_1080})


class WallpaperRefineStrategy:
    def __init__(self, image_shape: tuple[int, int], resolution: str):
        height, width = image_shape[:2]
        self.resolution = resolution
        self.sx, self.sy, _avg = scale_from_1080(width, height)
        self.s = min(self.sx, self.sy)

    def length(self, value_1080: float) -> int:
        return max(1, int(round(value_1080 * self.s)))

    def x(self, value_1080: float) -> int:
        return max(1, int(round(value_1080 * self.sx)))

    def y(self, value_1080: float) -> int:
        return max(1, int(round(value_1080 * self.sy)))

    def area(self, value_1080: float) -> int:
        return max(1, int(round(value_1080 * self.sx * self.sy)))

    def odd(self, value_1080: float) -> int:
        return odd_size(value_1080 * self.s)

    def odd_xy(self, x_1080: float, y_1080: float) -> tuple[int, int]:
        return odd_size(x_1080 * self.sx), odd_size(y_1080 * self.sy)

    def grid_step_candidates(self) -> tuple[list[int], list[int]]:
        return scaled_range(range(74, 79), self.sx), scaled_range(range(103, 114), self.sy)

    def target_folder_size(
        self,
        base_boxes: list[dict],
        image_shape: tuple[int, int],
    ) -> tuple[int, int]:
        if base_boxes:
            widths = np.array([int(row["bbox"][2]) - int(row["bbox"][0]) for row in base_boxes], dtype=np.float32)
            heights = np.array([int(row["bbox"][3]) - int(row["bbox"][1]) for row in base_boxes], dtype=np.float32)
            return (
                int(np.clip(np.median(widths), self.x(42), self.x(90))),
                int(np.clip(np.median(heights), self.y(44), self.y(110))),
            )
        h, w = image_shape[:2]
        return max(42, int(round(w / 26))), max(44, int(round(h / 12)))

    def template_crop_ratio(self) -> float:
        return 0.72

    def template_resize_size(self, target_w: int, target_h: int) -> tuple[int, int]:
        return int(target_w), max(self.y(18), int(target_h * self.template_crop_ratio()))

    def template_source_bbox(
        self,
        row: dict,
        target_w: int,
        target_h: int,
        image_shape: tuple[int, int],
    ) -> list[int]:
        _ = target_w, target_h, image_shape
        return [int(v) for v in row["bbox"]]

    def template_debug_enabled(self) -> bool:
        return False

    def raw_template_signal_threshold(self) -> float:
        return 0.34

    def template_match_offsets(self) -> list[tuple[int, int]]:
        return [(0, 0)]

    def template_verified(self, row: dict, template_keep_threshold: float) -> bool:
        return float(row.get("template_similarity", 0.0)) >= float(template_keep_threshold)

    def max_recovery_limit(self, requested: int) -> int:
        return int(requested)


class WallpaperRefine2KStrategy(WallpaperRefineStrategy):
    def grid_step_candidates(self) -> tuple[list[int], list[int]]:
        return list(range(74, 77)), list(range(101, 106))

    def target_folder_size(
        self,
        base_boxes: list[dict],
        image_shape: tuple[int, int],
    ) -> tuple[int, int]:
        if not base_boxes:
            _h, w = image_shape[:2]
            return max(58, int(round(w / 41))), 94

        widths = np.array([int(row["bbox"][2]) - int(row["bbox"][0]) for row in base_boxes], dtype=np.float32)
        heights = np.array([int(row["bbox"][3]) - int(row["bbox"][1]) for row in base_boxes], dtype=np.float32)
        contour_widths = []
        contour_heights = []
        for row in base_boxes:
            if "contour_width" in row and "contour_height" in row:
                contour_widths.append(float(row["contour_width"]))
                contour_heights.append(float(row["contour_height"]))
            elif row.get("contour_bbox"):
                x1, y1, x2, y2 = [int(v) for v in row["contour_bbox"]]
                contour_widths.append(float(x2 - x1))
                contour_heights.append(float(y2 - y1))

        expanded_w = float(np.median(widths))
        expanded_h = float(np.median(heights))
        if contour_widths and contour_heights:
            icon_w = float(np.median(np.array(contour_widths, dtype=np.float32)))
            icon_h = float(np.median(np.array(contour_heights, dtype=np.float32)))
            target_w = max(icon_w + 12.0, expanded_w * 0.63)
            target_h = max(icon_h + 42.0, expanded_h * 0.76)
        else:
            target_w = expanded_w * 0.63
            target_h = expanded_h * 0.76
        return int(np.clip(round(target_w), 58, 68)), int(np.clip(round(target_h), 88, 100))

    def template_crop_ratio(self) -> float:
        return 0.70

    def template_source_bbox(
        self,
        row: dict,
        target_w: int,
        target_h: int,
        image_shape: tuple[int, int],
    ) -> list[int]:
        height, width = image_shape[:2]
        x1, y1, _x2, _y2 = [int(v) for v in row["bbox"]]
        left = int(x1)
        top = int(y1)
        left = min(max(0, left), max(0, width - target_w))
        top = min(max(0, top), max(0, height - target_h))
        return [left, top, int(left + target_w), int(top + target_h)]

    def template_debug_enabled(self) -> bool:
        return True


class WallpaperRefine4KStrategy(WallpaperRefineStrategy):
    def grid_step_candidates(self) -> tuple[list[int], list[int]]:
        return list(range(74, 77)), list(range(88, 101))

    def target_folder_size(
        self,
        base_boxes: list[dict],
        image_shape: tuple[int, int],
    ) -> tuple[int, int]:
        if base_boxes:
            widths = np.array([int(row["bbox"][2]) - int(row["bbox"][0]) for row in base_boxes], dtype=np.float32)
            heights = np.array([int(row["bbox"][3]) - int(row["bbox"][1]) for row in base_boxes], dtype=np.float32)
            return int(round(float(np.median(widths)))), int(round(float(np.median(heights))))
        return super().target_folder_size(base_boxes, image_shape)

    def raw_template_signal_threshold(self) -> float:
        return 0.13

    def template_debug_enabled(self) -> bool:
        return True

    def template_match_offsets(self) -> list[tuple[int, int]]:
        values = [-12, 0, 12]
        return [(dx, dy) for dy in values for dx in values]

    def template_verified(self, row: dict, template_keep_threshold: float) -> bool:
        if not super().template_verified(row, template_keep_threshold):
            return False
        features = row.get("quality_features", {})
        white = row.get("white_component", {})
        has_white_anchor = int(white.get("count", 0)) > 0
        has_visual_structure = (
            float(features.get("edge_ratio", 0.0)) >= 0.04
            and float(features.get("val_std", 0.0)) >= 0.04
        )
        return has_white_anchor or has_visual_structure

    def max_recovery_limit(self, requested: int) -> int:
        return min(int(requested), 6)


def wallpaper_refine_strategy(image_shape: tuple[int, int], resolution: str) -> WallpaperRefineStrategy:
    if resolution == "1080p":
        return wallpaper_refine_1080p_strategy(image_shape)
    if resolution == "2k":
        return wallpaper_refine_2k_strategy(image_shape)
    if resolution == "4k":
        return wallpaper_refine_4k_strategy(image_shape)
    return WallpaperRefineStrategy(image_shape, resolution)


def wallpaper_refine_1080p_strategy(image_shape: tuple[int, int]) -> WallpaperRefineStrategy:
    return WallpaperRefineStrategy(image_shape, "1080p")


def wallpaper_refine_2k_strategy(image_shape: tuple[int, int]) -> WallpaperRefineStrategy:
    return WallpaperRefine2KStrategy(image_shape, "2k")


def wallpaper_refine_4k_strategy(image_shape: tuple[int, int]) -> WallpaperRefineStrategy:
    return WallpaperRefine4KStrategy(image_shape, "4k")


def large_warm_wallpaper_mask(image_bgr: np.ndarray, strategy: WallpaperRefineStrategy) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    warm = ((h >= 5) & (h <= 45) & (s >= 35) & (v >= 90)).astype(np.uint8) * 255
    k = strategy.odd(9)
    warm = cv2.morphologyEx(warm, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))

    num, labels, stats, _centers = cv2.connectedComponentsWithStats(warm, 8)
    height, width = warm.shape[:2]
    min_area = int(height * width * 0.004)
    large = np.zeros_like(warm)
    for idx in range(1, num):
        _x, _y, w, hgt, area = [int(v) for v in stats[idx]]
        if area >= min_area and (w >= width * 0.055 or hgt >= height * 0.055):
            large[labels == idx] = 255
    return large


def warm_search_mask_from_warm(warm_mask: np.ndarray, strategy: WallpaperRefineStrategy) -> np.ndarray:
    k = strategy.odd(21)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(warm_mask, kernel, iterations=1)


def resolve_background_path(bg_path: str | Path) -> Path:
    path = Path(bg_path)
    if path.exists():
        return path
    if path.name.lower() == "bg_1080p.png":
        fallback = path.with_name("1080P.png")
        if fallback.exists():
            return fallback
    return path


def crop_features(image_bgr: np.ndarray, box: list[int]) -> dict:
    x1, y1, x2, y2 = [int(v) for v in box]
    crop = image_bgr[y1:y2, x1:x2]
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    if crop.size == 0 or width == 0 or height == 0:
        return {
            "width": width,
            "height": height,
            "aspect": 0.0,
            "yellow_ratio": 0.0,
            "edge_ratio": 1.0,
            "sat_mean": 0.0,
            "val_std": 0.0,
        }

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    yellow = (h >= 5) & (h <= 45) & (s >= 35) & (v >= 90)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 140) > 0
    return {
        "width": width,
        "height": height,
        "aspect": float(width / max(1, height)),
        "yellow_ratio": float(yellow.mean()),
        "edge_ratio": float(edges.mean()),
        "sat_mean": float(s.mean() / 255.0),
        "val_std": float(v.std() / 255.0),
    }


def infer_typical_size(
    base_boxes: list[dict],
    image_shape: tuple[int, int],
    strategy: WallpaperRefineStrategy,
) -> tuple[int, int]:
    return strategy.target_folder_size(base_boxes, image_shape)


def build_folder_edge_templates(
    image_bgr: np.ndarray,
    base_boxes: list[dict],
    target_w: int,
    target_h: int,
    strategy: WallpaperRefineStrategy,
    debug_templates: list[dict] | None = None,
) -> list[np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 40, 140)
    resize_w, template_h = strategy.template_resize_size(target_w, target_h)
    templates: list[np.ndarray] = []
    for row in base_boxes:
        x1, y1, x2, y2 = strategy.template_source_bbox(row, target_w, target_h, image_bgr.shape)
        w, h = x2 - x1, y2 - y1
        if not (target_w * 0.65 <= w <= target_w * 1.45 and target_h * 0.55 <= h <= target_h * 1.40):
            continue
        crop = edge[y1 : min(y2, y1 + int(h * strategy.template_crop_ratio())), x1:x2]
        if crop.size == 0 or float((crop > 0).mean()) < 0.015:
            continue
        templates.append(cv2.resize(crop, (resize_w, template_h), interpolation=cv2.INTER_AREA))
        if debug_templates is not None:
            debug_templates.append({
                "template_index": len(templates),
                "source_bbox": [int(x1), int(y1), int(x2), int(y2)],
                "source_width": int(w),
                "source_height": int(h),
                "resize_width": int(resize_w),
                "resize_height": int(template_h),
                "base_bbox": [int(v) for v in row.get("bbox", [])],
                "contour_bbox": [int(v) for v in row.get("contour_bbox", [])],
            })
    return templates[:32]


def box_inside_region(box: list[int], region: np.ndarray, min_ratio: float) -> bool:
    x1, y1, x2, y2 = [int(v) for v in box]
    crop = region[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    cx, cy = int((x1 + x2) * 0.5), int((y1 + y2) * 0.5)
    center_ok = region[min(region.shape[0] - 1, cy), min(region.shape[1] - 1, cx)] > 0
    return center_ok or float((crop > 0).mean()) >= float(min_ratio)


def grid_cell_warm_status(box: list[int], warm_search_mask: np.ndarray) -> tuple[bool, float]:
    x1, y1, x2, y2 = [int(v) for v in box]
    cx, cy = int((x1 + x2) * 0.5), int((y1 + y2) * 0.5)
    center_inside = warm_search_mask[min(warm_search_mask.shape[0] - 1, cy), min(warm_search_mask.shape[1] - 1, cx)] > 0
    icon_y2 = y1 + max(1, int(round((y2 - y1) * 0.65)))
    icon_region = warm_search_mask[y1:icon_y2, x1:x2]
    overlap = 0.0 if icon_region.size == 0 else float((icon_region > 0).mean())
    return bool(center_inside), overlap


def grid_cell_should_be_checked(box: list[int], warm_search_mask: np.ndarray) -> bool:
    center_inside, overlap = grid_cell_warm_status(box, warm_search_mask)
    return center_inside or overlap >= 0.15


def white_component_features(
    image_bgr: np.ndarray,
    box: list[int],
    strategy: WallpaperRefineStrategy,
) -> dict:
    x1, y1, x2, y2 = [int(v) for v in box]
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return {"count": 0, "max_area": 0, "max_width": 0, "max_height": 0}
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    white = ((s < 82) & (v > 150)).astype(np.uint8) * 255
    kx, ky = strategy.odd_xy(3, 5)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((ky, kx), np.uint8), iterations=1)
    num, _labels, stats, _centers = cv2.connectedComponentsWithStats(white, 8)
    valid = []
    for idx in range(1, num):
        _x, _y, w, h, area = [int(v) for v in stats[idx]]
        if strategy.area(12) <= area <= strategy.area(900) and 1 <= w <= strategy.x(90) and 1 <= h <= strategy.y(55):
            valid.append((area, w, h))
    if not valid:
        return {"count": 0, "max_area": 0, "max_width": 0, "max_height": 0}
    area, w, h = max(valid, key=lambda item: item[0])
    return {"count": len(valid), "max_area": int(area), "max_width": int(w), "max_height": int(h)}


def template_similarity_for_box(
    edge: np.ndarray,
    templates: list[np.ndarray],
    box: list[int],
    target_w: int,
    target_h: int,
    strategy: WallpaperRefineStrategy,
) -> float:
    if not templates:
        return 0.0
    x1, y1, x2, y2 = [int(v) for v in box]
    resize_w, template_h = strategy.template_resize_size(target_w, target_h)
    height, width = edge.shape[:2]
    best = 0.0
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    for dx, dy in strategy.template_match_offsets():
        sx1 = x1 + int(dx)
        sy1 = y1 + int(dy)
        sx2 = sx1 + box_w
        sy2 = sy1 + box_h
        if sx1 < 0 or sy1 < 0 or sx2 > width or sy1 + template_h > height:
            continue
        crop = edge[sy1 : min(sy2, sy1 + template_h), sx1:sx2]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (resize_w, template_h), interpolation=cv2.INTER_AREA)
        for template in templates:
            if template.shape != crop.shape:
                template = cv2.resize(template, (resize_w, template_h), interpolation=cv2.INTER_AREA)
            score = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
            best = max(best, float(score[0, 0]))
    return float(best)


def desktop_grid_step_candidates(strategy: WallpaperRefineStrategy) -> tuple[list[int], list[int]]:
    return strategy.grid_step_candidates()


def axis_distance(value: int, anchor: int, step: int) -> int:
    raw = (value - anchor) % step
    if raw > step / 2:
        raw -= step
    return int(round(raw))


def estimate_axis_anchor(values: list[int], step: int) -> int:
    best_anchor = 0
    best_score: tuple[float, int] | None = None
    for anchor in range(step):
        distances = sorted(abs(axis_distance(value, anchor, step)) for value in values)
        core = distances[: max(1, int(len(distances) * 0.82))]
        score = (float(np.median(core)), sum(core))
        if best_score is None or score < best_score:
            best_score = score
            best_anchor = anchor
    return best_anchor


def estimate_axis_grid(values: list[int], steps: list[int]) -> tuple[int, int]:
    best: tuple[float, int, int, int] | None = None
    for step in steps:
        anchor = estimate_axis_anchor(values, step)
        distances = sorted(abs(axis_distance(value, anchor, step)) for value in values)
        core = distances[: max(1, int(len(distances) * 0.82))]
        score = (float(np.median(core)), sum(core), step, anchor)
        if best is None or score < best:
            best = score
    assert best is not None
    return best[2], best[3]


def infer_desktop_grid(base_boxes: list[dict], strategy: WallpaperRefineStrategy) -> dict | None:
    steps = desktop_grid_step_candidates(strategy)
    if not steps or len(base_boxes) < 6:
        return None
    x_values = [int(row["bbox"][0]) for row in base_boxes]
    y_values = [int(row["bbox"][1]) for row in base_boxes]
    x_step, x_anchor = estimate_axis_grid(x_values, steps[0])
    y_step, y_anchor = estimate_axis_grid(y_values, steps[1])
    return {"step": [int(x_step), int(y_step)], "anchor": [int(x_anchor), int(y_anchor)]}


def axis_positions(anchor: int, step: int, limit: int, size: int) -> list[int]:
    start = int(anchor)
    while start - step >= 0:
        start -= step
    values = []
    value = start
    while value + size <= limit:
        if value >= 0:
            values.append(int(value))
        value += step
    return values


def enumerate_grid_cells(
    image_shape: tuple[int, int],
    target_w: int,
    target_h: int,
    grid: dict,
) -> list[dict]:
    height, width = image_shape[:2]
    taskbar_y = int(round(height * 0.925))
    x_step, y_step = [int(v) for v in grid["step"]]
    x_anchor, y_anchor = [int(v) for v in grid["anchor"]]
    cells = []
    for y in axis_positions(y_anchor, y_step, taskbar_y, target_h):
        for x in axis_positions(x_anchor, x_step, width, target_w):
            cells.append({
                "cell_id": len(cells) + 1,
                "bbox": [int(x), int(y), int(x + target_w), int(y + target_h)],
                "center": [int(x + target_w * 0.5), int(y + target_h * 0.5)],
            })
    return cells


def assign_base_boxes_to_grid_cells(base_boxes: list[dict], all_grid_cells: list[dict]) -> set[int]:
    occupied: set[int] = set()
    if not all_grid_cells:
        return occupied

    for row in base_boxes:
        base_box = [int(v) for v in row["bbox"]]
        best_cell_id = None
        best_score: tuple[float, float] | None = None
        best_center_dist = 0.0
        best_top_left_dist = 0.0
        best_cell_box = None
        for cell in all_grid_cells:
            cell_box = [int(v) for v in cell["bbox"]]
            center_dist = center_distance(base_box, cell_box)
            top_left_dist = float(np.hypot(base_box[0] - cell_box[0], base_box[1] - cell_box[1]))
            score = (center_dist, top_left_dist)
            if best_score is None or score < best_score:
                best_score = score
                best_cell_id = int(cell["cell_id"])
                best_center_dist = center_dist
                best_top_left_dist = top_left_dist
                best_cell_box = cell_box
        if best_cell_id is None or best_cell_box is None:
            continue
        cell_w = max(1, best_cell_box[2] - best_cell_box[0])
        cell_h = max(1, best_cell_box[3] - best_cell_box[1])
        near_center = best_center_dist <= min(cell_w, cell_h) * 0.45
        near_top_left = best_top_left_dist <= min(cell_w, cell_h) * 0.35
        if near_center or near_top_left:
            occupied.add(best_cell_id)

    for cell in all_grid_cells:
        cell_box = [int(v) for v in cell["bbox"]]
        for row in base_boxes:
            if bbox_iou(cell_box, [int(v) for v in row["bbox"]]) >= 0.45:
                occupied.add(int(cell["cell_id"]))
                break
    return occupied


def recovery_boxes_overlap(box: list[int], rows: list[dict], distance_scale: float = 0.72) -> bool:
    for row in rows:
        old = [int(v) for v in row["bbox"]]
        max_dim = max(box[2] - box[0], box[3] - box[1], old[2] - old[0], old[3] - old[1])
        if bbox_iou(box, old) > 0.12 or center_distance(box, old) < max_dim * distance_scale:
            return True
    return False


def generate_raw_recoveries(
    image_bgr: np.ndarray,
    warm_mask: np.ndarray,
    base_boxes: list[dict],
    grid_cells: list[dict],
    target_w: int,
    target_h: int,
    occupied_cell_ids: set[int],
    strategy: WallpaperRefineStrategy,
    template_debug: dict | None = None,
) -> list[dict]:
    debug_templates = template_debug["templates"] if template_debug is not None else None
    debug_cells = template_debug["cells"] if template_debug is not None else None
    templates = build_folder_edge_templates(image_bgr, base_boxes, target_w, target_h, strategy, debug_templates)
    edge = cv2.Canny(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), 40, 140)
    raw = []
    for cell in grid_cells:
        box = [int(v) for v in cell["bbox"]]
        if int(cell["cell_id"]) in occupied_cell_ids:
            continue
        template_similarity = template_similarity_for_box(edge, templates, box, target_w, target_h, strategy)
        if debug_cells is not None:
            debug_cells.append({
                "cell_id": int(cell["cell_id"]),
                "bbox": box,
                "target_w": int(target_w),
                "target_h": int(target_h),
                "best_template_score": round(float(template_similarity), 4),
            })
        if not box_inside_region(box, warm_mask, 0.34):
            continue
        features = crop_features(image_bgr, box)
        white = white_component_features(image_bgr, box, strategy)
        has_white_anchor = white["count"] > 0 and white["max_area"] >= strategy.area(18)
        has_template_signal = template_similarity >= strategy.raw_template_signal_threshold()
        if features["yellow_ratio"] < 0.24:
            continue
        if features["edge_ratio"] > 0.31:
            continue
        if not (has_white_anchor or has_template_signal):
            continue
        score = (
            min(features["yellow_ratio"], 0.85) * 0.42
            + min(features["edge_ratio"], 0.22) * 0.75
            + min(features["val_std"], 0.30) * 0.45
            + min(max(template_similarity, 0.0), 0.70) * 0.55
            + min(white["max_area"] / float(strategy.area(500)), 0.35)
        )
        raw.append({
            "bbox": box,
            "center": cell["center"],
            "source": "wallpaper_grid_cell_recovery",
            "candidate_id": f"wallpaper_grid_{cell['cell_id']}",
            "final_score": round(float(score), 4),
            "quality_features": features,
            "white_component": white,
            "template_similarity": round(float(template_similarity), 4),
            "grid_cell_id": int(cell["cell_id"]),
        })
    return raw


def verify_template_recoveries(
    raw: list[dict],
    warm_mask: np.ndarray,
    occupied_cell_ids: set[int],
    template_keep_threshold: float,
    max_recovery: int,
    strategy: WallpaperRefineStrategy,
) -> list[dict]:
    verified = [row for row in raw if strategy.template_verified(row, template_keep_threshold)]
    kept: list[dict] = []
    for row in sorted(verified, key=lambda item: float(item.get("template_similarity", 0.0)), reverse=True):
        cell_id = int(row.get("grid_cell_id", -1))
        if not box_inside_region(row["bbox"], warm_mask, 0.34):
            continue
        if cell_id in occupied_cell_ids:
            continue
        if recovery_boxes_overlap(row["bbox"], kept, distance_scale=0.72):
            continue
        item = dict(row)
        item["accept_reason"] = "template_verified"
        kept.append(item)
        if len(kept) >= int(max_recovery):
            break
    return kept


def merge_final_boxes(base_boxes: list[dict], recovery: list[dict]) -> list[dict]:
    final = [dict(row) for row in base_boxes]
    for row in recovery:
        duplicate = False
        for old in base_boxes:
            if bbox_iou(row["bbox"], old["bbox"]) >= 0.45:
                duplicate = True
                break
        if duplicate or recovery_boxes_overlap(row["bbox"], final[len(base_boxes):], distance_scale=0.72):
            continue
        final.append(dict(row))
    for idx, row in enumerate(final, 1):
        row["box_id"] = idx
    return final


def draw_grid_cells(
    image_bgr: np.ndarray,
    cells: list[dict],
    out_path: Path,
    color: tuple[int, int, int],
    strategy: WallpaperRefineStrategy,
) -> None:
    vis = image_bgr.copy()
    thickness = strategy.length(1)
    for cell in cells:
        x1, y1, x2, y2 = [int(v) for v in cell["bbox"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
    cv2.imwrite(str(out_path), vis)


def draw_recovery_candidates(
    image_bgr: np.ndarray,
    candidates: list[dict],
    out_path: Path,
    strategy: WallpaperRefineStrategy,
) -> None:
    vis = image_bgr.copy()
    thickness = strategy.length(2)
    font_scale = 0.45 * strategy.s
    for idx, row in enumerate(candidates, 1):
        x1, y1, x2, y2 = [int(v) for v in row["bbox"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), thickness)
        cv2.putText(
            vis,
            str(idx),
            (x1, max(strategy.y(14), y1 - strategy.y(4))),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 165, 255),
            max(1, thickness // 2),
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_path), vis)


def draw_template_verified(
    image_bgr: np.ndarray,
    raw: list[dict],
    kept: list[dict],
    out_path: Path,
    strategy: WallpaperRefineStrategy,
) -> None:
    vis = image_bgr.copy()
    kept_ids = {int(row.get("grid_cell_id", -1)) for row in kept}
    thickness = strategy.length(2)
    font_scale = 0.38 * strategy.s
    for row in raw:
        cell_id = int(row.get("grid_cell_id", -1))
        x1, y1, x2, y2 = [int(v) for v in row["bbox"]]
        verified = cell_id in kept_ids
        color = (0, 255, 0) if verified else (0, 0, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        label = f"C{cell_id} t={float(row.get('template_similarity', 0.0)):.3f}"
        cv2.putText(
            vis,
            label,
            (x1, max(strategy.y(14), y1 - strategy.y(4))),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            max(1, thickness // 2),
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_path), vis)


def draw_templates_debug(
    image_bgr: np.ndarray,
    templates: list[dict],
    out_path: Path,
    strategy: WallpaperRefineStrategy,
) -> None:
    vis = image_bgr.copy()
    thickness = strategy.length(2)
    font_scale = 0.42 * strategy.s
    for row in templates:
        x1, y1, x2, y2 = [int(v) for v in row["source_bbox"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), thickness)
        cv2.putText(
            vis,
            str(row["template_index"]),
            (x1, max(strategy.y(14), y1 - strategy.y(4))),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 0, 255),
            max(1, thickness // 2),
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_path), vis)


def draw_final_boxes(
    image_bgr: np.ndarray,
    recovery: list[dict],
    final_boxes: list[dict],
    out_path: Path,
    strategy: WallpaperRefineStrategy,
) -> None:
    vis = image_bgr.copy()
    recovery_ids = {tuple(int(v) for v in row["bbox"]) for row in recovery}
    thickness = strategy.length(2)
    font_scale = 0.45 * strategy.s
    for row in final_boxes:
        box = [int(v) for v in row["bbox"]]
        x1, y1, x2, y2 = box
        is_recovery = tuple(box) in recovery_ids
        color = (0, 165, 255) if is_recovery else (0, 255, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            vis,
            str(row.get("box_id", "")),
            (x1, max(strategy.y(14), y1 - strategy.y(4))),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            max(1, thickness // 2),
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_path), vis)


def run_wallpaper_refine(
    image_bgr: np.ndarray,
    base_boxes: list[dict],
    bg_path: str | Path,
    resolution: str,
    out_dir: str | Path,
    gt_path: str | Path | None = None,
    image_path: str | Path | None = None,
    wallpaper_template_keep_threshold: float = 0.13,
    max_wallpaper_recovery: int = 10,
) -> dict:
    _ = gt_path, image_path
    out_dir = ensure_dir(out_dir)
    bg_path = resolve_background_path(bg_path)
    image_size = [int(image_bgr.shape[1]), int(image_bgr.shape[0])]
    strategy = wallpaper_refine_strategy(image_bgr.shape, resolution)
    warm_mask = large_warm_wallpaper_mask(image_bgr, strategy)
    warm_search_mask = warm_search_mask_from_warm(warm_mask, strategy)

    target_w, target_h = infer_typical_size(base_boxes, image_bgr.shape, strategy)
    grid = infer_desktop_grid(base_boxes, strategy)
    all_grid_cells = enumerate_grid_cells(image_bgr.shape, target_w, target_h, grid) if grid else []
    occupied_cell_ids = assign_base_boxes_to_grid_cells(base_boxes, all_grid_cells)
    empty_grid_cells = [
        cell
        for cell in all_grid_cells
        if int(cell["cell_id"]) not in occupied_cell_ids
        and grid_cell_should_be_checked([int(v) for v in cell["bbox"]], warm_search_mask)
    ]

    template_debug = {"templates": [], "cells": []} if strategy.template_debug_enabled() else None
    raw = generate_raw_recoveries(
        image_bgr=image_bgr,
        warm_mask=warm_mask,
        base_boxes=base_boxes,
        grid_cells=empty_grid_cells,
        target_w=target_w,
        target_h=target_h,
        occupied_cell_ids=occupied_cell_ids,
        strategy=strategy,
        template_debug=template_debug,
    )
    kept = verify_template_recoveries(
        raw=raw,
        warm_mask=warm_mask,
        occupied_cell_ids=occupied_cell_ids,
        template_keep_threshold=wallpaper_template_keep_threshold,
        max_recovery=strategy.max_recovery_limit(max_wallpaper_recovery),
        strategy=strategy,
    )
    final_boxes = merge_final_boxes(base_boxes, kept)

    cv2.imwrite(str(out_dir / "wallpaper_warm_mask.png"), warm_mask)
    draw_grid_cells(image_bgr, empty_grid_cells, out_dir / "wallpaper_empty_grid_cells.png", (0, 165, 255), strategy)
    draw_recovery_candidates(image_bgr, kept, out_dir / "wallpaper_recovery_candidates.png", strategy)
    draw_template_verified(image_bgr, raw, kept, out_dir / "wallpaper_template_verified.png", strategy)
    if template_debug is not None:
        template_debug["target_w"] = int(target_w)
        template_debug["target_h"] = int(target_h)
        template_debug["template_count"] = len(template_debug["templates"])
        template_debug["raw_recovery_template_score_range"] = [
            float(min([row["template_similarity"] for row in raw], default=0.0)),
            float(max([row["template_similarity"] for row in raw], default=0.0)),
        ]
        debug_prefix = f"wallpaper_{strategy.resolution}_template"
        save_json(out_dir / f"{debug_prefix}_debug.json", template_debug)
        draw_templates_debug(image_bgr, template_debug["templates"], out_dir / f"{debug_prefix}s_debug.png", strategy)
    draw_final_boxes(image_bgr, kept, final_boxes, out_dir / "final_refined_boxes.png", strategy)
    save_json(out_dir / "final_refined_boxes.json", final_boxes)

    report = {
        "resolution": resolution,
        "image_size": image_size,
        "background_path": str(bg_path),
        "base_final_box_count": len(base_boxes),
        "empty_warm_grid_cell_count": len(empty_grid_cells),
        "raw_recovery_count": len(raw),
        "template_verified_recovery_count": len(kept),
        "final_refined_box_count": len(final_boxes),
        "wallpaper_template_keep_threshold": float(wallpaper_template_keep_threshold),
        "max_wallpaper_recovery": int(max_wallpaper_recovery),
        "effective_max_wallpaper_recovery": int(strategy.max_recovery_limit(max_wallpaper_recovery)),
        "target_folder_size": [int(target_w), int(target_h)],
        "grid": grid,
        "scale": {"sx": strategy.sx, "sy": strategy.sy, "s": strategy.s},
        "artifacts": {
            "wallpaper_warm_mask": str(out_dir / "wallpaper_warm_mask.png"),
            "wallpaper_empty_grid_cells": str(out_dir / "wallpaper_empty_grid_cells.png"),
            "wallpaper_recovery_candidates": str(out_dir / "wallpaper_recovery_candidates.png"),
            "wallpaper_template_verified": str(out_dir / "wallpaper_template_verified.png"),
            "final_refined_boxes": str(out_dir / "final_refined_boxes.png"),
            "final_refined_boxes_json": str(out_dir / "final_refined_boxes.json"),
        },
        "recovery_boxes": kept,
    }
    save_json(out_dir / "wallpaper_refine_report.json", report)
    return report
