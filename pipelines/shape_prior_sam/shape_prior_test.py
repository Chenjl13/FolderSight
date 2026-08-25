from __future__ import annotations

import argparse
import logging
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

from src.config import AppConfig, RESOLUTION_MAP
from src.dexined_engine import DexiNedEngine
from src.image_processor import ImageProcessor
from src.utils import ensure_dir, save_json, scale_from_1080


def normalize_contour(contour: np.ndarray) -> np.ndarray | None:
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return None
    pts = contour.astype(np.float32).copy()
    pts[:, 0, 0] = (pts[:, 0, 0] - x) / float(w)
    pts[:, 0, 1] = (pts[:, 0, 1] - y) / float(h)
    return pts


def load_templates(folder_gt: Path) -> list[dict]:
    templates = []
    for idx, mask_path in enumerate(sorted(folder_gt.glob("*_mask.png")), 1):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        norm = normalize_contour(contour)
        if norm is None:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        extent = float(cv2.contourArea(contour)) / max(float(w * h), 1.0)
        templates.append({
            "template_id": idx,
            "mask_path": str(mask_path),
            "bbox": [int(x), int(y), int(x + w), int(y + h)],
            "width": int(w),
            "height": int(h),
            "aspect": float(w / max(h, 1)),
            "extent": extent,
            "contour": norm,
        })
    return templates


def template_stats(templates: list[dict], image_w: int, image_h: int) -> dict:
    aspects = np.array([t["aspect"] for t in templates], dtype=np.float32)
    extents = np.array([t["extent"] for t in templates], dtype=np.float32)
    widths = np.array([t["width"] for t in templates], dtype=np.float32)
    heights = np.array([t["height"] for t in templates], dtype=np.float32)
    return {
        "aspect_min": float(aspects.min() * 0.45),
        "aspect_max": float(aspects.max() * 2.25),
        "extent_min": float(max(0.02, extents.min() * 0.25)),
        "template_width_mean": float(widths.mean()),
        "template_height_mean": float(heights.mean()),
        "image_size": [int(image_w), int(image_h)],
    }


def draw_templates(templates: list[dict], out_path: Path) -> None:
    cell = 160
    canvas = np.full((cell, cell * max(len(templates), 1), 3), 255, dtype=np.uint8)
    for i, item in enumerate(templates):
        contour = item["contour"].copy()
        contour[:, 0, 0] = contour[:, 0, 0] * 110 + i * cell + 25
        contour[:, 0, 1] = contour[:, 0, 1] * 110 + 25
        contour_i = np.round(contour).astype(np.int32)
        cv2.drawContours(canvas, [contour_i], -1, (0, 0, 255), 2)
        cv2.putText(canvas, f"T{item['template_id']}", (i * cell + 12, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imwrite(str(out_path), canvas)


def connect_edges(edge_binary: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    edge_u8 = (edge_binary > 0).astype(np.uint8) * 255
    if iterations <= 0:
        return edge_u8
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(edge_u8, cv2.MORPH_CLOSE, kernel, iterations=int(iterations))


def count_external_contours(edge_u8: np.ndarray) -> int:
    contours, _ = cv2.findContours(edge_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)


# def candidate_contours(
#     edge_u8: np.ndarray,
#     stats: dict,
#     scale: float = 1.0,
# ) -> tuple[list[tuple[np.ndarray, list[int], float]], Counter]:
def candidate_contours(
    edge_u8: np.ndarray,
    stats: dict,
    scale: float = 1.0,
    resolution: str = "",
) -> tuple[list[tuple[np.ndarray, list[int], float]], Counter]:
    h, w = edge_u8.shape[:2]
    taskbar_y = int(round(h * 0.925))

    contours, _ = cv2.findContours(edge_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    prefiltered = []
    reasons = Counter()
    # for contour in contours:
    #     area = float(cv2.contourArea(contour))
    #     x, y, bw, bh = cv2.boundingRect(contour)
    #     if y >= taskbar_y:
    #         reasons["taskbar"] += 1
    #         continue
        
    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, bw, bh = cv2.boundingRect(contour)

        # 4K-only folder geometry prior.
        if resolution == "4k":
            template_w = float(stats["template_width_mean"])
            template_h = float(stats["template_height_mean"])

            min_folder_w = max(12, int(round(template_w * 0.55)))
            max_folder_w = int(round(template_w * 1.80))

            min_folder_h = max(12, int(round(template_h * 0.55)))
            max_folder_h = int(round(template_h * 1.80))

            if (
                bw < min_folder_w
                or bw > max_folder_w
                or bh < min_folder_h
                or bh > max_folder_h
            ):
                reasons["4k_bad_template_size"] += 1
                continue

        if y >= taskbar_y:
            reasons["taskbar"] += 1
            continue
        
        
        if min(bw, bh) < max(1, int(round(8 * scale))):
            reasons["too_thin"] += 1
            continue
        aspect = bw / max(bh, 1)
        if aspect < stats["aspect_min"] or aspect > stats["aspect_max"]:
            reasons["bad_aspect"] += 1
            continue
        extent = area / max(float(bw * bh), 1.0)
        if extent < stats["extent_min"]:
            reasons["low_extent"] += 1
            continue
        norm = normalize_contour(contour)
        if norm is None:
            reasons["invalid"] += 1
            continue
        prefiltered.append((norm, [int(x), int(y), int(x + bw), int(y + bh)], area, bw, bh))

    if not prefiltered:
        return [], reasons

    areas = np.array([x[2] for x in prefiltered], dtype=np.float32)
    widths = np.array([x[3] for x in prefiltered], dtype=np.float32)
    heights = np.array([x[4] for x in prefiltered], dtype=np.float32)
    area_lo, area_hi = np.percentile(areas, [10, 95])
    width_lo, width_hi = np.percentile(widths, [10, 95])
    height_lo, height_hi = np.percentile(heights, [10, 95])

    rows = []
    for norm, box, area, bw, bh in prefiltered:
        if area < area_lo * 0.65 or area > area_hi * 1.35:
            reasons["bad_size"] += 1
            continue
        if bw < width_lo * 0.65 or bw > width_hi * 1.35 or bh < height_lo * 0.65 or bh > height_hi * 1.35:
            reasons["bad_size"] += 1
            continue
        rows.append((norm, box, area))
    return rows, reasons


def draw_all_contours(image_bgr: np.ndarray, candidates: list[tuple[np.ndarray, list[int], float]], out_path: Path) -> None:
    vis = image_bgr.copy()
    _, _, scale = scale_from_1080(image_bgr.shape[1], image_bgr.shape[0])
    thickness = max(1, int(round(scale)))
    for _, box, _ in candidates:
        x1, y1, x2, y2 = box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (80, 200, 255), thickness)
    cv2.imwrite(str(out_path), vis)


def draw_top_matches(image_bgr: np.ndarray, matches: list[dict], out_path: Path) -> None:
    vis = image_bgr.copy()
    _, _, scale = scale_from_1080(image_bgr.shape[1], image_bgr.shape[0])
    thickness = max(1, int(round(2 * scale)))
    font_scale = 0.45 * scale
    for item in matches:
        x1, y1, x2, y2 = item["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        label = f"{item['candidate_id']} s={item['shape_score']:.3f} T{item['best_template_id']}"
        cv2.putText(
            vis,
            label,
            (x1, max(int(round(14 * scale)), y1 - int(round(4 * scale)))),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            max(1, thickness // 2),
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_path), vis)


def estimate_chamfer_size(matches: list[dict]) -> tuple[int, int]:
    pool = [m for m in matches[:30] if m["shape_score"] <= 0.10]
    if len(pool) < 5:
        pool = matches[:30]
    widths = [m["bbox"][2] - m["bbox"][0] for m in pool]
    heights = [m["bbox"][3] - m["bbox"][1] for m in pool]
    if not widths or not heights:
        return 46, 39
    return max(12, int(round(float(np.median(widths))))), max(12, int(round(float(np.median(heights)))))


def raster_template_edge(template: dict, width: int, height: int) -> np.ndarray:
    contour = template["contour"].copy()
    contour[:, 0, 0] = np.clip(contour[:, 0, 0] * max(width - 1, 1), 0, width - 1)
    contour[:, 0, 1] = np.clip(contour[:, 0, 1] * max(height - 1, 1), 0, height - 1)
    canvas = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(canvas, [np.round(contour).astype(np.int32)], -1, 255, 1)
    return (canvas > 0).astype(np.float32)


def box_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_low_score(candidates: list[dict], limit: int = 50) -> list[dict]:
    kept = []
    for item in sorted(candidates, key=lambda x: (x["chamfer_score"], -x["edge_coverage"])):
        box = item["bbox"]
        cx = (box[0] + box[2]) * 0.5
        cy = (box[1] + box[3]) * 0.5
        duplicate = False
        for old in kept:
            ob = old["bbox"]
            ocx = (ob[0] + ob[2]) * 0.5
            ocy = (ob[1] + ob[3]) * 0.5
            if box_iou(box, ob) >= 0.45 or np.hypot(cx - ocx, cy - ocy) < min(box[2] - box[0], box[3] - box[1]) * 0.55:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
            if len(kept) >= limit:
                break
    for idx, item in enumerate(kept, 1):
        item["candidate_id"] = idx
    return kept


def chamfer_match(
    edge_binary: np.ndarray,
    templates: list[dict],
    base_size: tuple[int, int],
    coverage_px: float = 3.0,
    scale: float = 1.0,
) -> list[dict]:
    edge_u8 = (edge_binary > 0).astype(np.uint8)
    dist = cv2.distanceTransform(1 - edge_u8, cv2.DIST_L2, 3).astype(np.float32)
    near = (dist <= float(coverage_px)).astype(np.float32)
    h, w = edge_binary.shape[:2]
    base_w, base_h = base_size
    coverage_px = float(coverage_px) * float(scale)
    scales = (0.80, 0.90, 1.00, 1.10, 1.25)
    rows = []

    for scale in scales:
        tw = max(8, int(round(base_w * scale)))
        th = max(8, int(round(base_h * scale)))
        if tw >= w or th >= h:
            continue
        for template in templates:
            kernel = raster_template_edge(template, tw, th)
            count = float(kernel.sum())
            if count <= 0:
                continue
            score_map = cv2.matchTemplate(dist, kernel, cv2.TM_CCORR) / count
            coverage_map = cv2.matchTemplate(near, kernel, cv2.TM_CCORR) / count

            local_kernel_size = max(1, int(round(9 * float(scale))))
            if local_kernel_size % 2 == 0:
                local_kernel_size += 1
            local_min = score_map == cv2.erode(score_map, np.ones((local_kernel_size, local_kernel_size), np.uint8))
            flat_scores = score_map.reshape(-1)
            take = min(600, flat_scores.size)
            idxs = np.argpartition(flat_scores, take - 1)[:take]
            for flat_idx in idxs:
                y, x = divmod(int(flat_idx), score_map.shape[1])
                if not local_min[y, x]:
                    continue
                rows.append({
                    "bbox": [int(x), int(y), int(x + tw), int(y + th)],
                    "chamfer_score": float(score_map[y, x]),
                    "mean_distance": float(score_map[y, x]),
                    "edge_coverage": float(coverage_map[y, x]),
                    "best_template_id": int(template["template_id"]),
                    "scale": float(scale),
                    "template_width": int(tw),
                    "template_height": int(th),
                })
    return nms_low_score(rows, 50)


def draw_chamfer_matches(image_bgr: np.ndarray, matches: list[dict], out_path: Path) -> None:
    vis = image_bgr.copy()
    _, _, scale = scale_from_1080(image_bgr.shape[1], image_bgr.shape[0])
    thickness = max(1, int(round(2 * scale)))
    font_scale = 0.42 * scale
    for item in matches:
        x1, y1, x2, y2 = item["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), thickness)
        label = f"{item['candidate_id']} c={item['chamfer_score']:.2f} cov={item['edge_coverage']:.2f} T{item['best_template_id']}"
        cv2.putText(
            vis,
            label,
            (x1, max(int(round(14 * scale)), y1 - int(round(4 * scale)))),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 255),
            max(1, thickness // 2),
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_path), vis)


def parse_args():
    p = argparse.ArgumentParser(description="DexiNed edge contour shape-prior smoke test.")
    p.add_argument("--image", default="examples/1080p/1.jpg")
    p.add_argument("--folder-gt", default="assets/folder_templates")
    p.add_argument("--output-dir", default="outputs/shape_prior_test")
    p.add_argument("--config", default="config.json")
    p.add_argument("--resolution", choices=["1080p", "2k", "4k"], default="1080p")
    p.add_argument("--sam-preprocess", action="store_true", default=True)
    p.add_argument("--no-sam-preprocess", dest="sam_preprocess", action="store_false")
    p.add_argument("--edge-close-kernel", type=int, default=3)
    p.add_argument("--edge-close-iterations", type=int, default=1)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    cfg = AppConfig.from_json(args.config)
    cfg.image.target_size = RESOLUTION_MAP[args.resolution]
    cfg.image.preprocess_enabled = bool(args.sam_preprocess)

    logger = logging.getLogger("shape_prior_test")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())

    templates = load_templates(Path(args.folder_gt))
    if len(templates) < 1:
        raise RuntimeError(f"No usable *_mask.png templates found under {args.folder_gt}")
    draw_templates(templates, out_dir / "templates.png")

    image_info = ImageProcessor(cfg).run(args.image, out_dir / "input")
    image_bgr = image_info["sam_input_bgr"]
    _, _, image_scale = scale_from_1080(image_bgr.shape[1], image_bgr.shape[0])
    dex = DexiNedEngine(cfg, logger).predict(image_bgr, out_dir / "dexined")
    stats = template_stats(templates, image_bgr.shape[1], image_bgr.shape[0])
    raw_edge = (dex["edge_binary"] > 0).astype(np.uint8) * 255
    raw_count = count_external_contours(raw_edge)
    # edge_close_kernel = max(1, int(round(args.edge_close_kernel * image_scale)))
    # if edge_close_kernel % 2 == 0:
    #     edge_close_kernel += 1
    if args.resolution == "4k":
        edge_close_kernel = max(1, int(args.edge_close_kernel))
    else:
        edge_close_kernel = max(1, int(round(args.edge_close_kernel * image_scale)))

    if edge_close_kernel % 2 == 0:
        edge_close_kernel += 1
    connected_edge = connect_edges(dex["edge_binary"], edge_close_kernel, args.edge_close_iterations)
    connected_count = count_external_contours(connected_edge)
    cv2.imwrite(str(out_dir / "edge_connected.png"), connected_edge)
    #candidates, filter_reasons = candidate_contours(connected_edge, stats, image_scale)
    candidates, filter_reasons = candidate_contours(
    connected_edge,
    stats,
    image_scale,
    resolution=args.resolution,
    )
    draw_all_contours(image_bgr, candidates, out_dir / "all_contours.png")

    matches = []
    for contour, box, area in candidates:
        scores = [float(cv2.matchShapes(contour, t["contour"], cv2.CONTOURS_MATCH_I1, 0.0)) for t in templates]
        best_idx = int(np.argmin(scores))
        matches.append({
            "bbox": box,
            "shape_score": float(scores[best_idx]),
            "template_scores": scores,
            "best_template_id": int(templates[best_idx]["template_id"]),
            "contour_area": float(area),
        })
    matches.sort(key=lambda x: x["shape_score"])
    for idx, item in enumerate(matches, 1):
        item["candidate_id"] = idx

    top50 = matches[:50]
    draw_top_matches(image_bgr, top50, out_dir / "top50_matches.png")
    chamfer_base_size = estimate_chamfer_size(matches)
    chamfer_matches = chamfer_match(dex["edge_binary"], templates, chamfer_base_size, scale=image_scale)
    draw_chamfer_matches(image_bgr, chamfer_matches, out_dir / "chamfer_top50.png")
    save_json(out_dir / "matches.json", {
        "image": args.image,
        "template_count": len(templates),
        "template_stats": stats,
        "raw_contour_count": raw_count,
        "connected_contour_count": connected_count,
        "filtered_contour_count": len(matches),
        "edge_connection": {
            "kernel": int(args.edge_close_kernel),
            "iterations": int(args.edge_close_iterations),
        },
        "filter_reasons": dict(filter_reasons),
        "top_count": len(top50),
        "matches": top50,
    })
    save_json(out_dir / "chamfer_matches.json", {
        "image": args.image,
        "template_count": len(templates),
        "base_size": [int(chamfer_base_size[0]), int(chamfer_base_size[1])],
        "top_count": len(chamfer_matches),
        "matches": chamfer_matches,
    })
    print(f"raw_contour_count={raw_count}")
    print(f"connected_contour_count={connected_count}")
    print(f"filtered_contour_count={len(matches)}")
    print(f"filter_reasons={dict(filter_reasons)}")
    print(out_dir / "top50_matches.png")
    print(out_dir / "chamfer_top50.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
