from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from candidate_fusion import fuse_candidate_sets, save_fusion
from shape_prior_test import (
    candidate_contours,
    chamfer_match,
    connect_edges,
    count_external_contours,
    draw_chamfer_matches,
    draw_templates,
    draw_top_matches,
    estimate_chamfer_size,
    load_templates,
    template_stats,
)
from wallpaper_refine import run_wallpaper_refine
from src.config import AppConfig, RESOLUTION_MAP
from src.cv_postprocess import CVPostProcessor
from src.dexined_engine import DexiNedEngine
from src.image_processor import ImageProcessor
from src.sam_engine import SamEngine
from src.utils import bbox_from_mask, ensure_dir, save_json, scale_from_1080
from src.visualizer import Visualizer


def expand_box(box: list[int], image_w: int, image_h: int, scale: float) -> list[int]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pad = max(int(round(8 * scale)), int(round(max(bw, bh) * 0.18)))
    return [
        int(max(0, x1 - pad)),
        int(max(0, y1 - pad)),
        int(min(image_w - 1, x2 + pad)),
        int(min(image_h - 1, y2 + pad)),
    ]


def expand_candidate_bbox(box: list[int], image_shape: tuple[int, int], ratio: float) -> list[int]:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    px = int(round(bw * float(ratio)))
    py = int(round(bh * float(ratio)))
    return [
        int(max(0, x1 - px)),
        int(max(0, y1 - py)),
        int(min(w - 1, x2 + px)),
        int(min(h - 1, y2 + py)),
    ]


def expand_candidate_list(items: list[dict], image_shape: tuple[int, int], ratio: float) -> list[dict]:
    rows = []
    for item in items:
        row = dict(item)
        row["bbox_original"] = [int(v) for v in item["bbox"]]
        row["bbox"] = expand_candidate_bbox(row["bbox_original"], image_shape, ratio)
        rows.append(row)
    return rows


def positive_point(edge_binary: np.ndarray, box: list[int]) -> list[int]:
    h, w = edge_binary.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return [int((box[0] + box[2]) * 0.5), int((box[1] + box[3]) * 0.5)]
    roi_edge = (edge_binary[y1:y2, x1:x2] > 0).astype(np.uint8)
    free = 1 - roi_edge
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist)
    if max_val <= 0:
        return [int((x1 + x2) * 0.5), int((y1 + y2) * 0.5)]
    return [int(x1 + max_loc[0]), int(y1 + max_loc[1])]


def negative_points(box: list[int], prompt_box: list[int], image_shape: tuple[int, int], scale: float) -> list[list[int]]:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = box
    bx1, by1, bx2, by2 = prompt_box
    cx, cy = int(round((x1 + x2) * 0.5)), int(round((y1 + y2) * 0.5))
    offset = max(5, int(round(8 * scale)))
    points = [
        [x1 - offset, cy],
        [x2 + offset, cy],
        [cx, y1 - offset],
        [cx, y2 + offset],
    ]
    safe = []
    for x, y in points:
        x = int(np.clip(x, bx1, min(bx2, w - 1)))
        y = int(np.clip(y, by1, min(by2, h - 1)))
        safe.append([x, y])
    return safe


def prompts_from_fused(fused: list[dict], edge_binary: np.ndarray, image_shape: tuple[int, int]) -> list[dict]:
    h, w = image_shape[:2]
    _, _, scale = scale_from_1080(w, h)
    prompts = []
    for item in fused:
        box = [int(v) for v in item["bbox"]]
        prompt_box = expand_box(box, w, h, scale)
        pos = positive_point(edge_binary, box)
        neg = negative_points(box, prompt_box, image_shape, scale)
        prompts.append({
            "prompt_id": len(prompts) + 1,
            "source": "shape_prior_fused",
            "candidate_id": int(item["candidate_id"]),
            "component_bbox": box,
            "box": prompt_box,
            "positive_points": [pos],
            "negative_points": neg,
            "proposal_score": float(item.get("matchshape_score") or item.get("chamfer_score") or 0.0),
            "fusion_sources": item["sources"],
        })
    return prompts


def fused_priority(item: dict) -> tuple:
    sources = set(item.get("sources", []))
    source_bonus = 0
    if "yellow" in sources:
        source_bonus += 2
    if "matchShapes" in sources:
        source_bonus += 2
    if "chamfer" in sources:
        source_bonus += 1
    matchshape = item.get("matchshape_score")
    chamfer = item.get("chamfer_score")
    yellow = item.get("yellow_pixel_ratio")
    return (
        -source_bonus,
        999.0 if matchshape is None else float(matchshape),
        999999.0 if chamfer is None else float(chamfer),
        -(0.0 if yellow is None else float(yellow)),
    )


def limit_fused_candidates(fused: list[dict], max_prompts: int | None) -> list[dict]:
    if max_prompts is None or max_prompts <= 0 or len(fused) <= max_prompts:
        return fused
    ranked = sorted(fused, key=fused_priority)
    kept = ranked[:max_prompts]
    for idx, item in enumerate(kept, 1):
        item["candidate_id"] = idx
    return kept

def yellow_candidates(
    image_bgr: np.ndarray,
    cfg: AppConfig,
    out_dir: str | Path,
) -> list[dict]:

    out_dir = Path(out_dir)

    h, w = image_bgr.shape[:2]
    sx, sy, s = scale_from_1080(w, h)
    c = cfg.shape_prior_sam

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([
        c.yellow_hue_low_relaxed,
        c.yellow_saturation_low_relaxed,
        c.yellow_value_low_relaxed
    ], dtype=np.uint8)

    upper = np.array([
        c.yellow_hue_high_relaxed,
        255,
        255
    ], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    cv2.imwrite(str(out_dir / "yellow_mask.png"), mask)

    oc = cfg.opencv_candidates
    k = max(1, int(round(oc.morph_kernel_1080 * s)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if oc.morph_open_iterations > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=int(oc.morph_open_iterations))
    if oc.morph_close_iterations > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=int(oc.morph_close_iterations))

    min_area = max(1, int(round(c.yellow_min_area_1080 * sx * sy)))
    max_area = max(min_area + 1, int(round(c.yellow_max_area_1080 * sx * sy)))
    min_w = max(1, int(round(c.yellow_min_width_1080 * sx)))
    min_h = max(1, int(round(c.yellow_min_height_1080 * sy)))
    max_w = max(min_w + 1, int(round(c.yellow_max_width_1080 * sx)))
    max_h = max(min_h + 1, int(round(c.yellow_max_height_1080 * sy)))
    taskbar_y = int(round(h * oc.taskbar_y_ratio))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, bw, bh = cv2.boundingRect(contour)
        if y >= taskbar_y:
            continue
        if area < min_area or area > max_area:
            continue
        if bw < min_w or bh < min_h or bw > max_w or bh > max_h:
            continue
        ratio = bw / max(bh, 1)
        if ratio < c.yellow_min_aspect_ratio or ratio > c.yellow_max_aspect_ratio:
            continue
        yellow_pixels = int(np.count_nonzero(mask[y:y + bh, x:x + bw]))
        yellow_pixel_ratio = yellow_pixels / max(bw * bh, 1)
        if yellow_pixel_ratio < c.yellow_min_pixel_ratio or yellow_pixel_ratio > c.yellow_max_pixel_ratio:
            continue
        rows.append({
            "candidate_id": len(rows) + 1,
            "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
            "contour_area": area,
            "yellow_pixel_ratio": float(yellow_pixel_ratio),
        })
    return rows


def post_shape_score(mask: np.ndarray, templates: list[dict]) -> tuple[float, int, list[float], list[int] | None]:
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 999.0, -1, [], None
    contour = max(contours, key=cv2.contourArea)
    norm = None
    x, y, w, h = cv2.boundingRect(contour)
    if w > 0 and h > 0:
        norm = contour.astype(np.float32).copy()
        norm[:, 0, 0] = (norm[:, 0, 0] - x) / float(w)
        norm[:, 0, 1] = (norm[:, 0, 1] - y) / float(h)
    if norm is None:
        return 999.0, -1, [], [int(x), int(y), int(x + w), int(y + h)]
    scores = [float(cv2.matchShapes(norm, t["contour"], cv2.CONTOURS_MATCH_I1, 0.0)) for t in templates]
    best_idx = int(np.argmin(scores))
    return float(scores[best_idx]), int(templates[best_idx]["template_id"]), scores, [int(x), int(y), int(x + w), int(y + h)]


def yellow_ratio(image_bgr: np.ndarray, mask: np.ndarray, cfg: AppConfig) -> float:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    c = cfg.shape_prior_sam
    lower = np.array([c.yellow_hue_low, c.yellow_saturation_low, c.yellow_value_low], dtype=np.uint8)
    upper = np.array([c.yellow_hue_high, c.yellow_saturation_high, c.yellow_value_high], dtype=np.uint8)
    yellow = cv2.inRange(hsv, lower, upper) > 0
    region = mask.astype(bool)
    total = int(region.sum())
    if total <= 0:
        return 0.0
    return float(np.count_nonzero(yellow & region) / total)


def post_filter_results(
    image_bgr: np.ndarray,
    sam_results: list[dict],
    prompts: list[dict],
    templates: list[dict],
    cfg: AppConfig,
    out_dir: Path,
) -> tuple[list[dict], list[dict]]:
    prompt_by_id = {p["prompt_id"]: p for p in prompts}
    c = cfg.shape_prior_sam
    rows = []
    kept = []
    masks_dir = out_dir / "sam" / "masks"
    for row in sam_results:
        mask_path = row.get("mask_path")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path else None
        if mask is None:
            continue
        mask_bool = mask > 0
        p = prompt_by_id.get(row["prompt_id"], {})
        shape_score, template_id, template_scores, contour_bbox = post_shape_score(mask_bool, templates)
        shape_similarity = float(np.exp(-shape_score / max(float(c.shape_similarity_scale), 1e-6)))
        yr = yellow_ratio(image_bgr, mask_bool, cfg)
        sam_score = float(row.get("sam_score", 0.0))
        folder_score = (
            float(c.folder_score_sam_weight) * sam_score
            + float(c.folder_score_shape_weight) * shape_similarity
            + float(c.folder_score_yellow_weight) * yr
        )
        keep = folder_score >= float(c.folder_score_threshold)
        reject_reason = "" if keep else "low_folder_score"
        enriched = {
            **row,
            "candidate_id": p.get("candidate_id"),
            "source": p.get("fusion_sources", []),
            "bbox_before_sam": p.get("component_bbox"),
            "bbox_after_sam": bbox_from_mask(mask_bool),
            "post_shape_score": shape_score,
            "post_shape_template_id": template_id,
            "post_shape_template_scores": template_scores,
            "shape_similarity": shape_similarity,
            "yellow_ratio": yr,
            "folder_score": float(folder_score),
            "keep": bool(keep),
            "reject_reason": reject_reason,
            "post_contour_bbox": contour_bbox,
        }
        rows.append(enriched)
        if keep:
            kept.append(enriched)
    save_json(out_dir / "sam_results.json", rows)
    return kept, rows


def draw_post_filter(image_bgr: np.ndarray, rows: list[dict], out_path: Path) -> None:
    vis = image_bgr.copy()
    h, w = image_bgr.shape[:2]
    _, _, scale = scale_from_1080(w, h)
    thickness = max(1, int(round(2 * scale)))
    font_scale = 0.45 * scale
    for item in rows:
        box = item.get("bbox_after_sam") or item.get("bbox_before_sam")
        if not box:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        color = (0, 255, 0) if item.get("keep") else (0, 0, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            vis,
            f"{float(item.get('folder_score', 0.0)):.2f}",
            (x1, max(int(round(14 * scale)), y1 - int(round(4 * scale)))),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            max(1, thickness // 2),
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_path), vis)
    
def draw_yellow_candidates(
    image_bgr: np.ndarray,
    items: list[dict],
    out_path: Path,
) -> None:
    vis = image_bgr.copy()
    h, w = image_bgr.shape[:2]
    _, _, scale = scale_from_1080(w, h)
    thickness = max(1, int(round(2 * scale)))
    font_scale = 0.45 * scale

    for idx, item in enumerate(items, 1):
        box = item.get("bbox")
        if not box:
            continue

        x1, y1, x2, y2 = [int(v) for v in box]

        cv2.rectangle(
            vis,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            thickness,
        )

        cv2.putText(
            vis,
            str(idx),
            (x1, max(int(round(14 * scale)), y1 - int(round(4 * scale)))),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 255),
            max(1, thickness // 2),
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), vis)


def parse_args():
    p = argparse.ArgumentParser(description="Shape-prior fused candidates -> SAM smoke test.")
    p.add_argument("--image", default="examples/1080p/1.jpg")
    p.add_argument("--folder-gt", default="assets/folder_templates")
    p.add_argument("--output-dir", default="outputs/shape_prior_sam_test")
    p.add_argument("--config", default="config.json")
    p.add_argument("--resolution", choices=["1080p", "2k", "4k"], default="1080p")
    p.add_argument("--matchshape-top", type=int, default=None)
    p.add_argument("--chamfer-top", type=int, default=None)
    p.add_argument("--edge-close-kernel", type=int, default=3)
    p.add_argument("--edge-close-iterations", type=int, default=1)
    p.add_argument("--wallpaper-bg", default="bg/bg_1080P.png")
    p.add_argument("--wallpaper-template-keep-threshold", type=float, default=None)
    p.add_argument("--max-wallpaper-recovery", type=int, default=None)
    p.add_argument("--max-prompts", type=int, default=None, help="Limit fused candidates sent to SAM. Default uses all fused candidates.")
    p.add_argument("--sam-single-mask", action="store_true", help="Disable SAM multimask output for faster runs.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()
    out_dir = ensure_dir(args.output_dir)
    cfg = AppConfig.from_json(args.config)

    # Production defaults come from the resolution-specific JSON config.
    # CLI values, when explicitly provided, override the config temporarily.
    wallpaper_template_keep_threshold = (
        float(args.wallpaper_template_keep_threshold)
        if args.wallpaper_template_keep_threshold is not None
        else float(cfg.wallpaper_refine.template_keep_threshold)
    )
    max_wallpaper_recovery = (
        int(args.max_wallpaper_recovery)
        if args.max_wallpaper_recovery is not None
        else int(cfg.wallpaper_refine.max_recovery)
    )
    cfg.image.target_size = RESOLUTION_MAP[args.resolution]
    cfg.image.preprocess_enabled = True
    cfg.sam.edge_score_weight = 0.1
    cfg.sam.sam_score_weight = 0.9
    if args.sam_single_mask:
        cfg.sam.multimask_output = False
    matchshape_top_n = int(args.matchshape_top if args.matchshape_top is not None else cfg.shape_prior_sam.matchshape_top)
    chamfer_top_n = int(args.chamfer_top if args.chamfer_top is not None else cfg.shape_prior_sam.chamfer_top)
    stage_times = {}

    def mark(stage: str, start: float) -> float:
        now = time.time()
        stage_times[stage] = round(now - start, 3)
        return now

    logger = logging.getLogger("shape_prior_sam_test")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())

    templates = load_templates(Path(args.folder_gt))
    draw_templates(templates, out_dir / "templates.png")

    stage_t = time.time()
    image_info = ImageProcessor(cfg).run(args.image, out_dir / "input")
    image_bgr = image_info["sam_input_bgr"]
    stage_t = mark("image_preprocess", stage_t)
    dex = DexiNedEngine(cfg, logger).predict(image_bgr, out_dir / "dexined")
    stage_t = mark("dexined", stage_t)
    _, _, image_scale = scale_from_1080(image_bgr.shape[1], image_bgr.shape[0])

    stats = template_stats(templates, image_bgr.shape[1], image_bgr.shape[0])
    edge_close_kernel = max(1, int(round(args.edge_close_kernel * image_scale)))
    if edge_close_kernel % 2 == 0:
        edge_close_kernel += 1
    connected = connect_edges(dex["edge_binary"], edge_close_kernel, args.edge_close_iterations)
    cv2.imwrite(str(out_dir / "edge_connected.png"), connected)
    #shape_candidates, filter_reasons = candidate_contours(connected, stats, image_scale)
    shape_candidates, filter_reasons = candidate_contours(
    connected,
    stats,
    image_scale,
    resolution=args.resolution,
    )
    raw_count = count_external_contours((dex["edge_binary"] > 0).astype(np.uint8) * 255)
    connected_count = count_external_contours(connected)
    stage_t = mark("edge_shape_candidates", stage_t)

    shape_matches = []
    for contour, box, area in shape_candidates:
        scores = [float(cv2.matchShapes(contour, t["contour"], cv2.CONTOURS_MATCH_I1, 0.0)) for t in templates]
        best_idx = int(np.argmin(scores))
        shape_matches.append({
            "bbox": box,
            "shape_score": float(scores[best_idx]),
            "template_scores": scores,
            "best_template_id": int(templates[best_idx]["template_id"]),
            "contour_area": float(area),
        })
    shape_matches.sort(key=lambda x: x["shape_score"])
    for idx, item in enumerate(shape_matches, 1):
        item["candidate_id"] = idx

    expand_ratio = float(cfg.shape_prior_sam.pre_fusion_expand_ratio)
    matchshape_input = expand_candidate_list(shape_matches[:matchshape_top_n], image_bgr.shape, expand_ratio)
    chamfer_base_size = estimate_chamfer_size(shape_matches)
    chamfer_matches = chamfer_match(dex["edge_binary"], templates, chamfer_base_size, scale=image_scale)
    chamfer_input = expand_candidate_list(chamfer_matches[:chamfer_top_n], image_bgr.shape, expand_ratio)
    draw_top_matches(image_bgr, matchshape_input, out_dir / "matchshape_top.png")
    draw_chamfer_matches(image_bgr, chamfer_input, out_dir / "chamfer_top.png")
    stage_t = mark("shape_chamfer_match", stage_t)

    yellow_rows = yellow_candidates(image_info["resized_bgr"], cfg, out_dir)
    yellow_input = expand_candidate_list(yellow_rows, image_bgr.shape, expand_ratio)
    fused = fuse_candidate_sets(matchshape_input, chamfer_input, yellow_input)
    fused_before_limit_count = len(fused)
    fused = limit_fused_candidates(fused, args.max_prompts)
    draw_yellow_candidates(image_bgr, yellow_input, out_dir / "yellow_candidates.png")
    save_fusion(image_bgr, fused, out_dir)
    prompts = prompts_from_fused(fused, dex["edge_binary"], image_bgr.shape)
    save_json(out_dir / "prompts.json", prompts)
    Visualizer().draw_prompts(image_bgr, prompts, out_dir / "prompt_visualization.png")
    stage_t = mark("candidate_fusion_prompt", stage_t)

    sam_results = SamEngine(cfg, logger).segment(image_bgr, prompts, dex["edge_binary"], out_dir / "sam", "dexined_rerank")
    Visualizer().draw_sam_results(image_bgr, sam_results, out_dir / "sam_visualization.png")
    stage_t = mark("sam", stage_t)
    kept_results, post_rows = post_filter_results(image_bgr, sam_results, prompts, templates, cfg, out_dir)

    draw_post_filter(image_bgr, post_rows, out_dir / "post_filter_visualization.png")
    stage_t = mark("post_filter", stage_t)

    cv_result = CVPostProcessor(cfg).run(image_bgr, kept_results, out_dir / "opencv")
    boxes_path = Path(cv_result["boxes_path"])
    boxes_json_path = Path(cv_result["boxes_json_path"])
    final_boxes_path = out_dir / "boxes.png"
    final_json_path = out_dir / "boxes.json"
    cv2.imwrite(str(final_boxes_path), cv2.imread(str(boxes_path)))
    save_json(final_json_path, cv_result["boxes"])
    stage_t = mark("cv_postprocess", stage_t)

    wallpaper_report = run_wallpaper_refine(
        image_bgr=image_info["resized_bgr"],
        base_boxes=cv_result["boxes"],
        bg_path=args.wallpaper_bg,
        resolution=args.resolution,
        out_dir=out_dir,
        image_path=args.image,
        wallpaper_template_keep_threshold=wallpaper_template_keep_threshold,
        max_wallpaper_recovery=max_wallpaper_recovery,
    )
    print(f"base final box count: {wallpaper_report['base_final_box_count']}")
    print(f"empty warm grid cell count: {wallpaper_report['empty_warm_grid_cell_count']}")
    print(f"raw recovery count: {wallpaper_report['raw_recovery_count']}")
    print(f"template verified recovery count: {wallpaper_report['template_verified_recovery_count']}")
    print(f"final refined box count: {wallpaper_report['final_refined_box_count']}")
    mark("wallpaper_refine", stage_t)

    report = {
        "image": args.image,
        "resolution": args.resolution,
        "raw_contour_count": raw_count,
        "connected_contour_count": connected_count,
        "filtered_contour_count": len(shape_matches),
        "filter_reasons": dict(filter_reasons),
        "matchshape_candidate_count": len(matchshape_input),
        "chamfer_candidate_count": len(chamfer_input),
        "matchshape_input_count": len(matchshape_input),
        "chamfer_input_count": len(chamfer_input),
        "yellow_candidate_count": len(yellow_input),
        "fused_candidate_count_before_limit": fused_before_limit_count,
        "fused_candidate_count": len(fused),
        "sam_result_count": len(sam_results),
        "post_filter_count": len(kept_results),
        "post_filter_keep_count": len(kept_results),
        "post_filter_reject_count": len(post_rows) - len(kept_results),
        "final_base_box_count": cv_result["final_box_count"],
        "final_box_count": cv_result["final_box_count"],
        "wallpaper_refine": {
            "base_final_box_count": wallpaper_report["base_final_box_count"],
            "empty_warm_grid_cell_count": wallpaper_report["empty_warm_grid_cell_count"],
            "raw_recovery_count": wallpaper_report["raw_recovery_count"],
            "template_verified_recovery_count": wallpaper_report["template_verified_recovery_count"],
            "final_refined_box_count": wallpaper_report["final_refined_box_count"],
            "wallpaper_template_keep_threshold": wallpaper_report["wallpaper_template_keep_threshold"],
            "max_wallpaper_recovery": wallpaper_report["max_wallpaper_recovery"],
        },
        "elapsed_sec": round(time.time() - t0, 3),
        "stage_times_sec": stage_times,
        "speed_settings": {
            "max_prompts": args.max_prompts,
            "sam_single_mask": bool(args.sam_single_mask),
            "sam_multimask_output": bool(cfg.sam.multimask_output),
        },
        "artifacts": {
            "matchshape_top": str(out_dir / "matchshape_top.png"),
            "chamfer_top": str(out_dir / "chamfer_top.png"),
            "yellow_candidates": str(out_dir / "yellow_candidates.png"),
            "fused_candidates_png": str(out_dir / "fused_candidates.png"),
            "fused_candidates_json": str(out_dir / "fused_candidates.json"),
            "prompt_visualization": str(out_dir / "prompt_visualization.png"),
            "sam_visualization": str(out_dir / "sam_visualization.png"),
            "sam_results": str(out_dir / "sam_results.json"),
            "post_filter_visualization": str(out_dir / "post_filter_visualization.png"),
            "boxes_png": str(final_boxes_path),
            "boxes_json": str(final_json_path),
            **wallpaper_report["artifacts"],
        },
    }
    save_json(out_dir / "run_report.json", report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
