from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.utils import ensure_dir, save_json


def bbox_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_distance(a: list[int], b: list[int]) -> float:
    ax, ay = (a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5
    bx, by = (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5
    return float(np.hypot(ax - bx, ay - by))


def should_merge(a: list[int], b: list[int]) -> bool:
    if bbox_iou(a, b) > 0.4:
        return True
    aw, ah = a[2] - a[0], a[3] - a[1]
    bw, bh = b[2] - b[0], b[3] - b[1]
    return center_distance(a, b) < 0.45 * min(max(aw, ah), max(bw, bh))


def union_box(boxes: list[list[int]]) -> list[int]:
    return [
        int(min(b[0] for b in boxes)),
        int(min(b[1] for b in boxes)),
        int(max(b[2] for b in boxes)),
        int(max(b[3] for b in boxes)),
    ]


def normalize_matchshape(item: dict) -> dict:
    return {
        "source": "matchShapes",
        "bbox": [int(x) for x in item["bbox"]],
        "shape_score": float(item["shape_score"]),
        "best_template_id": int(item["best_template_id"]),
        "candidate_id": int(item.get("candidate_id", 0)),
    }


def normalize_chamfer(item: dict) -> dict:
    return {
        "source": "chamfer",
        "bbox": [int(x) for x in item["bbox"]],
        "chamfer_score": float(item["chamfer_score"]),
        "mean_distance": float(item.get("mean_distance", item["chamfer_score"])),
        "edge_coverage": float(item["edge_coverage"]),
        "best_template_id": int(item["best_template_id"]),
        "candidate_id": int(item.get("candidate_id", 0)),
    }


def normalize_yellow(item: dict) -> dict:
    return {
        "source": "yellow",
        "bbox": [int(x) for x in item["bbox"]],
        "yellow_pixel_ratio": float(item.get("yellow_pixel_ratio", 0.0)),
        "contour_area": float(item.get("contour_area", 0.0)),
        "candidate_id": int(item.get("candidate_id", 0)),
    }


def fuse_candidates(matchshape: list[dict], chamfer: list[dict]) -> list[dict]:
    items = [normalize_matchshape(x) for x in matchshape] + [normalize_chamfer(x) for x in chamfer]
    return fuse_candidate_items(items)


def fuse_candidate_sets(matchshape: list[dict], chamfer: list[dict], yellow: list[dict]) -> list[dict]:
    items = (
        [normalize_matchshape(x) for x in matchshape]
        + [normalize_chamfer(x) for x in chamfer]
        + [normalize_yellow(x) for x in yellow]
    )
    return fuse_candidate_items(items)


def fuse_candidate_items(items: list[dict]) -> list[dict]:
    groups: list[dict] = []
    for item in items:
        merged = False
        for group in groups:
            if should_merge(item["bbox"], group["bbox"]):
                group["sources"].append(item)
                group["bbox"] = union_box([x["bbox"] for x in group["sources"]])
                merged = True
                break
        if not merged:
            groups.append({"bbox": item["bbox"], "sources": [item]})

    fused = []
    for idx, group in enumerate(groups, 1):
        sources = group["sources"]
        ms = [x for x in sources if x["source"] == "matchShapes"]
        ch = [x for x in sources if x["source"] == "chamfer"]
        best_ms = min(ms, key=lambda x: x["shape_score"]) if ms else None
        best_ch = min(ch, key=lambda x: x["chamfer_score"]) if ch else None
        yellow = [x for x in sources if x["source"] == "yellow"]
        best_yellow = max(yellow, key=lambda x: x["yellow_pixel_ratio"]) if yellow else None
        fused.append({
            "candidate_id": idx,
            "bbox": group["bbox"],
            "sources": sorted({x["source"] for x in sources}),
            "matchshape_score": None if best_ms is None else best_ms["shape_score"],
            "chamfer_score": None if best_ch is None else best_ch["chamfer_score"],
            "edge_coverage": None if best_ch is None else best_ch["edge_coverage"],
            "yellow_pixel_ratio": None if best_yellow is None else best_yellow["yellow_pixel_ratio"],
            "best_template_id": (best_ms or best_ch or {"best_template_id": -1})["best_template_id"],
            "members": sources,
        })
    return fused


def draw_fused_candidates(image_bgr: np.ndarray, fused: list[dict], out_path: str | Path) -> None:
    vis = image_bgr.copy()
    for item in fused:
        x1, y1, x2, y2 = item["bbox"]
        color = (0, 255, 255) if len(item["sources"]) > 1 else (0, 255, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{item['candidate_id']} {'+'.join(item['sources'])}"
        cv2.putText(vis, label, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    ensure_dir(Path(out_path).parent)
    cv2.imwrite(str(out_path), vis)


def save_fusion(image_bgr: np.ndarray, fused: list[dict], out_dir: str | Path) -> dict:
    out_dir = ensure_dir(out_dir)
    json_path = out_dir / "fused_candidates.json"
    png_path = out_dir / "fused_candidates.png"
    save_json(json_path, fused)
    draw_fused_candidates(image_bgr, fused, png_path)
    return {"json_path": str(json_path), "png_path": str(png_path)}
