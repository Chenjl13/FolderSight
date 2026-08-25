from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


RESOLUTION_DIRS = {
    "1080p": "1080p",
    "2k": "2k",
    "4k": "4k",
}

IOU_THRESHOLDS = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]


@dataclass
class BoxItem:
    bbox: list[float]
    score: float
    item_id: str


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def collect_gt_boxes(obj) -> list[BoxItem]:
    if isinstance(obj, dict) and isinstance(obj.get("objects"), list):
        return [
            BoxItem([float(v) for v in row["bbox"]], 1.0, str(row.get("id", idx + 1)))
            for idx, row in enumerate(obj["objects"])
            if isinstance(row.get("bbox"), list) and len(row["bbox"]) == 4
        ]
    if isinstance(obj, dict) and isinstance(obj.get("folders"), list):
        return [
            BoxItem([float(v) for v in row["bbox"]], 1.0, str(row.get("folder_id", idx + 1)))
            for idx, row in enumerate(obj["folders"])
            if isinstance(row.get("bbox"), list) and len(row["bbox"]) == 4
        ]
    boxes: list[BoxItem] = []
    if isinstance(obj, dict):
        if isinstance(obj.get("bbox"), list) and len(obj["bbox"]) == 4:
            boxes.append(BoxItem([float(v) for v in obj["bbox"]], 1.0, str(obj.get("id", len(boxes) + 1))))
        for value in obj.values():
            boxes.extend(collect_gt_boxes(value))
    elif isinstance(obj, list):
        for value in obj:
            boxes.extend(collect_gt_boxes(value))
    return boxes


def collect_pred_boxes(obj, box_key: str = "bbox") -> list[BoxItem]:
    rows = obj if isinstance(obj, list) else obj.get("boxes", []) if isinstance(obj, dict) else []
    boxes = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        box = row.get(box_key)
        if not (isinstance(box, list) and len(box) == 4):
            box = row.get("bbox")
        if not (isinstance(box, list) and len(box) == 4):
            continue
        score = row.get("final_score", row.get("sam_score", row.get("template_similarity", 1.0)))
        boxes.append(BoxItem([float(v) for v in box], float(score), str(row.get("box_id", idx + 1))))
    return boxes


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def greedy_match(preds: list[BoxItem], gts: list[BoxItem], threshold: float) -> list[dict]:
    pairs = []
    for pi, pred in enumerate(preds):
        for gi, gt in enumerate(gts):
            score = iou(pred.bbox, gt.bbox)
            if score >= threshold:
                pairs.append((score, pi, gi))
    pairs.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches = []
    for score, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append({"pred_index": pi, "gt_index": gi, "iou": float(score)})
    return matches


def box_center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5


def point_in_box(point: tuple[float, float], box: list[float]) -> bool:
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def center_hit_score(pred: BoxItem, gt: BoxItem) -> float:
    pred_center = box_center(pred.bbox)
    gt_center = box_center(gt.bbox)
    if point_in_box(pred_center, gt.bbox) or point_in_box(gt_center, pred.bbox):
        return 1.0
    return 0.0


def greedy_center_match(preds: list[BoxItem], gts: list[BoxItem]) -> list[dict]:
    pairs = []
    for pi, pred in enumerate(preds):
        for gi, gt in enumerate(gts):
            score = center_hit_score(pred, gt)
            if score > 0:
                pairs.append((iou(pred.bbox, gt.bbox), pi, gi))
    pairs.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches = []
    for score, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append({"pred_index": pi, "gt_index": gi, "iou": float(score)})
    return matches


def precision_recall(preds: list[BoxItem], gts: list[BoxItem], threshold: float, match_mode: str = "iou") -> dict:
    matches = greedy_center_match(preds, gts) if match_mode == "center" else greedy_match(preds, gts, threshold)
    tp = len(matches)
    fp = len(preds) - tp
    fn = len(gts) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    detection_accuracy = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    matched = [m["iou"] for m in matches]
    best_gt = [max((iou(pred.bbox, gt.bbox) for pred in preds), default=0.0) for gt in gts]
    return {
        "iou_threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "accuracy": detection_accuracy,
        "f1": f1,
        "mean_matched_iou": float(np.mean(matched)) if matched else 0.0,
        "mean_best_iou_per_gt": float(np.mean(best_gt)) if best_gt else 0.0,
    }


def ap_at_iou(preds: list[BoxItem], gts: list[BoxItem], threshold: float) -> tuple[float, list[dict]]:
    ordered = sorted(preds, key=lambda x: x.score, reverse=True)
    used_g: set[int] = set()
    points = []
    tp = 0
    fp = 0
    for rank, pred in enumerate(ordered, 1):
        best_iou = 0.0
        best_g = -1
        for gi, gt in enumerate(gts):
            if gi in used_g:
                continue
            score = iou(pred.bbox, gt.bbox)
            if score > best_iou:
                best_iou = score
                best_g = gi
        if best_iou >= threshold and best_g >= 0:
            used_g.add(best_g)
            tp += 1
            matched = True
        else:
            fp += 1
            matched = False
        points.append({
            "rank": rank,
            "score": pred.score,
            "tp": tp,
            "fp": fp,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / len(gts) if gts else 0.0,
            "matched": matched,
            "best_iou": best_iou,
        })

    if not gts:
        return 0.0, points
    recalls = np.array([0.0] + [p["recall"] for p in points] + [1.0], dtype=np.float32)
    precisions = np.array([1.0] + [p["precision"] for p in points] + [0.0], dtype=np.float32)
    for idx in range(len(precisions) - 2, -1, -1):
        precisions[idx] = max(precisions[idx], precisions[idx + 1])
    ap = 0.0
    for idx in range(1, len(recalls)):
        if recalls[idx] != recalls[idx - 1]:
            ap += float((recalls[idx] - recalls[idx - 1]) * precisions[idx])
    return ap, points


def aggregate_metrics(samples: list[tuple[str, list[BoxItem], list[BoxItem]]], threshold: float, match_mode: str = "iou") -> dict:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    matched_ious = []
    best_gt_ious = []
    for _sample_id, preds, gts in samples:
        matches = greedy_center_match(preds, gts) if match_mode == "center" else greedy_match(preds, gts, threshold)
        total_tp += len(matches)
        total_fp += len(preds) - len(matches)
        total_fn += len(gts) - len(matches)
        matched_ious.extend([m["iou"] for m in matches])
        best_gt_ious.extend([max((iou(pred.bbox, gt.bbox) for pred in preds), default=0.0) for gt in gts])
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    accuracy = total_tp / (total_tp + total_fp + total_fn) if total_tp + total_fp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "iou_threshold": threshold,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "mean_best_iou_per_gt": float(np.mean(best_gt_ious)) if best_gt_ious else 0.0,
    }


def aggregate_ap(samples: list[tuple[str, list[BoxItem], list[BoxItem]]], threshold: float) -> tuple[float, list[dict]]:
    ap_values = []
    merged_points = []
    for sample_id, preds, gts in samples:
        ap, points = ap_at_iou(preds, gts, threshold)
        if gts:
            ap_values.append(ap)
        for point in points:
            row = dict(point)
            row["sample_id"] = sample_id
            merged_points.append(row)
    merged_points.sort(key=lambda row: float(row["score"]), reverse=True)
    return (float(np.mean(ap_values)) if ap_values else 0.0), merged_points


def draw_comparison(image_path: Path, preds: list[BoxItem], gts: list[BoxItem], out_path: Path, threshold: float) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    matches = greedy_match(preds, gts, threshold)
    matched_p = {m["pred_index"] for m in matches}
    matched_g = {m["gt_index"] for m in matches}
    vis = image.copy()
    scale = max(1.0, image.shape[1] / 1920.0)
    thick = max(2, int(round(2 * scale)))
    font = 0.45 * scale
    for idx, gt in enumerate(gts):
        color = (0, 180, 0) if idx in matched_g else (0, 0, 255)
        x1, y1, x2, y2 = [int(round(v)) for v in gt.bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
    for idx, pred in enumerate(preds):
        color = (255, 0, 0) if idx in matched_p else (0, 165, 255)
        x1, y1, x2, y2 = [int(round(v)) for v in pred.bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
        cv2.putText(vis, pred.item_id, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, font, color, max(1, thick // 2), cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_outputs(out_dir: Path, summary_rows: list[dict], pr_points_by_key: dict[str, list[dict]], ious_by_resolution: dict[str, list[float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable, skip charts: {exc}")
        return

    charts = out_dir / "charts"
    charts.mkdir(parents=True, exist_ok=True)

    res_rows = [r for r in summary_rows if r["scope"] == "resolution" and abs(float(r["iou_threshold"]) - 0.5) < 1e-9]
    if res_rows:
        labels = [r["resolution"] for r in res_rows]
        x = np.arange(len(labels))
        width = 0.25
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x - width, [r["precision"] for r in res_rows], width, label="Precision@0.5")
        ax.bar(x, [r["recall"] for r in res_rows], width, label="Recall@0.5")
        ax.bar(x + width, [r["f1"] for r in res_rows], width, label="F1@0.5")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x, labels)
        ax.set_ylabel("score")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(charts / "metrics_by_resolution.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for key, points in pr_points_by_key.items():
        if not points:
            continue
        ax.plot([p["recall"] for p in points], [p["precision"] for p in points], label=key)
        plotted = True
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend()
    fig.tight_layout()
    fig.savefig(charts / "precision_recall_curve_iou50.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for resolution, values in ious_by_resolution.items():
        if values:
            ax.hist(values, bins=np.linspace(0, 1, 21), alpha=0.45, label=resolution)
            plotted = True
    ax.set_xlabel("matched IoU at threshold 0.5")
    ax.set_ylabel("count")
    if plotted:
        ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(charts / "matched_iou_histogram.png", dpi=160)
    plt.close(fig)


def numeric_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10**9, path.name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate dataset predictions against input/<resolution>/gt JSON files.")
    p.add_argument("--input-root", default="input")
    p.add_argument("--pred-root", default="outputs/dataset_eval")
    p.add_argument("--output-dir", default="outputs/dataset_eval/evaluation")
    p.add_argument("--resolutions", nargs="+", choices=["1080p", "2k", "4k"], default=["1080p", "2k", "4k"])
    p.add_argument("--pred-file", default="final_refined_boxes.json")
    p.add_argument("--pred-box-key", default="bbox", help="Prediction box key to evaluate, e.g. bbox or contour_bbox. Falls back to bbox per row.")
    p.add_argument("--match-iou", type=float, default=0.5)
    p.add_argument("--match-mode", choices=["iou", "center"], default="iou")
    p.add_argument("--visualize-limit", type=int, default=30)
    p.add_argument("--include-missing", action="store_true", help="Also list GT files without predictions in sample_metrics.csv.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    pred_root = Path(args.pred_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = []
    threshold_rows = []
    samples_by_resolution: dict[str, list[tuple[str, list[BoxItem], list[BoxItem]]]] = {}
    matched_ious: dict[str, list[float]] = {}

    for resolution in args.resolutions:
        res_dir = RESOLUTION_DIRS[resolution]
        gt_dir = input_root / res_dir / "gt"
        image_dir = input_root / res_dir
        gt_files = sorted(gt_dir.glob("*.json"), key=numeric_key)
        visualized = 0
        for gt_path in gt_files:
            sample_id = gt_path.stem
            pred_path = pred_root / resolution / sample_id / args.pred_file
            if not pred_path.exists():
                if args.include_missing:
                    sample_rows.append({
                        "resolution": resolution,
                        "sample_id": sample_id,
                        "status": "missing_prediction",
                        "gt_count": len(collect_gt_boxes(load_json(gt_path))),
                        "pred_count": 0,
                        "precision@0.5": 0.0,
                        "recall@0.5": 0.0,
                        "accuracy@0.5": 0.0,
                        "f1@0.5": 0.0,
                        "mean_matched_iou@0.5": 0.0,
                    })
                continue
            gts = collect_gt_boxes(load_json(gt_path))
            preds = collect_pred_boxes(load_json(pred_path), args.pred_box_key)
            samples_by_resolution.setdefault(resolution, []).append((sample_id, preds, gts))
            metric = precision_recall(preds, gts, args.match_iou, args.match_mode)
            sample_rows.append({
                "resolution": resolution,
                "sample_id": sample_id,
                "status": "ok",
                "gt_count": len(gts),
                "pred_count": len(preds),
                "precision@0.5": metric["precision"],
                "recall@0.5": metric["recall"],
                "accuracy@0.5": metric["accuracy"],
                "f1@0.5": metric["f1"],
                "mean_matched_iou@0.5": metric["mean_matched_iou"],
            })
            matches = greedy_center_match(preds, gts) if args.match_mode == "center" else greedy_match(preds, gts, args.match_iou)
            matched_ious.setdefault(resolution, []).extend([m["iou"] for m in matches])
            if visualized < args.visualize_limit:
                image_path = image_dir / f"{sample_id}.jpg"
                draw_comparison(image_path, preds, gts, out_dir / "visualizations" / resolution / f"{sample_id}.png", args.match_iou)
                visualized += 1

    summary_rows = []
    pr_points_by_key = {}
    map_values_by_resolution = {}
    for resolution in args.resolutions:
        samples = samples_by_resolution.get(resolution, [])
        ap_values = []
        for threshold in IOU_THRESHOLDS:
            row = aggregate_metrics(samples, threshold, args.match_mode)
            ap, points = aggregate_ap(samples, threshold)
            ap_values.append(ap)
            row.update({
                "scope": "resolution",
                "resolution": resolution,
                "gt_count": sum(len(gts) for _sid, _preds, gts in samples),
                "pred_count": sum(len(preds) for _sid, preds, _gts in samples),
                "ap": ap,
            })
            summary_rows.append(row)
            if threshold == 0.5:
                pr_points_by_key[resolution] = points

        summary_rows.append({
            "scope": "resolution_map",
            "resolution": resolution,
            "iou_threshold": "0.50:0.95",
            "gt_count": sum(len(gts) for _sid, _preds, gts in samples),
            "pred_count": sum(len(preds) for _sid, preds, _gts in samples),
            "tp": "",
            "fp": "",
            "fn": "",
            "precision": "",
            "recall": "",
            "f1": "",
            "accuracy": "",
            "mean_matched_iou": "",
            "mean_best_iou_per_gt": "",
            "ap": float(np.mean(ap_values)) if ap_values else 0.0,
        })
        map_values_by_resolution[resolution] = ap_values

    all_samples = [sample for rows in samples_by_resolution.values() for sample in rows]
    for threshold in IOU_THRESHOLDS:
        row = aggregate_metrics(all_samples, threshold, args.match_mode)
        ap, points = aggregate_ap(all_samples, threshold)
        row.update({
            "scope": "all",
            "resolution": "all",
            "gt_count": sum(len(gts) for _sid, _preds, gts in all_samples),
            "pred_count": sum(len(preds) for _sid, preds, _gts in all_samples),
            "ap": ap,
        })
        summary_rows.append(row)
        if threshold == 0.5:
            pr_points_by_key["all"] = points

    write_csv(out_dir / "sample_metrics.csv", sample_rows)
    write_csv(out_dir / "summary_metrics.csv", summary_rows)
    save_json(out_dir / "summary_metrics.json", summary_rows)
    plot_outputs(out_dir, summary_rows, pr_points_by_key, matched_ious)

    ok = [r for r in sample_rows if r["status"] == "ok"]
    print(json.dumps({
        "evaluated_samples": len(ok),
        "missing_predictions": len(sample_rows) - len(ok),
        "summary_csv": str(out_dir / "summary_metrics.csv"),
        "sample_csv": str(out_dir / "sample_metrics.csv"),
        "charts": str(out_dir / "charts"),
        "visualizations": str(out_dir / "visualizations"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
