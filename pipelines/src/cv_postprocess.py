from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import AppConfig
from .utils import bbox_iou, containment, ensure_dir, save_json, scale_from_1080


class CVPostProcessor:
    """Convert selected SAM masks into OpenCV contours and final rectangles."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def run(self, image_bgr: np.ndarray, sam_results: list[dict], out_dir: str | Path) -> dict:
        out_dir = ensure_dir(out_dir)
        h, w = image_bgr.shape[:2]
        sx, sy, s = scale_from_1080(w, h)
        c = self.cfg.cv_postprocess

        kernel_size = max(1, int(round(c.morph_kernel_1080 * s)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        min_area = max(1, int(round(c.min_area_1080 * sx * sy)))
        max_area = max(min_area + 1, int(round(c.max_area_1080 * sx * sy)))
        min_w = max(1, int(round(c.min_width_1080 * sx)))
        min_h = max(1, int(round(c.min_height_1080 * sy)))
        max_w = max(min_w + 1, int(round(c.max_width_1080 * sx)))
        max_h = max(min_h + 1, int(round(c.max_height_1080 * sy)))
        taskbar_y = int(round(h * c.taskbar_y_ratio))

        merged_mask = np.zeros((h, w), dtype=np.uint8)
        raw_boxes = []
        for row in sam_results:
            mask_path = row.get("mask_path")
            if not mask_path or not Path(mask_path).exists():
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.uint8) * 255
            if c.morph_open_iterations > 0:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=int(c.morph_open_iterations))
            if c.morph_close_iterations > 0:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=int(c.morph_close_iterations))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
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

            contour_box = [int(x), int(y), int(x + bw), int(y + bh)]
            box = self._desktop_cell_box(contour_box, w, h, sx, sy, taskbar_y)
            out_w, out_h = box[2] - box[0], box[3] - box[1]
            raw_boxes.append({
                "prompt_id": row.get("prompt_id"),
                "prompt_source": row.get("prompt_source", ""),
                "bbox": box,
                "contour_bbox": contour_box,
                "width": int(out_w),
                "height": int(out_h),
                "contour_width": int(bw),
                "contour_height": int(bh),
                "contour_area": area,
                "mask_area": int(np.count_nonzero(mask)),
                "sam_score": float(row.get("sam_score", 0.0)),
                "edge_alignment_score": float(row.get("edge_alignment_score", 0.0)),
                "final_score": float(row.get("final_score", 0.0)),
                "mask_path": str(mask_path),
                "_mask": mask,
            })

        raw_contours = [
            {k: v for k, v in item.items() if k != "_mask"}
            for item in raw_boxes
        ]

        boxes = self._deduplicate(raw_boxes)
        for item in boxes:
            merged_mask[item.pop("_mask") > 0] = 255

        thickness = max(1, int(round(c.draw_thickness_1080 * s)))
        boxed = image_bgr.copy()
        contour_vis = image_bgr.copy()
        for idx, item in enumerate(boxes, 1):
            x1, y1, x2, y2 = item["bbox"]
            cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            cv2.putText(
                boxed,
                str(idx),
                (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45 * s,
                (0, 255, 0),
                max(1, thickness // 2),
                cv2.LINE_AA,
            )
            mask = cv2.imread(item["mask_path"], cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), thickness)
            item["box_id"] = idx

        boxes_path = out_dir / "boxes.png"
        contours_path = out_dir / "contours.png"
        merged_path = out_dir / "combined_mask.png"
        json_path = out_dir / "boxes.json"
        cv2.imwrite(str(boxes_path), boxed)
        cv2.imwrite(str(contours_path), contour_vis)
        cv2.imwrite(str(merged_path), merged_mask)
        save_json(json_path, boxes)

        return {
            "raw_box_count": len(raw_boxes),
            "raw_contours": raw_contours,
            "final_box_count": len(boxes),
            "boxes": boxes,
            "boxes_path": str(boxes_path),
            "contours_path": str(contours_path),
            "combined_mask_path": str(merged_path),
            "boxes_json_path": str(json_path),
        }

    def _deduplicate(self, items: list[dict]) -> list[dict]:
        c = self.cfg.cv_postprocess
        ordered = sorted(items, key=lambda x: (x["final_score"], x["sam_score"]), reverse=True)
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

    # def _desktop_cell_box(
    #     self,
    #     contour_box: list[int],
    #     image_w: int,
    #     image_h: int,
    #     sx: float,
    #     sy: float,
    #     taskbar_y: int,
    # ) -> list[int]:
    #     c = self.cfg.cv_postprocess
    #     if not c.desktop_cell_expand_enabled:
    #         return contour_box

    #     x1, y1, x2, y2 = contour_box
    #     cx = (x1 + x2) * 0.5 + float(c.desktop_cell_center_dx_1080) * sx
    #     cy = (y1 + y2) * 0.5 + float(c.desktop_cell_center_dy_1080) * sy
    #     cell_w = max(1, int(round(c.desktop_cell_width_1080 * sx)))
    #     cell_h = max(1, int(round(c.desktop_cell_height_1080 * sy)))

    #     nx1 = int(round(cx - cell_w * 0.5))
    #     ny1 = int(round(cy - cell_h * 0.5))
    #     nx2 = nx1 + cell_w
    #     ny2 = ny1 + cell_h

    #     if nx1 < 0:
    #         nx2 -= nx1
    #         nx1 = 0
    #     if nx2 > image_w:
    #         shift = nx2 - image_w
    #         nx1 = max(0, nx1 - shift)
    #         nx2 = image_w
    #     if ny1 < 0:
    #         ny2 -= ny1
    #         ny1 = 0
    #     # max_y = min(image_h, taskbar_y)
    #     max_y = image_h
        
    #     if ny2 > max_y:
    #         shift = ny2 - max_y
    #         ny1 = max(0, ny1 - shift)
    #         ny2 = max_y
    #     return [int(nx1), int(ny1), int(nx2), int(ny2)]
    
    def _desktop_cell_box(
    self,
    contour_box: list[int],
    image_w: int,
    image_h: int,
    sx: float,
    sy: float,
    taskbar_y: int,
    ) -> list[int]:
        c = self.cfg.cv_postprocess

        if not c.desktop_cell_expand_enabled:
            return contour_box

        x1, y1, x2, y2 = contour_box

        cx = (
            (x1 + x2) * 0.5
            + float(c.desktop_cell_center_dx_1080) * sx
        )

        cy = (
            (y1 + y2) * 0.5
            + float(c.desktop_cell_center_dy_1080) * sy
        )

        cell_w = max(
            1,
            int(round(c.desktop_cell_width_1080 * sx))
        )

        cell_h = max(
            1,
            int(round(c.desktop_cell_height_1080 * sy))
        )

        nx1 = int(round(cx - cell_w * 0.5))
        ny1 = int(round(cy - cell_h * 0.5))
        nx2 = nx1 + cell_w
        ny2 = ny1 + cell_h

        # --------------------------------------------------
        # 4K, DPI = 100
        #
        # Current 4K desktop-cell boxes cover the folder icon
        # well but often stop before the filename.
        #
        # Extend only the bottom edge. Do NOT move the top
        # edge, otherwise the icon localization is changed.
        # --------------------------------------------------
        if image_w >= 3800 and image_h >= 2100:
            ny2 += int(c.bottom_expand_4k)

        if nx1 < 0:
            nx2 -= nx1
            nx1 = 0

        if nx2 > image_w:
            shift = nx2 - image_w
            nx1 = max(0, nx1 - shift)
            nx2 = image_w

        if ny1 < 0:
            ny2 -= ny1
            ny1 = 0

        # Do not restrict to taskbar_y here because a desktop
        # object's filename box may legitimately approach the
        # taskbar region.
        max_y = image_h

        if ny2 > max_y:
            ny2 = max_y

        return [
            int(nx1),
            int(ny1),
            int(nx2),
            int(ny2),
        ]
