from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import AppConfig
from .utils import bbox_from_mask, choose_device, edge_alignment_score, ensure_dir, save_json, scale_from_1080


class SamEngine:
    def __init__(self, cfg: AppConfig, logger):
        self.cfg = cfg
        self.logger = logger
        self.device = choose_device(cfg.sam.device)
        self.processor = None
        self.model = None

    def _load(self):
        if self.model is not None:
            return
        from transformers import SamModel, SamProcessor

        model_path = Path(self.cfg.sam.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"SAM model directory not found: {model_path}")
        self.processor = SamProcessor.from_pretrained(str(model_path), local_files_only=True)
        self.model = SamModel.from_pretrained(str(model_path), local_files_only=True).to(self.device)
        self.model.eval()
        self.logger.info("SAM loaded from %s on %s", model_path, self.device)

    @staticmethod
    def _torch_from_array(torch, arr: np.ndarray, dtype):
        arr = np.ascontiguousarray(arr)
        tensor = torch.frombuffer(arr.tobytes(), dtype=dtype)
        return tensor.reshape(*arr.shape)

    @staticmethod
    def _post_process_masks(torch, pred_masks, original_size, reshaped_size, threshold: float):
        import torch.nn.functional as F

        masks = pred_masks.detach().cpu()
        while masks.ndim > 4 and masks.shape[0] == 1:
            masks = masks[0]
        if masks.ndim == 4 and masks.shape[0] == 1:
            masks = masks[0]
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        if masks.ndim == 3:
            masks = masks.unsqueeze(0)

        masks = F.interpolate(masks.float(), size=(1024, 1024), mode="bilinear", align_corners=False)
        resized_h, resized_w = int(reshaped_size[0]), int(reshaped_size[1])
        masks = masks[..., :resized_h, :resized_w]
        orig_h, orig_w = int(original_size[0]), int(original_size[1])
        masks = F.interpolate(masks, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        masks = masks[0] if masks.shape[0] == 1 else masks
        return masks > float(threshold)

    @staticmethod
    def _mask_to_numpy(mask_tensor) -> np.ndarray:
        try:
            return mask_tensor.detach().to("cpu", dtype=__import__("torch").uint8).numpy().astype(bool)
        except Exception:
            return np.array(mask_tensor.to(__import__("torch").uint8).tolist(), dtype=bool)

    def segment(
        self,
        image_bgr: np.ndarray,
        prompts: list[dict],
        edge_binary: np.ndarray,
        out_dir: str | Path,
        prompt_mode: str = "dexined",
    ) -> list[dict]:
        self._load()
        import torch

        out_dir = ensure_dir(out_dir)
        masks_dir = ensure_dir(out_dir / "masks")
        all_masks_dir = ensure_dir(out_dir / "all_multimasks") if self.cfg.output.save_all_multimasks else None
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(image_rgb)
        h, w = image_bgr.shape[:2]
        _, _, scale = scale_from_1080(w, h)
        min_area = max(1, int(round(self.cfg.sam.min_mask_area_1080 * scale * scale)))
        sigma = max(0.5, float(self.cfg.sam.edge_match_sigma_1080) * scale)

        # Encode image once; decode prompts one by one for maximum compatibility
        # across transformers versions.
        base_inputs = self.processor(images=pil, return_tensors="pt")
        pixel_values = base_inputs["pixel_values"].to(self.device, dtype=torch.float32)
        with torch.no_grad():
            image_embeddings = self.model.get_image_embeddings(pixel_values)

        results = []
        for prompt in prompts:
            pts = prompt["positive_points"] + prompt["negative_points"]
            labels = [1] * len(prompt["positive_points"]) + [0] * len(prompt["negative_points"])

            encoded = self.processor(
                images=pil,
                input_points=[[pts]],
                input_labels=[[labels]],
                input_boxes=[[prompt["box"]]],
                return_tensors="pt",
            )

            kwargs = {
                "image_embeddings": image_embeddings,
                "input_points": encoded["input_points"].to(self.device, dtype=torch.float32),
                "input_labels": encoded["input_labels"].to(self.device, dtype=torch.int64),
                "input_boxes": encoded["input_boxes"].to(self.device, dtype=torch.float32),
                "multimask_output": bool(self.cfg.sam.multimask_output),
            }
            with torch.no_grad():
                outputs = self.model(**kwargs)

            post = self._post_process_masks(
                torch,
                outputs.pred_masks,
                encoded["original_sizes"][0],
                encoded["reshaped_input_sizes"][0],
                float(self.cfg.sam.mask_threshold),
            )

            scores = outputs.iou_scores.detach().float().cpu().squeeze()
            if scores.ndim == 0:
                scores = scores.unsqueeze(0)
            elif scores.ndim > 1:
                scores = scores.reshape(-1)

            candidates = []
            for m_idx in range(post.shape[0]):
                mask = self._mask_to_numpy(post[m_idx])
                sam_score = float(scores[min(m_idx, len(scores) - 1)].item())
                edge_score = edge_alignment_score(mask, edge_binary, sigma) if prompt_mode in {"dexined", "dexined_rerank"} else 0.0
                if prompt_mode == "dexined":
                    final_score = (
                        float(self.cfg.sam.sam_score_weight) * sam_score
                        + float(self.cfg.sam.edge_score_weight) * edge_score
                    )
                elif prompt_mode == "dexined_rerank":
                    edge_weight = float(np.clip(self.cfg.sam.edge_score_weight, 0.0, 1.0))
                    final_score = (1.0 - edge_weight) * sam_score + edge_weight * edge_score
                else:
                    final_score = sam_score
                area = int(mask.sum())
                candidate = {
                    "mask_index": int(m_idx),
                    "sam_score": sam_score,
                    "edge_score": edge_score,
                    "edge_alignment_score": edge_score,
                    "final_score": float(final_score),
                    "area": area,
                    "mask": mask,
                }
                candidates.append(candidate)
                if all_masks_dir is not None:
                    cv2.imwrite(
                        str(all_masks_dir / f"prompt_{prompt['prompt_id']:04d}_mask_{m_idx}.png"),
                        mask.astype(np.uint8) * 255,
                    )

            if not candidates:
                continue
            valid = [x for x in candidates if x["area"] >= min_area]
            best = max(valid or candidates, key=lambda x: x["final_score"])
            best_index = int(best["mask_index"])
            multimasks = []
            for item in candidates:
                multimasks.append({
                    "mask_index": int(item["mask_index"]),
                    "sam_score": float(item["sam_score"]),
                    "edge_score": float(item["edge_score"]),
                    "edge_alignment_score": float(item["edge_alignment_score"]),
                    "final_score": float(item["final_score"]),
                    "area": int(item["area"]),
                    "selected": int(item["mask_index"]) == best_index,
                })
            mask = best.pop("mask")
            bbox = bbox_from_mask(mask)
            mask_path = masks_dir / f"mask_{prompt['prompt_id']:04d}.png"
            if self.cfg.output.save_individual_masks:
                cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
                mask_path_str = str(mask_path)
            else:
                mask_path_str = ""

            results.append({
                "prompt_id": prompt["prompt_id"],
                "prompt_source": prompt.get("source", ""),
                "prompt_mode": prompt_mode,
                "prompt_box": prompt["box"],
                "positive_points": prompt["positive_points"],
                "negative_points": prompt["negative_points"],
                "point_labels": labels,
                "sam_bbox": bbox,
                "mask_area": int(mask.sum()),
                "mask_path": mask_path_str,
                "multimasks": multimasks,
                **best,
            })

        save_json(out_dir / "sam_results.json", results)
        return results
