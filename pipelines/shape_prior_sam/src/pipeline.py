from __future__ import annotations

from pathlib import Path
import time

from .config import AppConfig
from .cv_postprocess import CVPostProcessor
from .dexined_engine import DexiNedEngine
from .image_processor import ImageProcessor
from .prompt_generator import PromptGenerator
from .sam_engine import SamEngine
from .utils import ensure_dir, save_json
from .visualizer import Visualizer


class DexiNedSamPipeline:
    def __init__(self, cfg: AppConfig, run_dir: Path, logger, prompt_mode: str = "dexined"):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.logger = logger
        self.prompt_mode = prompt_mode
        self.image_processor = ImageProcessor(cfg)
        self.dexined = DexiNedEngine(cfg, logger)
        self.prompt_generator = PromptGenerator(cfg)
        self.sam = SamEngine(cfg, logger)
        self.cv_post = CVPostProcessor(cfg)
        self.visualizer = Visualizer()

    def run(self, image_path: str | Path) -> dict:
        t0 = time.time()
        input_dir = ensure_dir(self.run_dir / "input")
        dex_dir = ensure_dir(self.run_dir / "dexined")
        prompt_dir = ensure_dir(self.run_dir / "prompts")
        sam_dir = ensure_dir(self.run_dir / "sam")
        cv_dir = ensure_dir(self.run_dir / "opencv")

        self.logger.info("1/6 Resize + optional preprocessing")
        image_info = self.image_processor.run(image_path, input_dir)
        sam_input = image_info["sam_input_bgr"]

        self.logger.info("2/6 OpenCV yellow candidates")
        candidate_info = self.prompt_generator.generate_candidates(sam_input, cv_dir)
        candidates = candidate_info["candidates"]
        self.logger.info("OpenCV candidates: %d", len(candidates))

        self.logger.info("3/6 DexiNed edge prior")
        dex = self.dexined.predict(sam_input, dex_dir)
        if dex["edge_probability"].shape[:2] != sam_input.shape[:2]:
            raise RuntimeError(
                f"DexiNed edge size {dex['edge_probability'].shape[:2]} != SAM input size {sam_input.shape[:2]}"
            )

        self.logger.info("4/6 Generate SAM prompts")
        prompts = self.prompt_generator.generate_prompts(
            candidates,
            sam_input.shape,
            dex["edge_probability"],
            candidate_info["yellow_mask"],
            self.prompt_mode,
            prompt_dir,
        )
        self.visualizer.draw_prompts(sam_input, prompts, prompt_dir / "prompt_visualization.png")
        self.logger.info("Generated %d prompt groups", len(prompts))

        self.logger.info("5/6 SAM segmentation")
        results = self.sam.segment(sam_input, prompts, dex["edge_binary"], sam_dir, self.prompt_mode) if prompts else []
        self.visualizer.draw_sam_results(sam_input, results, sam_dir / "sam_visualization.png")

        self.logger.info("6/6 OpenCV contour extraction + bbox drawing")
        cv_result = self.cv_post.run(sam_input, results, cv_dir) if self.cfg.cv_postprocess.enabled else {
            "raw_box_count": 0,
            "final_box_count": 0,
            "boxes": [],
            "boxes_path": "",
            "contours_path": "",
            "combined_mask_path": "",
            "boxes_json_path": "",
        }

        report = {
            "image": str(image_path),
            "target_size": list(self.cfg.image.target_size),
            "preprocess_enabled": bool(self.cfg.image.preprocess_enabled),
            "prompt_mode": self.prompt_mode,
            "edge_pixel_ratio": dex["edge_pixel_ratio"],
            "opencv_candidate_count": len(candidates),
            "prompt_count": len(prompts),
            "sam_result_count": len(results),
            "final_box_count": cv_result["final_box_count"],
            "elapsed_sec": round(time.time() - t0, 3),
            "artifacts": {
                "resized": image_info["resized_path"],
                "sam_input": image_info["sam_input_path"],
                "preprocessed": image_info["sam_input_path"],
                "yellow_mask": candidate_info["yellow_mask_path"],
                "candidate_visualization": candidate_info["candidate_visualization_path"],
                "candidates": candidate_info["candidates_json_path"],
                "edge_probability": dex["edge_probability_path"],
                "edge_binary": dex["edge_binary_path"],
                "edge_overlay": dex["edge_overlay_path"],
                "prompt_visualization": str(prompt_dir / "prompt_visualization.png"),
                "sam_visualization": str(sam_dir / "sam_visualization.png"),
                "masks": str(sam_dir / "masks"),
                "sam_results": str(sam_dir / "sam_results.json"),
                "opencv_boxes": cv_result["boxes_path"],
                "opencv_contours": cv_result["contours_path"],
                "opencv_combined_mask": cv_result["combined_mask_path"],
                "opencv_boxes_json": cv_result["boxes_json_path"],
            },
        }
        save_json(self.run_dir / "run_report.json", report)
        self.logger.info(
            "Finished: prompts=%d SAM=%d boxes=%d elapsed=%.3fs",
            len(prompts),
            len(results),
            cv_result["final_box_count"],
            report["elapsed_sec"],
        )
        return report
