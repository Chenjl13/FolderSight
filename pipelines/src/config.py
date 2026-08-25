from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json


RESOLUTION_MAP = {
    "1080p": (1920, 1080),
    "2k": (2560, 1440),
    "4k": (3840, 2160),
}


@dataclass
class ImageConfig:
    target_size: tuple[int, int] = (1920, 1080)
    preprocess_enabled: bool = True
    bilateral_d: int = 5
    bilateral_sigma_color: float = 45.0
    bilateral_sigma_space: float = 45.0
    saturation_gain: float = 1.18
    saturation_bias: float = 4.0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)


@dataclass
class DexiNedConfig:
    # Local checkout of the official DexiNed repository.
    repo_path: str = "models/dexined"
    # Leave empty to auto-discover a .pth/.pt file under the DexiNed checkpoints directory.
    checkpoint_path: str = "models/dexined/10_model.pth"
    device: str = "auto"
    edge_threshold: float = 0.30
    # Official PyTorch DexiNed input normalization is BGR mean subtraction.
    mean_bgr: tuple[float, float, float] = (103.939, 116.779, 123.68)
    pad_multiple: int = 16


@dataclass
class OpenCVCandidateConfig:
    hue_low: int = 15
    hue_high: int = 45
    saturation_low: int = 55
    saturation_high: int = 255
    value_low: int = 70
    value_high: int = 255
    morph_kernel_1080: int = 3
    morph_open_iterations: int = 0
    morph_close_iterations: int = 1
    morph_dilate_iterations: int = 1
    min_area_1080: int = 120
    max_area_1080: int = 3500
    min_width_1080: int = 18
    min_height_1080: int = 10
    max_width_1080: int = 80
    max_height_1080: int = 65
    min_yellow_pixel_ratio: float = 0.18
    max_yellow_pixel_ratio: float = 0.92
    min_aspect_ratio: float = 0.12
    max_aspect_ratio: float = 8.0
    taskbar_y_ratio: float = 0.925
    nms_iou: float = 0.72
    containment_ratio: float = 0.92
    max_candidates: int = 260
    draw_thickness_1080: int = 2


@dataclass
class PromptConfig:
    box_expand_ratio: float = 0.22
    min_box_expand_1080: int = 7
    positive_search_margin_ratio: float = 0.08
    max_prompts: int = 260
    negative_directions: int = 8
    negative_edge_threshold: float = 0.35
    negative_search_band_1080: int = 3
    negative_outside_offset_1080: int = 4
    negative_safe_margin_1080: int = 5
    negative_search_distance_1080: int = 24


@dataclass
class SamConfig:
    model_path: str = "models/sam"
    device: str = "auto"
    multimask_output: bool = True
    mask_threshold: float = 0.0
    min_mask_area_1080: int = 60
    sam_score_weight: float = 0.70
    edge_score_weight: float = 0.30
    edge_match_sigma_1080: float = 2.5


@dataclass
class CVPostprocessConfig:
    enabled: bool = True
    morph_kernel_1080: int = 3
    morph_open_iterations: int = 0
    morph_close_iterations: int = 1
    min_area_1080: int = 120
    max_area_1080: int = 18000
    min_width_1080: int = 8
    min_height_1080: int = 8
    max_width_1080: int = 165
    max_height_1080: int = 165
    min_aspect_ratio: float = 0.18
    max_aspect_ratio: float = 5.5
    taskbar_y_ratio: float = 0.925
    nms_iou: float = 0.60
    containment_ratio: float = 0.90
    draw_thickness_1080: int = 2
    desktop_cell_expand_enabled: bool = True
    desktop_cell_width_1080: int = 74
    desktop_cell_height_1080: int = 92
    desktop_cell_center_dx_1080: int = 3
    desktop_cell_center_dy_1080: int = 16
    bottom_expand_4k: int = 20


@dataclass
class WallpaperRefineConfig:
    template_keep_threshold: float = 0.13
    max_recovery: int = 10


@dataclass
class ShapePriorSamConfig:
    matchshape_top: int = 15
    chamfer_top: int = 40
    yellow_hue_low: int = 15
    yellow_hue_high: int = 45
    yellow_saturation_low: int = 55
    yellow_saturation_high: int = 255
    yellow_value_low: int = 70
    yellow_value_high: int = 255
    folder_score_sam_weight: float = 0.35
    folder_score_shape_weight: float = 0.40
    folder_score_yellow_weight: float = 0.25
    shape_similarity_scale: float = 0.35
    folder_score_threshold: float = 0.20
    pre_fusion_expand_ratio: float = 0.12
    yellow_hue_low_relaxed: int = 12
    yellow_hue_high_relaxed: int = 50
    yellow_saturation_low_relaxed: int = 35
    yellow_value_low_relaxed: int = 55
    yellow_min_area_1080: int = 60
    yellow_max_area_1080: int = 9000
    yellow_min_width_1080: int = 10
    yellow_min_height_1080: int = 6
    yellow_max_width_1080: int = 140
    yellow_max_height_1080: int = 110
    yellow_min_aspect_ratio: float = 0.08
    yellow_max_aspect_ratio: float = 10.0
    yellow_min_pixel_ratio: float = 0.08
    yellow_max_pixel_ratio: float = 0.98


@dataclass
class OutputConfig:
    root_dir: str = "outputs"
    save_individual_masks: bool = True
    save_all_multimasks: bool = False


@dataclass
class AppConfig:
    image: ImageConfig = field(default_factory=ImageConfig)
    dexined: DexiNedConfig = field(default_factory=DexiNedConfig)
    opencv_candidates: OpenCVCandidateConfig = field(default_factory=OpenCVCandidateConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    sam: SamConfig = field(default_factory=SamConfig)
    cv_postprocess: CVPostprocessConfig = field(default_factory=CVPostprocessConfig)
    wallpaper_refine: WallpaperRefineConfig = field(default_factory=WallpaperRefineConfig)
    shape_prior_sam: ShapePriorSamConfig = field(default_factory=ShapePriorSamConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "AppConfig":
        cfg = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for section_name, values in data.items():
            if not hasattr(cfg, section_name) or not isinstance(values, dict):
                continue
            section = getattr(cfg, section_name)
            for key, value in values.items():
                if not hasattr(section, key):
                    continue
                current = getattr(section, key)
                if isinstance(current, tuple) and isinstance(value, list):
                    if current and isinstance(current[0], tuple):
                        value = tuple(tuple(x) for x in value)
                    else:
                        value = tuple(value)
                setattr(section, key, value)
        return cfg
