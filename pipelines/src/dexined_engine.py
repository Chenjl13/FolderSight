from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import cv2
import numpy as np

from .config import AppConfig
from .utils import choose_device, ensure_dir, save_json


class DexiNedEngine:
    """Loads a local checkout of the official DexiNed model implementation."""

    def __init__(self, cfg: AppConfig, logger):
        self.cfg = cfg
        self.logger = logger
        self.device = choose_device(cfg.dexined.device)
        self.model = None
        self.checkpoint = None

    def _resolve_checkpoint(self, repo: Path) -> Path:
        configured = str(self.cfg.dexined.checkpoint_path).strip()
        if configured:
            p = Path(configured)
            if not p.is_absolute():
                p = Path.cwd() / p
            if p.exists():
                return p.resolve()
            raise FileNotFoundError(f"Configured DexiNed checkpoint not found: {p}")

        candidates = sorted((repo / "checkpoints").rglob("*.pth"))
        if not candidates:
            candidates = sorted((repo / "checkpoints").rglob("*.pt"))
        if not candidates:
            raise FileNotFoundError(
                f"No DexiNed .pth/.pt checkpoint found under {repo / 'checkpoints'}. "
                "Pass --dexined-checkpoint explicitly."
            )
        # Prefer names commonly used by the official repository.
        preferred = [p for p in candidates if "model" in p.name.lower()]
        return (preferred[-1] if preferred else candidates[-1]).resolve()

    def _load(self):
        if self.model is not None:
            return
        import torch

        repo = Path(self.cfg.dexined.repo_path).resolve()
        model_py = repo / "model.py"
        if not model_py.exists():
            raise FileNotFoundError(
                f"DexiNed model.py not found: {model_py}. "
                "Set dexined.repo_path in the config to a local DexiNed checkout."
            )
        ckpt = self._resolve_checkpoint(repo)

        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        spec = importlib.util.spec_from_file_location("local_dexined_model", model_py)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import DexiNed model from {model_py}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "DexiNed"):
            raise AttributeError(f"{model_py} does not define class DexiNed")

        model = module.DexiNed().to(self.device)
        payload = torch.load(str(ckpt), map_location=self.device)
        if isinstance(payload, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in payload and isinstance(payload[key], dict):
                    payload = payload[key]
                    break
        if not isinstance(payload, dict):
            raise TypeError("Unsupported DexiNed checkpoint format; expected a state_dict-like dict")

        state = {}
        for key, value in payload.items():
            state[key[7:] if key.startswith("module.") else key] = value
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            self.logger.warning("DexiNed missing keys (first 12): %s", missing[:12])
        if unexpected:
            self.logger.warning("DexiNed unexpected keys (first 12): %s", unexpected[:12])
        model.eval()
        self.model = model
        self.checkpoint = ckpt
        self.logger.info("DexiNed loaded: %s on %s", ckpt, self.device)

    def predict(self, image_bgr: np.ndarray, out_dir: str | Path) -> dict:
        self._load()
        import torch

        out_dir = ensure_dir(out_dir)
        orig_h, orig_w = image_bgr.shape[:2]
        multiple = max(1, int(self.cfg.dexined.pad_multiple))
        pad_h = (multiple - orig_h % multiple) % multiple
        pad_w = (multiple - orig_w % multiple) % multiple
        padded = cv2.copyMakeBorder(
            image_bgr,
            0,
            pad_h,
            0,
            pad_w,
            cv2.BORDER_REFLECT_101,
        ) if (pad_h or pad_w) else image_bgr

        arr = padded.astype(np.float32)
        arr -= np.asarray(self.cfg.dexined.mean_bgr, dtype=np.float32).reshape(1, 1, 3)
        arr = np.ascontiguousarray(arr.transpose(2, 0, 1))
        tensor = torch.frombuffer(arr.tobytes(), dtype=torch.float32)
        tensor = tensor.reshape(1, 3, padded.shape[0], padded.shape[1]).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            fused = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
            prob_tensor = torch.sigmoid(fused).squeeze().detach().float().cpu()
            prob = np.array(prob_tensor.tolist(), dtype=np.float32)

        if prob.ndim != 2:
            prob = np.squeeze(prob)
        if prob.shape != padded.shape[:2]:
            prob = cv2.resize(prob, (padded.shape[1], padded.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Force exact coordinate alignment with SAM input.
        prob = np.clip(prob[:orig_h, :orig_w], 0.0, 1.0).astype(np.float32)
        if prob.shape != (orig_h, orig_w):
            prob = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        binary = (prob >= float(self.cfg.dexined.edge_threshold)).astype(np.uint8)

        prob_path = out_dir / "edge_probability.png"
        binary_path = out_dir / "edge_binary.png"
        overlay_path = out_dir / "edge_overlay.png"
        cv2.imwrite(str(prob_path), np.round(prob * 255.0).astype(np.uint8))
        cv2.imwrite(str(binary_path), binary * 255)
        red = image_bgr.copy()
        red[binary > 0] = (0, 0, 255)
        overlay = cv2.addWeighted(image_bgr, 0.75, red, 0.25, 0)
        cv2.imwrite(str(overlay_path), overlay)

        report = {
            "device": self.device,
            "checkpoint": str(self.checkpoint),
            "input_size": [orig_w, orig_h],
            "padded_size": [int(padded.shape[1]), int(padded.shape[0])],
            "output_size": [int(prob.shape[1]), int(prob.shape[0])],
            "edge_threshold": float(self.cfg.dexined.edge_threshold),
            "edge_pixel_ratio": float(binary.mean()),
            "edge_probability_path": str(prob_path),
            "edge_binary_path": str(binary_path),
            "edge_overlay_path": str(overlay_path),
        }
        save_json(out_dir / "dexined_report.json", report)
        return {**report, "edge_probability": prob, "edge_binary": binary}
