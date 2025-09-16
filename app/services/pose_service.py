from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
from PIL import Image

from app.utils.image_io import load_pil_from_bytes


class PoseServiceUnavailable(RuntimeError):
    pass


@dataclass
class PoseResult:
    label: str
    score: float
    details: Dict[str, object]

class HalpeService:
    """PyTorch/MMPose inference for RTMPose (Halpe-26) models using .pth.

    Loads an RTMPose Halpe-26 checkpoint via MMPose and runs top-down
    keypoint detection. Returns normalized [x, y] in [0,1] for the first
    detected person and computes a simple posture label/score.
    """

    _instance: Optional["HalpeService"] = None

    def __init__(self, *_: object, **__: object):
        import os
        import warnings
        try:
            import torch  # type: ignore
            import numpy as _np  # type: ignore
            # Allow legacy NumPy reconstruct used in older checkpoints when weights_only=True
            try:
                torch.serialization.add_safe_globals([_np.core.multiarray._reconstruct])  # type: ignore[attr-defined]
            except Exception:
                pass
        except Exception:
            # Torch import errors will surface later via MMPose import
            pass
        try:
            from mmpose.apis import MMPoseInferencer  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise PoseServiceUnavailable(f"MMPose import failed: {exc}") from exc

        # Defaults align with repo's provided files
        self.cfg_path = os.environ.get(
            "HALPE_CFG",
            "models/rtmpose-s_8xb1024-700e_body8-halpe26-256x192.py",
        )
        self.pth_path = os.environ.get(
            "HALPE_PTH",
            "models/rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.pth",
        )
        self.device = os.environ.get("HALPE_DEVICE", "cpu")
        # Detector cadence: reuse last bbox to avoid running person detector every frame
        try:
            self.det_every = max(1, int(os.environ.get("HALPE_DET_EVERY", "5")))
        except Exception:
            self.det_every = 5
        self._frame_index = 0
        self._last_bbox: Optional[Tuple[float, float, float, float]] = None

        if not os.path.isfile(self.cfg_path):
            raise PoseServiceUnavailable(
                f"HALPE_CFG not found: {self.cfg_path}. Place your config file accordingly."
            )
        if not os.path.isfile(self.pth_path):
            raise PoseServiceUnavailable(
                f"HALPE_PTH not found: {self.pth_path}. Place your .pth checkpoint accordingly."
            )

        # Prepare a checkpoint path that MMPose can safely load
        safe_weights_path = self._prepare_weights(self.pth_path)

        try:
            # Let MMPose load an internal person detector by default when needed.
            self._infer = MMPoseInferencer(
                pose2d=self.cfg_path,
                pose2d_weights=safe_weights_path,
                device=self.device,
            )
        except Exception as exc:  # noqa: BLE001
            raise PoseServiceUnavailable(f"Failed to initialize MMPose: {exc}") from exc

    @classmethod
    def get_instance(cls) -> "HalpeService":
        if cls._instance is None:
            cls._instance = HalpeService()
        return cls._instance

    def analyze_image(self, image_bytes: bytes) -> Dict[str, object]:
        img = load_pil_from_bytes(image_bytes)
        rgb = np.array(img)
        H, W = rgb.shape[0], rgb.shape[1]
        self._frame_index += 1
        # Use cached bbox most frames to skip running the detector
        bboxes = None
        if self._last_bbox is not None and (self._frame_index % self.det_every != 0):
            bboxes = [self._last_bbox]
        try:
            if bboxes is not None:
                gen = self._infer(rgb, bboxes=bboxes, return_vis=False)
            else:
                gen = self._infer(rgb, return_vis=False)
            result = next(gen)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"torch_infer_failed: {exc}"}

        preds = result.get("predictions", None)
        # predictions is expected as List[List[Dict]]
        if not isinstance(preds, (list, tuple)) or len(preds) == 0:
            return {"error": "no_person"}
        instances = preds[0]
        if not isinstance(instances, (list, tuple)) or len(instances) == 0:
            return {"error": "no_person"}
        # Select the most relevant person: prefer largest, centered bbox
        chosen = self._select_person(instances, W, H)
        kpts_raw = chosen.get("keypoints") if isinstance(chosen, dict) else None
        kxy_px = np.array(kpts_raw, dtype=np.float32) if kpts_raw is not None else np.zeros((0, 2), dtype=np.float32)
        # Some models output (K,3) with scores; take x,y only
        if kxy_px.ndim == 2 and kxy_px.shape[1] >= 2:
            kxy_px = kxy_px[:, :2]
        if kxy_px.ndim != 2 or kxy_px.shape[1] < 2:
            return {"error": "invalid_keypoints"}

        # Update last bbox if available on this result
        try:
            bb = chosen.get("bbox")
            if isinstance(bb, (list, tuple)) and len(bb) >= 4:
                x1, y1, x2, y2 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
                self._last_bbox = (x1, y1, x2, y2)
        except Exception:
            pass

        # Normalize to [0,1]
        kxy_norm = self._normalize_keypoints(kxy_px, W, H)

        # Simple posture heuristic based on shoulders and head
        label, score, metrics = self._simple_posture_heuristic_halpe26(kxy_px)
        metrics["landmarks"] = kxy_norm.tolist()

        return {"label": label, "score": float(score), "details": metrics, "kpts": kxy_norm.tolist()}

    @staticmethod
    def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = max(inter + area_a + area_b - inter, 1e-6)
        return inter / union

    def _select_person(self, instances: List[Dict[str, object]], width: int, height: int) -> Dict[str, object]:
        # Score by bbox area and proximity to image center; prefer continuity via IoU with last bbox
        cx, cy = width * 0.5, height * 0.5
        best = None
        best_score = -1e9
        for inst in instances:
            if not isinstance(inst, dict):
                continue
            bb = inst.get("bbox")
            if not isinstance(bb, (list, tuple)) or len(bb) < 4:
                continue
            x1, y1, x2, y2 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            area = w * h
            pcx, pcy = x1 + w * 0.5, y1 + h * 0.5
            # Normalize distance to [0, ~1.4]
            dx = (pcx - cx) / max(width, 1)
            dy = (pcy - cy) / max(height, 1)
            dist_penalty = (dx * dx + dy * dy) ** 0.5
            cont_bonus = 0.0
            if self._last_bbox is not None:
                cont_bonus = self._iou((x1, y1, x2, y2), self._last_bbox) * 0.5
            score = area - dist_penalty * (width * height * 0.1) + cont_bonus * (width * height * 0.05)
            if score > best_score:
                best_score = score
                best = inst
        return best or instances[0]

    @staticmethod
    def _normalize_keypoints(kxy_px: np.ndarray, width: int, height: int) -> np.ndarray:
        out = np.zeros_like(kxy_px, dtype=np.float32)
        if kxy_px.size == 0:
            return out
        out[:, 0] = np.clip(kxy_px[:, 0] / max(width - 1, 1), 0.0, 1.0)
        out[:, 1] = np.clip(kxy_px[:, 1] / max(height - 1, 1), 0.0, 1.0)
        return out

    @staticmethod
    def _simple_posture_heuristic_halpe26(kxy_px: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        # Halpe-26 approximate indices
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 2
        NOSE = 0
        NECK = 1
        CHEST = 18

        def get_point(idx: int) -> Optional[np.ndarray]:
            if idx < 0 or idx >= kxy_px.shape[0]:
                return None
            p = kxy_px[idx]
            # Expect shape (2,) for x,y
            if p is None:
                return None
            p = np.asarray(p, dtype=np.float32)
            if p.shape[0] < 2:
                return None
            if not np.all(np.isfinite(p[:2])):
                return None
            return p[:2]

        ls = get_point(LEFT_SHOULDER)
        rs = get_point(RIGHT_SHOULDER)
        head = get_point(NOSE)
        if head is None:
            head = get_point(NECK)
        if head is None:
            head = get_point(CHEST)
        if ls is None or rs is None or head is None:
            return ("unknown", 0.0, {"reason": "insufficient keypoints"})

        shoulder_mid = (ls + rs) / 2.0
        shoulder_width = float(np.linalg.norm(rs - ls))
        vertical_drop = float(head[1] - shoulder_mid[1])
        if shoulder_width <= 1e-6:
            return ("unknown", 0.0, {"reason": "degenerate shoulder width"})

        drop_ratio = vertical_drop / max(shoulder_width, 1e-6)
        bad_threshold = 0.32
        good_threshold = 0.20

        if drop_ratio >= bad_threshold:
            label = "bad"
            score = HalpeService._sigmoid_scale(drop_ratio, center=bad_threshold, sharpness=8.0)
        elif drop_ratio <= good_threshold:
            label = "good"
            score = 1.0 - HalpeService._sigmoid_scale(drop_ratio, center=good_threshold, sharpness=8.0)
            score = max(0.5, score)
        else:
            label = "unknown"
            score = 0.5

        metrics = {
            "drop_ratio": drop_ratio,
            "vertical_drop_px": vertical_drop,
            "shoulder_width_px": shoulder_width,
            "good_threshold": good_threshold,
            "bad_threshold": bad_threshold,
        }
        return label, float(score), metrics

    @staticmethod
    def _sigmoid_scale(x: float, center: float, sharpness: float = 5.0) -> float:
        z = (x - center) * sharpness
        return 1.0 / (1.0 + float(np.exp(-z)))

    # --- Checkpoint preparation helpers ---
    def _prepare_weights(self, src_path: str) -> str:
        import os
        import tempfile
        import torch  # type: ignore
        import numpy as _np  # type: ignore

        obj = None
        # Try safe weights-only load first
        try:
            with self._torch_safe_globals([
                _np.core.multiarray._reconstruct,  # type: ignore[attr-defined]
                getattr(_np.core.multiarray, 'scalar', None),  # type: ignore[attr-defined]
                _np.ndarray, _np.dtype, getattr(_np, 'ufunc', type(lambda: None)),
            ]):
                obj = torch.load(src_path, map_location='cpu', weights_only=True)
        except Exception:
            obj = None

        if obj is None:
            # Optional unsafe fallback if explicitly allowed
            if os.environ.get('HALPE_UNSAFE_LOAD', '0') == '1':
                with self._torch_safe_globals([
                    _np.core.multiarray._reconstruct,  # type: ignore[attr-defined]
                    getattr(_np.core.multiarray, 'scalar', None),  # type: ignore[attr-defined]
                    _np.ndarray, _np.dtype, getattr(_np, 'ufunc', type(lambda: None)),
                ]):
                    obj = torch.load(src_path, map_location='cpu', weights_only=False)
            else:
                # If we cannot load safely, propagate a clear message
                raise PoseServiceUnavailable(
                    "Weights-only load failed. Set HALPE_UNSAFE_LOAD=1 to permit legacy checkpoint loading, "
                    "or convert the checkpoint to a pure state_dict."
                )

        # Extract state dict
        if isinstance(obj, dict) and 'state_dict' in obj:
            state_dict = obj['state_dict']
        elif isinstance(obj, dict):
            # Assume this is already param_name -> tensor mapping
            state_dict = obj
        else:
            raise PoseServiceUnavailable("Unsupported checkpoint format: expected dict or state_dict")

        # Save sanitized checkpoint
        fd, tmp_path = tempfile.mkstemp(prefix='halpe26_safe_', suffix='.pth')
        os.close(fd)
        torch.save({'state_dict': state_dict}, tmp_path)
        return tmp_path

    from contextlib import contextmanager
    @contextmanager
    def _torch_safe_globals(self, globals_list: List[object]):  # type: ignore[name-defined]
        import torch  # type: ignore
        try:
            ctx = getattr(torch.serialization, 'safe_globals', None)
        except Exception:
            ctx = None
        if ctx is not None:
            with ctx([g for g in globals_list if g is not None]):
                yield
        else:
            # Fallback: best-effort global allow-list
            try:
                torch.serialization.add_safe_globals([g for g in globals_list if g is not None])  # type: ignore[arg-type]
            except Exception:
                pass
            yield
