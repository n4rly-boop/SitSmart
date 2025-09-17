from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.services.pose_service import PoseService
from app.utils.image_io import load_pil_from_bytes


@dataclass
class PostureFeatures:
    shoulder_line_angle_deg: float
    head_tilt_deg: Optional[float]
    head_to_shoulder_distance_px: float
    head_to_shoulder_distance_ratio: float
    shoulder_width_px: float


class PostureFeatureExtractor:
    """Compute geometric features from MediaPipe Pose landmarks.

    Notes:
    - Angles are in degrees.
    - Image coordinate system has origin at top-left with +Y downward. We convert
      to a standard math coordinate (+Y up) when computing angles to keep
      intuitive signs: positive shoulder angle means left shoulder higher.
    - Distances are provided in pixels and also normalized by shoulder width.
    """

    # MediaPipe Pose indices
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_EAR = 7
    RIGHT_EAR = 8

    def extract_features(self, image_bytes: bytes) -> Dict[str, object]:
        # Reuse existing pose pipeline to get landmarks (normalized [0,1])
        analysis = PoseService.get_instance().analyze_image(image_bytes)
        details = analysis.get("details") or {}
        landmarks_norm: Optional[List[List[float]]] = details.get("landmarks")  # type: ignore[assignment]
        if landmarks_norm is None or (hasattr(landmarks_norm, "__len__") and len(landmarks_norm) == 0):
            return {
                "features": None,
                "landmarks": None,
                "reason": details.get("reason", "no landmarks"),
            }

        # Load image to recover width/height for pixel scale
        pil = load_pil_from_bytes(image_bytes)
        width, height = pil.size

        kxy_norm = np.array(landmarks_norm, dtype=np.float32)
        kxy_px = np.empty_like(kxy_norm)
        kxy_px[:, 0] = kxy_norm[:, 0] * float(width)
        kxy_px[:, 1] = kxy_norm[:, 1] * float(height)

        features = self._compute_features(kxy_px)

        return {
            "features": {
                "shoulder_line_angle_deg": float(features.shoulder_line_angle_deg),
                "head_tilt_deg": None if features.head_tilt_deg is None else float(features.head_tilt_deg),
                "head_to_shoulder_distance_px": float(features.head_to_shoulder_distance_px),
                "head_to_shoulder_distance_ratio": float(features.head_to_shoulder_distance_ratio),
                "shoulder_width_px": float(features.shoulder_width_px),
            },
            "landmarks": kxy_norm.tolist(),
        }

    def _compute_features(self, kxy_px: np.ndarray) -> PostureFeatures:
        def get_point(idx: int) -> Optional[np.ndarray]:
            if 0 <= idx < kxy_px.shape[0]:
                p = kxy_px[idx]
                if not np.any(np.isnan(p)):
                    return p
            return None

        ls = get_point(self.LEFT_SHOULDER)
        rs = get_point(self.RIGHT_SHOULDER)
        if ls is None or rs is None:
            return PostureFeatures(
                shoulder_line_angle_deg=0.0,
                head_tilt_deg=None,
                head_to_shoulder_distance_px=0.0,
                head_to_shoulder_distance_ratio=0.0,
                shoulder_width_px=0.0,
            )

        # Shoulder line angle (relative to horizontal, + when left shoulder is higher)
        dx = float(ls[0] - rs[0])
        dy_img = float(ls[1] - rs[1])  # +down in image space
        dy = -dy_img  # convert to +up for angle intuition
        shoulder_line_angle_deg = math.degrees(math.atan2(dy, dx))

        # Head center from ears -> eyes -> nose fallback
        le = get_point(self.LEFT_EAR)
        re = get_point(self.RIGHT_EAR)
        if le is None or re is None:
            le = get_point(self.LEFT_EYE)
            re = get_point(self.RIGHT_EYE)
        head_center: Optional[np.ndarray] = None
        head_tilt_deg: Optional[float] = None
        if le is not None and re is not None:
            head_center = (le + re) / 2.0
            # Head tilt (roll): ear-to-ear line relative to horizontal
            dx_he = float(le[0] - re[0])
            dy_he = -float(le[1] - re[1])
            head_tilt_deg = math.degrees(math.atan2(dy_he, dx_he))
        else:
            nose = get_point(self.NOSE)
            if nose is not None:
                head_center = nose

        shoulder_width_px = float(np.linalg.norm(rs - ls))
        if shoulder_width_px <= 1e-6 or head_center is None:
            return PostureFeatures(
                shoulder_line_angle_deg=shoulder_line_angle_deg,
                head_tilt_deg=head_tilt_deg,
                head_to_shoulder_distance_px=0.0,
                head_to_shoulder_distance_ratio=0.0,
                shoulder_width_px=shoulder_width_px,
            )

        # Perpendicular distance from head center to the shoulder line
        # Line through points A=rs and B=ls; distance from point P=head_center
        ax, ay = float(rs[0]), float(rs[1])
        bx, by = float(ls[0]), float(ls[1])
        px, py = float(head_center[0]), float(head_center[1])
        numerator = abs((by - ay) * px - (bx - ax) * py + bx * ay - by * ax)
        denom = math.hypot(by - ay, bx - ax)
        head_to_shoulder_distance_px = numerator / max(denom, 1e-6)
        head_to_shoulder_distance_ratio = head_to_shoulder_distance_px / max(shoulder_width_px, 1e-6)

        return PostureFeatures(
            shoulder_line_angle_deg=shoulder_line_angle_deg,
            head_tilt_deg=head_tilt_deg,
            head_to_shoulder_distance_px=head_to_shoulder_distance_px,
            head_to_shoulder_distance_ratio=head_to_shoulder_distance_ratio,
            shoulder_width_px=shoulder_width_px,
        )


