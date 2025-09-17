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


class PoseService:
    _instance: Optional["PoseService"] = None

    def __init__(self):
        try:
            import mediapipe as mp  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise PoseServiceUnavailable(f"MediaPipe import failed: {exc}") from exc

        self._mp = mp
        # Light-weight model for real-time
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @classmethod
    def get_instance(cls) -> "PoseService":
        if cls._instance is None:
            cls._instance = PoseService()
        return cls._instance

    def analyze_image(self, image_bytes: bytes) -> Dict[str, object]:
        pil: Image.Image = load_pil_from_bytes(image_bytes)
        # MediaPipe expects RGB np array
        rgb = np.array(pil)

        results = self._pose.process(rgb)
        if not results.pose_landmarks:
            return {"label": "unknown", "score": 0.0, "details": {"reason": "no landmarks"}}

        landmarks = results.pose_landmarks.landmark
        kxy_px = self._landmarks_to_numpy_px(landmarks, image_shape=rgb.shape)
        kxy_norm = self._landmarks_to_numpy_norm(landmarks)

        label, score, metrics = self._simple_posture_heuristic(kxy_px)
        metrics["landmarks"] = kxy_norm.tolist()  # list of [x,y] in [0,1]
        return {"label": label, "score": float(score), "details": metrics}

    @staticmethod
    def _landmarks_to_numpy_px(landmarks: List[object], image_shape: Tuple[int, int, int]) -> np.ndarray:
        h, w = image_shape[0], image_shape[1]
        arr = np.zeros((33, 2), dtype=np.float32)
        for i, lm in enumerate(landmarks[:33]):
            arr[i, 0] = lm.x * w
            arr[i, 1] = lm.y * h
        return arr

    @staticmethod
    def _landmarks_to_numpy_norm(landmarks: List[object]) -> np.ndarray:
        arr = np.zeros((33, 2), dtype=np.float32)
        for i, lm in enumerate(landmarks[:33]):
            arr[i, 0] = lm.x
            arr[i, 1] = lm.y
        return arr

    @staticmethod
    def _simple_posture_heuristic(kxy: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        # MediaPipe Pose indices for shoulders/ears/nose equivalent
        # https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        NOSE = 0
        LEFT_EAR = 7
        RIGHT_EAR = 8
        LEFT_EYE = 2
        RIGHT_EYE = 5

        def get_point(idx: int) -> Optional[np.ndarray]:
            if idx < kxy.shape[0]:
                p = kxy[idx]
                if not np.any(np.isnan(p)):
                    return p
            return None

        ls = get_point(LEFT_SHOULDER)
        rs = get_point(RIGHT_SHOULDER)
        le = get_point(LEFT_EAR)
        if le is None:
            le = get_point(LEFT_EYE)
        re = get_point(RIGHT_EAR)
        if re is None:
            re = get_point(RIGHT_EYE)
        head = None
        nose = get_point(NOSE)
        if le is not None and re is not None:
            head = (le + re) / 2.0
        elif nose is not None:
            head = nose

        if ls is None or rs is None or head is None:
            return ("unknown", 0.0, {"reason": "insufficient keypoints"})

        shoulder_mid = (ls + rs) / 2.0
        shoulder_width = float(np.linalg.norm(rs - ls))
        vertical_drop = float(head[1] - shoulder_mid[1])  # +down in image space

        if shoulder_width <= 1e-6:
            return ("unknown", 0.0, {"reason": "degenerate shoulder width"})

        drop_ratio = vertical_drop / max(shoulder_width, 1e-6)
        # Rough thresholds; tune later for MediaPipe scale
        bad_threshold = 0.32
        good_threshold = 0.20

        if drop_ratio >= bad_threshold:
            label = "bad"
            score = PoseService._sigmoid_scale(drop_ratio, center=bad_threshold, sharpness=8.0)
        elif drop_ratio <= good_threshold:
            label = "good"
            score = 1.0 - PoseService._sigmoid_scale(drop_ratio, center=good_threshold, sharpness=8.0)
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
        return label, score, metrics

    @staticmethod
    def _sigmoid_scale(x: float, center: float, sharpness: float = 5.0) -> float:
        z = (x - center) * sharpness
        return 1.0 / (1.0 + float(np.exp(-z)))
