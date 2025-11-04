import pickle
import numpy as np
from typing import Optional
from app.api.schemas import ModelAnalysisRequest, ModelAnalysisResponse


class MLService:
    _instance: Optional['MLService'] = None
    FEATURE_ORDER = [
        'shoulder_line_angle_deg',
        'head_tilt_deg',
        'head_to_shoulder_distance_px',
        'head_to_shoulder_distance_ratio',
        'shoulder_width_px'
    ]

    def __init__(
            self,
            model_path: str = "app/models/model.pkl",
            scaler_path: str = "app/models/scaler.pkl"
    ):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        print(f"[MLService] Model loaded from {model_path}")
        print(f"[MLService] Scaler loaded from {scaler_path}")

    @classmethod
    def get_instance(cls) -> 'MLService':
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def analyze(self, request: ModelAnalysisRequest) -> ModelAnalysisResponse:
        features_dict = request.features.model_dump()
        ANGLE_KEYS = {'shoulder_line_angle_deg', 'head_tilt_deg'}

        feature_vector = np.array([
            (abs(features_dict.get(key, 0.0)) if key in ANGLE_KEYS else features_dict.get(key, 0.0))
            for key in self.FEATURE_ORDER
        ]).reshape(1, -1)

        X_scaled = self.scaler.transform(feature_vector)
        probabilities = self.model.predict_proba(X_scaled)

        bad_prob = float(probabilities[0][0])
        good_prob = float(probabilities[0][1])
        assert abs(good_prob + bad_prob - 1.0) < 1e-9

        return ModelAnalysisResponse(
            bad_posture_prob=bad_prob
        )