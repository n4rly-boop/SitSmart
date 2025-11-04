

import os
from typing import Dict, Optional

from app.services.feature_buffer import FeatureBuffer


class FeatureAggregateService:
    _instance: Optional["FeatureAggregateService"] = None

    def __init__(self, window_seconds: Optional[int] = None) -> None:
        self._buffer = FeatureBuffer(window_seconds=window_seconds)

    @classmethod
    def get_instance(cls) -> "FeatureAggregateService":
        if cls._instance is None:
            seconds = int(os.getenv("FEATURE_BUFFER_SECONDS", "5"))
            cls._instance = FeatureAggregateService(window_seconds=seconds)
        return cls._instance

    # ------------- API -------------
    def add_features(self, features: Dict[str, float]) -> None:
        self._buffer.add(features)

    def mean_features(self) -> Optional[Dict[str, float]]:
        return self._buffer.mean()

    def last_features(self) -> Optional[Dict[str, float]]:
        last_val = None
        try:
            last_val = self._buffer.last()
        except Exception:
            last_val = None
        if isinstance(last_val, tuple) and len(last_val) == 2 and isinstance(last_val[1], dict):
            return last_val[1]
        if isinstance(last_val, dict):
            return last_val
        return None

    def clear(self) -> None:
        # Reinitialize buffer preserving the window size
        seconds = self._buffer.window_seconds
        from app.services.feature_buffer import FeatureBuffer as _FB
        self._buffer = _FB(window_seconds=seconds)


