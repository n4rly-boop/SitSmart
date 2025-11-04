from typing import Dict, Optional

from app.services.feature_ranges import FeatureRanges


class CalibrationService:
    """Singleton that manages calibration mode and persistent feature ranges.

    - While calibrating, incoming feature snapshots update the ranges.
    - On calibration start: ranges are reset and calibration is enabled.
    - On calibration stop: history is cleared, ranges persist for the session.
    """

    _instance: Optional["CalibrationService"] = None

    def __init__(self) -> None:
        self._ranges = FeatureRanges()
        self._calibrating: bool = False

    @classmethod
    def get_instance(cls) -> "CalibrationService":
        if cls._instance is None:
            cls._instance = CalibrationService()
        return cls._instance

    def start(self) -> None:
        # Reset ranges each time calibration starts
        self._ranges = FeatureRanges()
        self._calibrating = True

    def stop(self) -> None:
        # Persist ranges, stop calibration and clear history
        self._calibrating = False
        try:
            from app.services.history_service import HistoryService
            HistoryService.get_instance().clear_history()
        except Exception:
            pass

    def is_calibrating(self) -> bool:
        return bool(self._calibrating)

    def update_from_features(self, features: Optional[Dict[str, float]]) -> None:
        if not self._calibrating:
            return
        try:
            self._ranges.update(features)
        except Exception:
            pass

    def get_range_width(self, key: str) -> Optional[float]:
        try:
            return self._ranges.get_range_width(key)
        except Exception:
            return None

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        try:
            return self._ranges.snapshot()
        except Exception:
            return {}


