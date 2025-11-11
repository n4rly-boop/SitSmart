import threading
import time
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.config import get_history_config

@dataclass
class NotificationRecord:
    bad_posture_prob: float
    threshold: float
    timestamp_ms: int
    delta: Optional[float] = None
    f1_features: Optional[Dict[str, float]] = None  # averaged snapshot at notification time


class HistoryService:
    """In-memory global history of features and notification outcomes.

    Responsibilities:
    - Store ALL features with timestamps (no sliding window)
    - On each notification, compute delta after a configurable delay and store
      a record vector: [bad_posture_prob, delta, threshold, timestamp]
    """

    _instance: Optional["HistoryService"] = None

    def __init__(self) -> None:
        # Notification history – never trimmed per requirements
        self._notification_history: List[NotificationRecord] = []

        # Load config
        config = get_history_config()
        self._delta_range_seconds: float = config.delta_range_seconds

        # Concurrency control for histories
        self._lock = threading.RLock()


    @classmethod
    def get_instance(cls) -> "HistoryService":
        if cls._instance is None:
            cls._instance = HistoryService()
        return cls._instance

    # --------------- Feature access via service ---------------
    def _get_features(self, use_mean: bool = True) -> Optional[Dict[str, float]]:
        """Get features from FeatureAggregateService.
        
        Args:
            use_mean: If True, return mean features (f1), else return last features (f2).
        """
        try:
            from app.services.feature_aggregate_service import FeatureAggregateService
            service = FeatureAggregateService.get_instance()
            return service.mean_features() if use_mean else service.last_features()
        except Exception:
            return None

    # --------------- Notification handling ---------------
    def on_notification(self, bad_posture_prob: float, threshold: float, timestamp_ms: int, f1_features: Optional[Dict[str, float]] = None) -> None:
        # Create minimal record and compute f1/f2 asynchronously without self-HTTP

        record = NotificationRecord(
            bad_posture_prob=float(bad_posture_prob),
            threshold=float(threshold),
            timestamp_ms=int(timestamp_ms),
            delta=None,
            f1_features=f1_features,
        )
        with self._lock:
            self._notification_history.append(record)
            record_index = len(self._notification_history) - 1

        # Compute delta asynchronously after the configured window
        threading.Thread(
            target=self._compute_and_store_delta,
            args=(record_index,),
            daemon=True,
        ).start()

    def _compute_and_store_delta(self, record_index: int) -> None:
        """Compute and store delta for a notification record after delay."""
        # Sleep until t + delta_range_seconds
        try:
            sleep_seconds = max(0.0, float(self._delta_range_seconds))
            time.sleep(sleep_seconds)
        except Exception:
            pass

        # Get record and f1 features (with lock)
        with self._lock:
            if record_index < 0 or record_index >= len(self._notification_history):
                return
            rec = self._notification_history[record_index]
            f1 = rec.f1_features

        # Fetch features outside lock to avoid blocking
        if f1 is None:
            f1 = self._get_features(use_mean=True)
        f2 = self._get_features(use_mean=False)

        # Compute delta outside lock
        delta_value = self._compute_delta_value(f1, f2)

        # Update record (with lock)
        with self._lock:
            # Re-check bounds in case history was cleared
            if record_index < 0 or record_index >= len(self._notification_history):
                return
            self._notification_history[record_index].delta = delta_value
            if f1 is not None:
                self._notification_history[record_index].f1_features = f1

    # --------------- History maintenance ---------------
    def clear_history(self) -> None:
        with self._lock:
            self._notification_history.clear()

    def _compute_delta_value(
        self,
        features_t0: Optional[Dict[str, float]],
        features_t1: Optional[Dict[str, float]],
    ) -> Optional[float]:
        """Delta definition per spec:
        normalized absolute difference between features at t0 (notification) and
        features at t1 = t0 + delta_range_seconds, normalized by new/old - 1.
        We aggregate across feature dimensions by mean of per-dimension deltas.
        """
        if not features_t0 or not features_t1:
            return None
        # Use the same ordering as ML feature vector
        from app.services.ml_service import MLService  # local import to avoid cycles

        deltas: List[float] = []
        for key in MLService.FEATURE_ORDER:
            try:
                v0 = float(features_t0.get(key, 0.0))
                v1 = float(features_t1.get(key, 0.0))
                if abs(v0) <= 1e-12:
                    continue
                # Use CalibrationService ranges for normalization; fallback to 1
                try:
                    from app.services.calibration_service import CalibrationService
                    rng = CalibrationService.get_instance().get_range_width(key)
                except Exception:
                    rng = None
                if not rng or rng <= 0:
                    rng = 1.0
                if key.endswith("_deg"):
                    rng /= 2.0
                rel = abs(v1 - v0) / rng
                deltas.append(rel)
            except Exception:
                continue
        if not deltas:
            return None
        # Root mean of absolute normalized differences
        return self._delta_to_scalar(deltas)

    def _delta_to_scalar(self, delta: List[float]) -> float:
        """Convert list of delta values to a single scalar using root mean formula."""
        bias = 0.1
        numerator = max(sum(delta) - bias, 0.0)
        denominator = max(float(len(delta) - bias), 1.0)
        return float(math.sqrt(numerator / denominator))

    # --------------- Accessors ---------------
    def get_notification_history(self) -> List[NotificationRecord]:
        """Get a copy of all notification history records."""
        with self._lock:
            return list(self._notification_history)