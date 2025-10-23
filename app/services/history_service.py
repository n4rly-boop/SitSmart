from __future__ import annotations

import os
import json
import urllib.request
import threading
import time
import math
from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional, Tuple

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

        # Use in-process service for features to avoid self-HTTP deadlocks
        self._api_base: str = os.getenv("RL_API_BASE_URL", "http://127.0.0.1:8000/api")

        # Config
        self._delta_range_seconds: float = float(os.getenv("RL_DELTA_RANGE_SECONDS", "5"))
        self._feature_time_tolerance_ms: int = int(os.getenv("RL_FEATURE_TOLERANCE_MS", "2000"))

        # Concurrency control for histories
        self._lock = threading.RLock()

        # Feature ranges tracker (angles: max |value|, others: min/max)
        self._feature_ranges = FeatureRanges()

    @classmethod
    def get_instance(cls) -> "HistoryService":
        if cls._instance is None:
            cls._instance = HistoryService()
        return cls._instance

    # --------------- Feature access via service ---------------
    def _get_f1_features(self) -> Optional[Dict[str, float]]:
        try:
            from app.services.feature_aggregate_service import FeatureAggregateService
            return FeatureAggregateService.get_instance().mean_features()
        except Exception:
            return None

    def _get_f2_features(self) -> Optional[Dict[str, float]]:
        try:
            from app.services.feature_aggregate_service import FeatureAggregateService
            return FeatureAggregateService.get_instance().last_features()
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
        # Sleep until t + delta_range_seconds
        try:
            sleep_seconds = max(0.0, float(self._delta_range_seconds))
            time.sleep(sleep_seconds)
        except Exception:
            pass

        with self._lock:
            if record_index < 0 or record_index >= len(self._notification_history):
                return
            rec = self._notification_history[record_index]
            f1 = rec.f1_features

        # If f1 was not provided, obtain it now (background) from aggregate mean
        if f1 is None:
            f1 = self._get_f1_features()

        # Fetch f2 (latest sample) from in-process service
        f2 = self._get_f2_features()

        # Update feature ranges based on observed feature snapshots
        try:
            self._feature_ranges.update(f1)
            self._feature_ranges.update(f2)
        except Exception:
            pass

        # Compute delta outside lock to avoid long holds
        delta_value = self._compute_delta_value(f1, f2)

        with self._lock:
            # Update the stored record
            self._notification_history[record_index].delta = delta_value
            self._notification_history[record_index].f1_features = f1

    # --------------- Feature ranges API ---------------
    def update_feature_ranges(self, features: Optional[Dict[str, float]]) -> None:
        """Public helper to update ranges from arbitrary feature snapshots."""
        try:
            self._feature_ranges.update(features)
        except Exception:
            pass

    def get_feature_ranges_snapshot(self) -> Dict[str, Dict[str, float]]:
        """Return a copy of current tracked ranges for inspection/export."""
        return self._feature_ranges.snapshot()

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
                rng = self._feature_ranges.get_max_abs(key)
                if key.endswith("_deg"):
                    rng /= 2.0
                rel = abs(v1 - v0) / rng
                deltas.append(rel)
            except Exception:
                continue
        if not deltas:
            return None
        # Mean of absolute normalized differences
        return float(sum(deltas) / float(len(deltas)))

    # --------------- Accessors ---------------
    def get_notification_history(self) -> List[NotificationRecord]:
        with self._lock:
            return list(self._notification_history)

    def get_history_vectors(self) -> List[Tuple[float, Optional[float], float, int]]:
        with self._lock:
            return [
                (r.bad_posture_prob, r.delta, r.threshold, r.timestamp_ms)
                for r in self._notification_history
            ]

    def get_meaningful_delta_threshold(self) -> Optional[float]:
        """User-specific dynamic meaningful delta threshold using full history.

        Uses configurable quantile across observed deltas to adapt to the user.
        """
        q = float(os.getenv("RL_MEANINGFUL_DELTA_QUANTILE", "0.5"))
        with self._lock:
            deltas = [r.delta for r in self._notification_history if r.delta is not None]
        if not deltas:
            return None
        try:
            # Compute quantile via median if q==0.5, otherwise interpolate
            if q == 0.5:
                return float(median(deltas))
            sorted_vals = sorted(deltas)
            if len(sorted_vals) == 1:
                return float(sorted_vals[0])
            pos = q * (len(sorted_vals) - 1)
            lo = int(pos)
            hi = min(lo + 1, len(sorted_vals) - 1)
            frac = pos - lo
            return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)
        except Exception:
            return None



class FeatureRanges:
    """Tracks per-feature ranges over time.

    - For angle features (keys ending with "_deg"), stores the maximum absolute value observed.
    - For other features, stores minimal [min, max] that covers all observations so far.
    """

    def __init__(self) -> None:
        self._max_abs_rng_by_key: Dict[str, Tuple[float, float]] = {}
        self._lock = threading.RLock()

    def update(self, features: Optional[Dict[str, list[float]]]) -> None:
        if not features:
            return
        with self._lock:
            for key, value in features.items():
                try:
                    v = value
                except Exception:
                    continue
                if not math.isfinite(v):
                    continue
            
                prev_rng = self._max_abs_rng_by_key.get(key)
                
                if prev_rng is None:
                    self._max_abs_rng_by_key[key] = (v, v)
                else:
                    mn, mx = prev_rng
                    if v < mn:
                        mn = v
                    if v > mx:
                        mx = v
                    self._max_abs_rng_by_key[key] = (mn, mx)

    def get_max_abs(self, key: str) -> Optional[float]:
        with self._lock:
            rng = self._max_abs_rng_by_key.get(key)
            if rng is None:
                return 1
            return rng[1] - rng[0]

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            out: Dict[str, Dict[str, float]] = {}
            for k, v in self._max_abs_rng_by_key.items():
                out[k] = {"max": float(v[1]), "min": float(v[0])}
            return out
