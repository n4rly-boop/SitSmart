import math
import threading
from typing import Dict, Optional, Tuple


class FeatureRanges:
    """Tracks per-feature ranges over time.

    - For angle features (keys ending with "_deg"), consumers may treat the
      range differently, but we store raw min/max here.
    - For other features, stores minimal [min, max] that covers all observations.
    """

    def __init__(self) -> None:
        self._min_max_by_key: Dict[str, Tuple[float, float]] = {}
        self._lock = threading.RLock()

    def update(self, features: Optional[Dict[str, float]]) -> None:
        if not features:
            return
        with self._lock:
            for key, value in features.items():
                try:
                    v = float(value)
                except Exception:
                    continue
                if not math.isfinite(v):
                    continue

                prev = self._min_max_by_key.get(key)
                if prev is None:
                    self._min_max_by_key[key] = (v, v)
                else:
                    mn, mx = prev
                    if v < mn:
                        mn = v
                    if v > mx:
                        mx = v
                    self._min_max_by_key[key] = (mn, mx)

    def get_range_width(self, key: str) -> Optional[float]:
        with self._lock:
            rng = self._min_max_by_key.get(key)
            if rng is None:
                return None
            return float(rng[1] - rng[0])

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            out: Dict[str, Dict[str, float]] = {}
            for k, (mn, mx) in self._min_max_by_key.items():
                out[k] = {"max": float(mx), "min": float(mn)}
            return out


