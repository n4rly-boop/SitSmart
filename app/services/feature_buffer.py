from __future__ import annotations

import os
import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple


class FeatureBuffer:
    """Sliding time window buffer for posture features.

    Keeps (timestamp_ms, features) pairs for a configurable number of seconds
    and computes mean features over the current window.
    """

    def __init__(self, window_seconds: Optional[int] = None):
        self.window_seconds = int(os.getenv("FEATURE_BUFFER_SECONDS", "5")) if window_seconds is None else int(window_seconds)
        self._buf: Deque[Tuple[int, Dict[str, float]]] = deque()

    def add(self, features: Optional[Dict[str, float]]) -> None:
        if not features:
            return
        try:
            now_ms = int(time.time() * 1000)
            self._buf.append((now_ms, features))
            self._trim(now_ms)
        except Exception:
            pass

    def _trim(self, now_ms: int) -> None:
        horizon_ms = now_ms - max(0, int(self.window_seconds) * 1000)
        while self._buf and self._buf[0][0] < horizon_ms:
            try:
                self._buf.popleft()
            except Exception:
                break

    def mean(self) -> Optional[Dict[str, float]]:
        if not self._buf:
            return None
        # Sum per key for numeric fields present in FeatureVector
        keys = (
            "shoulder_line_angle_deg",
            "head_tilt_deg",
            "head_to_shoulder_distance_px",
            "head_to_shoulder_distance_ratio",
            "shoulder_width_px",
        )
        sums: Dict[str, float] = {k: 0.0 for k in keys}
        counts: Dict[str, int] = {k: 0 for k in keys}
        n = 0
        for _, f in self._buf:
            n += 1
            for k in keys:
                v = f.get(k)
                if v is None:
                    continue
                try:
                    sums[k] += float(v)
                    counts[k] += 1
                except Exception:
                    pass
        if n == 0:
            return None
        out: Dict[str, float] = {}
        for k in keys:
            c = counts.get(k, 0)
            if c > 0:
                out[k] = sums[k] / float(c)
            else:
                out[k] = 0.0
        return out

    def last(self) -> Optional[Dict[str, float]]:
        """Return the most recent raw feature sample if available."""
        if not self._buf:
            return None
        try:
            return self._buf[-1][1]
        except Exception:
            return None
