from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Optional

from app.services.history_service import HistoryService


@dataclass
class RLOptions:
    # Exploration probability
    epsilon: float = float(os.getenv("RL_EPSILON", "0.2"))
    # Step to change threshold during exploration/exploitation
    threshold_step: float = float(os.getenv("RL_THRESHOLD_STEP", "0.05"))
    # Allowed threshold range
    min_threshold: float = float(os.getenv("RL_MIN_THRESHOLD", "0.3"))
    max_threshold: float = float(os.getenv("RL_MAX_THRESHOLD", "0.95"))
    # Initial default when no history
    default_threshold: float = float(os.getenv("ML_BAD_PROB_THRESHOLD", "0.6"))


class EpsilonGreedyAgent:
    _instance: Optional["EpsilonGreedyAgent"] = None

    def __init__(self, options: Optional[RLOptions] = None) -> None:
        self.options = options or RLOptions()
        self._current_threshold: float = float(self.options.default_threshold)

    @classmethod
    def get_instance(cls) -> "EpsilonGreedyAgent":
        if cls._instance is None:
            cls._instance = EpsilonGreedyAgent()
        return cls._instance

    def suggest_threshold(self, current_ml_bad_prob: float) -> float:
        """Return threshold by maximizing recency- and threshold-smoothed expected delta.

        We estimate E[delta | threshold=t] via kernel smoothing across historical
        (delta, threshold) observations with exponential recency decay, then pick
        the minimal t whose expected delta is within a slack of the best.
        """
        hist = HistoryService.get_instance()
        vectors = hist.get_history_vectors()  # (bad_prob, delta, threshold, ts)

        # If no delta has ever been computed yet, keep current setting
        has_any_delta = any(delta is not None for _, delta, _, _ in vectors)
        if not has_any_delta:
            return float(self._current_threshold)

        # Config
        min_thr = float(self.options.min_threshold)
        max_thr = float(self.options.max_threshold)
        grid_step = float(os.getenv("RL_GRID_STEP", "0.05"))
        bw = float(os.getenv("RL_THRESHOLD_KERNEL_BW", str(max(grid_step, 1e-6))))
        tau = float(os.getenv("RL_RECENCY_TAU_SEC", "600"))  # seconds
        prior_weight = float(os.getenv("RL_PRIOR_WEIGHT", "1.0"))
        slack_frac = float(os.getenv("RL_DELTA_SLACK_FRAC", "0.1"))
        slack_abs = float(os.getenv("RL_DELTA_SLACK_ABS", "0.0"))

        # Candidate thresholds grid (+ include current)
        candidates: list[float] = []
        t = min_thr
        while t <= max_thr + 1e-9:
            candidates.append(round(t, 4))
            t += grid_step
        if self._current_threshold < min_thr or self._current_threshold > max_thr:
            self._current_threshold = max(min_thr, min(max_thr, float(self._current_threshold)))
        if all(abs(c - float(self._current_threshold)) > 1e-9 for c in candidates):
            candidates.append(float(self._current_threshold))

        now_ms = int(time.time() * 1000)
        # Compute smoothed expected delta for each candidate
        expected_delta_by_cand: dict[float, float] = {}
        for cand in candidates:
            sum_w = float(prior_weight)
            sum_w_delta = 0.0  # prior mean assumed 0
            for _, delta, thr, ts in vectors:
                if delta is None:
                    continue
                # Time decay
                try:
                    age_s = max(0.0, (now_ms - int(ts)) / 1000.0)
                except Exception:
                    age_s = 0.0
                w_time = 1.0 if tau <= 0.0 else (2.718281828 ** (-age_s / tau))
                # Threshold proximity kernel (Gaussian)
                diff = float(thr) - float(cand)
                w_thr = 1.0 if bw <= 0.0 else (2.718281828 ** (-(diff * diff) / (2.0 * bw * bw)))
                w = w_time * w_thr
                sum_w += w
                sum_w_delta += w * float(delta)
            expected_delta_by_cand[cand] = (sum_w_delta / sum_w) if sum_w > 0.0 else 0.0

        # Choose minimal threshold within slack of the best expected delta
        best_mean = max(expected_delta_by_cand.values()) if expected_delta_by_cand else 0.0
        if best_mean <= 0.0:
            chosen = float(self._current_threshold)
        else:
            slack = max(slack_abs, best_mean * slack_frac)
            viable = [c for c, m in expected_delta_by_cand.items() if m >= (best_mean - slack)]
            chosen = min(viable) if viable else float(self._current_threshold)

        # Exploration vs exploitation
        do_explore = random.random() < float(self.options.epsilon)
        thr = float(chosen)
        if do_explore:
            # Nudge towards lower thresholds if we are currently above the score
            direction = 1 if current_ml_bad_prob >= thr else -1
            step = float(self.options.threshold_step) * float(direction)
            thr = thr + step

        # Clamp and store
        thr = max(min_thr, min(max_thr, thr))
        self._current_threshold = thr
        return float(thr)

    def get_current_threshold(self) -> float:
        return float(self._current_threshold)

