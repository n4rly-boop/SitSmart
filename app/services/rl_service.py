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
        """Return notification threshold based on full history and epsilon-greedy policy.

        Goal: find minimal threshold that still keeps delta meaningful for the user.
        - With probability epsilon: explore by nudging the threshold down or up
        - Otherwise: exploit using history-derived success statistics
        """
        hist = HistoryService.get_instance()
        meaningful_delta = hist.get_meaningful_delta_threshold()

        # If we have no notion of what meaningful delta is yet, return current setting
        if meaningful_delta is None:
            return float(self._current_threshold)

        # Evaluate successes at different thresholds from history
        vectors = hist.get_history_vectors()  # (bad_prob, delta, threshold, ts)
        successes_by_threshold: dict[float, int] = {}
        counts_by_threshold: dict[float, int] = {}
        for bad_prob, delta, thr, _ in vectors:
            if delta is None:
                continue
            counts_by_threshold[thr] = counts_by_threshold.get(thr, 0) + 1
            if delta >= meaningful_delta:
                successes_by_threshold[thr] = successes_by_threshold.get(thr, 0) + 1

        def success_rate(thr: float) -> float:
            c = counts_by_threshold.get(thr, 0)
            if c == 0:
                return 0.0
            return float(successes_by_threshold.get(thr, 0)) / float(c)

        # Exploit: choose minimal threshold whose success rate is near-best
        unique_thresholds = sorted(set(counts_by_threshold.keys()))
        chosen = None
        if unique_thresholds:
            best_rate = 0.0
            for thr in unique_thresholds:
                r = success_rate(thr)
                if r > best_rate:
                    best_rate = r
            # Allow slack on best rate to favor lower thresholds
            slack = float(os.getenv("RL_SUCCESS_RATE_SLACK", "0.05"))
            viable = [thr for thr in unique_thresholds if success_rate(thr) >= (best_rate - slack)]
            if viable:
                chosen = min(viable)

        # Exploration vs Exploitation decision
        do_explore = random.random() < float(self.options.epsilon)
        if chosen is None:
            # No data yet – explore around current threshold
            do_explore = True

        thr = float(chosen) if chosen is not None else float(self._current_threshold)
        print(f"chosen: {chosen}, current_threshold: {thr}, do_explore: {do_explore}")
    
        if do_explore:
            # Move threshold towards minimal boundary to find minimal viable
            direction = -1 if current_ml_bad_prob >= thr else 1
            step = float(self.options.threshold_step) * float(direction)
            thr = thr + step

        # Clamp and store
        thr = max(self.options.min_threshold, min(self.options.max_threshold, thr))
        self._current_threshold = thr
        return float(thr)

    def get_current_threshold(self) -> float:
        return float(self._current_threshold)

