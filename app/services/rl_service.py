from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from app.services.history_service import HistoryService


@dataclass
class ThresholdBanditConfig:
    """Configuration for the contextual bandit that tunes the notification threshold."""

    theta_grid: Optional[List[float]] = None
    ucb_beta: float = float(os.getenv("RL_UCB_BETA", "0.7"))
    min_valid_delta: float = float(os.getenv("RL_MIN_VALID_DELTA", "0.02"))
    cooldown_seconds: float = float(os.getenv("NOTIFICATION_COOLDOWN_SECONDS", "60.0"))
    pending_alert_timeout_seconds: float = float(os.getenv("RL_PENDING_ALERT_TIMEOUT_SECONDS", "30.0"))

    def build_grid(self) -> List[float]:
        """Construct θ-grid; defaults to [0.55, 0.60, ..., 0.90]."""
        if self.theta_grid:
            return [float(round(th, 4)) for th in self.theta_grid]
        min_thr = float(os.getenv("RL_MIN_THRESHOLD", "0.55"))
        max_thr = float(os.getenv("RL_MAX_THRESHOLD", "0.9"))
        step = float(os.getenv("RL_GRID_STEP", "0.05"))
        grid: List[float] = []
        current = min_thr
        while current <= max_thr + 1e-9:
            grid.append(round(current, 4))
            current += step
        if not grid:
            grid = [round(min_thr, 4)]
        return grid


@dataclass
class DecisionRecord:
    """Bookkeeping for a single notification experiment."""

    tick_id: int
    theta: float
    decision_ms: int
    bad_prob: float
    history_index: Optional[int] = None
    reward_applied: bool = False


class ThresholdBanditAgent:
    """Contextual bandit that tunes the ML bad-posture threshold using UCB1."""

    def __init__(self, config: Optional[ThresholdBanditConfig] = None) -> None:
        self.config = config or ThresholdBanditConfig()
        self.theta_grid = self.config.build_grid()
        self.count: Dict[float, int] = {th: 0 for th in self.theta_grid}
        self.sum_reward: Dict[float, float] = {th: 0.0 for th in self.theta_grid}
        self.mean_reward: Dict[float, float] = {th: 0.0 for th in self.theta_grid}
        self.total_steps: int = 1
        self._next_tick_id: int = 0
        self._current_theta: float = self.theta_grid[0]

        # Pending bookkeeping
        self._awaiting_alert: Deque[DecisionRecord] = deque()
        self._decisions_by_tick: Dict[int, DecisionRecord] = {}
        self._decisions_by_history: Dict[int, DecisionRecord] = {}
        self._history_index: int = 0
        self._last_notification_ms: int = -10_000_000_000

        self._pending_alert_timeout_ms: int = int(
            max(0.0, self.config.pending_alert_timeout_seconds) * 1000
        )
        self._cooldown_ms: int = int(max(0.0, self.config.cooldown_seconds) * 1000)
        self.min_valid_delta: float = float(self.config.min_valid_delta)

    def step(self, p_t: float, now_seconds: Optional[float] = None) -> Dict[str, float]:
        """Consume one ML tick. Threshold updates only when a notification fires."""
        now = time.time() if now_seconds is None else now_seconds
        now_ms = int(now * 1000)
        self._process_history(now)

        bad_prob = float(max(0.0, min(1.0, p_t)))
        can_alert = (now_ms - self._last_notification_ms) >= self._cooldown_ms

        if not can_alert:
            return {
                "tick_id": -1.0,
                "theta": float(self._current_theta),
                "alert": 0.0,
            }

        theta = self._select_theta(bad_prob)
        if theta is None:
            return {
                "tick_id": -1.0,
                "theta": float(self._current_theta),
                "alert": 0.0,
            }

        self._current_theta = theta
        tick_id = self._next_tick_id
        self._next_tick_id += 1
        self.count[theta] += 1
        self.total_steps += 1
        self._last_notification_ms = now_ms

        decision = DecisionRecord(
            tick_id=tick_id,
            theta=theta,
            decision_ms=now_ms,
            bad_prob=bad_prob,
        )
        self._awaiting_alert.append(decision)
        self._decisions_by_tick[tick_id] = decision

        return {"tick_id": float(tick_id), "theta": float(theta), "alert": 1.0}

    def suggest_threshold(self, p_t: float) -> float:
        """Public API: pick θ given current bad posture probability."""
        return float(self.step(p_t)["theta"])

    def observe_delta(self, tick_id: int, delta: Optional[float]) -> None:
        """Optional manual reward hook for tests/simulations."""
        decision = self._decisions_by_tick.get(int(tick_id))
        if decision is None or decision.reward_applied:
            return
        reward = self._safe_reward(delta)
        if reward is None:
            reward = 0.0
        self._apply_reward(decision, reward)

    def diagnostics(self) -> Dict[str, object]:
        ranked = sorted(
            [(th, self.mean_reward[th], self.count[th]) for th in self.theta_grid],
            key=lambda x: x[1],
            reverse=True,
        )
        return {
            "total_steps": int(self.total_steps - 1),
            "last_notification_ms": int(self._last_notification_ms),
            "pending_alerts": len(self._awaiting_alert),
            "top_thresholds": ranked[:3],
        }

    def get_current_threshold(self) -> float:
        return float(self._current_theta)

    # ----- Internals -----

    def _select_theta(self, bad_prob: float) -> Optional[float]:
        eligible = [th for th in self.theta_grid if th <= bad_prob + 1e-9]
        if not eligible:
            return None
        for th in eligible:
            if self.count[th] == 0:
                return th
        t = max(self.total_steps, 2)
        best_theta = self.theta_grid[0]
        best_score = float("-inf")
        log_term = math.log(float(t))
        for th in eligible:
            n = self.count[th]
            mean = self.mean_reward[th]
            bonus = self.config.ucb_beta * math.sqrt(log_term / float(n))
            score = mean + bonus
            if score > best_score:
                best_score = score
                best_theta = th
        return best_theta

    def _process_history(self, now_seconds: float) -> None:
        now_ms = int(now_seconds * 1000)
        timeout_ms = self._pending_alert_timeout_ms
        try:
            history = HistoryService.get_instance().get_notification_history()
        except Exception:
            history = []

        if timeout_ms > 0:
            while self._awaiting_alert:
                pending = self._awaiting_alert[0]
                if (now_ms - pending.decision_ms) <= timeout_ms:
                    break
                self._awaiting_alert.popleft()
                pending.reward_applied = True
                self._decisions_by_tick.pop(pending.tick_id, None)

        while self._history_index < len(history):
            record = history[self._history_index]
            try:
                ts_ms = int(record.timestamp_ms)
            except Exception:
                ts_ms = now_ms
            self._last_notification_ms = max(self._last_notification_ms, ts_ms)
            if self._awaiting_alert:
                decision = self._awaiting_alert.popleft()
                decision.history_index = self._history_index
                self._decisions_by_history[self._history_index] = decision
            self._history_index += 1

        for idx in list(self._decisions_by_history.keys()):
            if idx >= len(history):
                continue
            decision = self._decisions_by_history[idx]
            if decision.reward_applied:
                self._decisions_by_history.pop(idx, None)
                continue
            record = history[idx]
            reward = self._safe_reward(getattr(record, "delta", None))
            if reward is None:
                continue
            self._apply_reward(decision, reward)
            self._decisions_by_history.pop(idx, None)

    def _safe_reward(self, delta: Optional[float]) -> Optional[float]:
        if delta is None:
            return None
        try:
            val = float(delta)
        except (TypeError, ValueError):
            return 0.0
        if val < 0.0 or val > 1.0:
            return 0.0
        if val < self.min_valid_delta:
            return 0.0
        return val

    def _apply_reward(self, decision: DecisionRecord, reward: float) -> None:
        theta = decision.theta
        self.sum_reward[theta] += reward
        self.mean_reward[theta] = self.sum_reward[theta] / max(self.count[theta], 1)
        decision.reward_applied = True
        self._decisions_by_tick.pop(decision.tick_id, None)
        try:
            self._awaiting_alert.remove(decision)
        except ValueError:
            pass


class EpsilonGreedyAgent(ThresholdBanditAgent):
    """Backward-compatible name retained for the API surface."""

    _instance: Optional["EpsilonGreedyAgent"] = None

    @classmethod
    def get_instance(cls) -> "EpsilonGreedyAgent":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


if __name__ == "__main__":
    # Minimal demonstration run with synthetic deltas
    import random

    agent = ThresholdBanditAgent()
    now = time.time()
    for _ in range(200):
        p = random.random()
        step = agent.step(p, now)
        if step["alert"] > 0.5:
            # Sample delta proportional to probability, clip to [0,1]
            delta = max(0.0, min(1.0, p + random.uniform(-0.1, 0.1)))
            agent.observe_delta(int(step["tick_id"]), delta)
        now += 2.0
    print("Diagnostics:", agent.diagnostics())
