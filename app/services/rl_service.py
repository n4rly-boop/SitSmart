from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import numpy as np

from app.services.history_service import HistoryService


@dataclass
class LinUCBConfig:
    """Hyper-parameters for the threshold LinUCB agent."""

    alpha: float = float(os.getenv("RL_ALPHA", 0.7))
    eta: float = float(os.getenv("RL_THRESHOLD_STEP", 0.03))
    lambda_reg: float = float(os.getenv("RL_LAMBDA_REG", 1e-3))
    tau_min: float = float(os.getenv("RL_TAU_MIN", 0.5))
    tau_max: float = float(os.getenv("RL_TAU_MAX", 0.95))
    gamma: float = float(os.getenv("RL_FORGETTING_GAMMA", 0.99))
    penalty_notification: float = float(os.getenv("RL_PENALTY_NOTIF", 0.05))
    penalty_frequency: float = float(os.getenv("RL_PENALTY_FREQUENCY", 0.1))
    default_time_since_last: float = float(os.getenv("RL_DEFAULT_TIME_SINCE_LAST", 30.0))
    max_time_since_last: float = float(os.getenv("RL_MAX_TIME_SINCE_LAST", 600.0))
    reward_clip_min: float = -1.0
    reward_clip_max: float = 1.0
    initial_threshold: float = float(os.getenv("ML_BAD_PROB_THRESHOLD", 0.6))

    def actions(self) -> tuple[float, float, float]:
        step = abs(float(self.eta))
        return (-step, 0.0, step)


@dataclass
class DecisionState:
    tick_id: int
    timestamp_ms: int
    bad_prob: float
    time_since_last: float
    context: np.ndarray
    action: float
    threshold_after: float
    notify: bool
    history_index: Optional[int] = None


@dataclass
class _ModelState:
    A: np.ndarray
    A_inv: np.ndarray
    b: np.ndarray

    @classmethod
    def create(cls, dim: int, regularization: float) -> "_ModelState":
        A = np.eye(dim, dtype=np.float64) * float(regularization)
        return cls(A=A, A_inv=np.linalg.inv(A), b=np.zeros(dim, dtype=np.float64))

    def theta(self) -> np.ndarray:
        return self.A_inv @ self.b


class ThresholdLinUCBAgent:
    """Online LinUCB agent that adapts the global notification threshold."""

    _instance: Optional["ThresholdLinUCBAgent"] = None

    def __init__(self, config: Optional[LinUCBConfig] = None) -> None:
        self._config = config or LinUCBConfig()
        self._dim = 6  # Context: [1, bad_prob, tsl, bad_prob^2, tsl^2, cross]
        self._actions = self._config.actions()
        self._models: Dict[float, _ModelState] = {
            action: _ModelState.create(self._dim, self._config.lambda_reg) for action in self._actions
        }

        thr = float(self._config.initial_threshold)
        self._current_threshold: float = float(self._clip(thr))
        self._tick: int = 0
        self._history_seen: int = 0
        self._last_notification_ms: Optional[int] = None
        self._last_reward: Optional[float] = None
        self._last_decision_threshold: float = self._current_threshold

        self._staged: Dict[int, DecisionState] = {}
        self._pending: Deque[DecisionState] = deque()
        self._awaiting_delta: Dict[int, DecisionState] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @classmethod
    def get_instance(cls) -> "ThresholdLinUCBAgent":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def decide(self, bad_prob: float, now_seconds: Optional[float] = None) -> Dict[str, float | bool | int | None]:
        """Run the LinUCB policy once and stage the outcome."""
        now_ms = self._now_ms(now_seconds)
        self._refresh_history()

        prob = float(np.clip(bad_prob, 0.0, 1.0))
        tsl = self._time_since_last(now_ms)
        context = self._build_context(prob, tsl)
        action = self._select_action(context)

        new_threshold = self._clip(self._current_threshold + action)
        notify = prob >= new_threshold

        tick_id = self._tick
        self._tick += 1
        decision = DecisionState(
            tick_id=tick_id,
            timestamp_ms=now_ms,
            bad_prob=prob,
            time_since_last=tsl,
            context=context,
            action=action,
            threshold_after=new_threshold,
            notify=notify,
        )
        self._staged[tick_id] = decision
        self._current_threshold = new_threshold
        self._last_decision_threshold = new_threshold

        return {
            "tick_id": tick_id,
            "notify": notify,
            "new_threshold": float(new_threshold),
            "chosen_action": float(action),
            "last_reward": None if self._last_reward is None else float(self._last_reward),
        }

    def commit_decision(self, tick_id: int, *, sent: bool, timestamp_ms: Optional[int] = None) -> None:
        """Finalize a staged decision once the notification pipeline acts."""
        decision = self._staged.pop(int(tick_id), None)
        if decision is None or not sent:
            return
        if timestamp_ms is not None:
            decision.timestamp_ms = int(timestamp_ms)
        self._last_notification_ms = decision.timestamp_ms
        self._pending.append(decision)
        self._refresh_history()

    def estimate_threshold(self, bad_prob: float, now_seconds: Optional[float] = None) -> float:
        """Return the next threshold adjustment without mutating state."""
        now_ms = self._now_ms(now_seconds)
        self._refresh_history()
        prob = float(np.clip(bad_prob, 0.0, 1.0))
        tsl = self._time_since_last(now_ms)
        context = self._build_context(prob, tsl)
        action = self._select_action(context)
        return float(self._clip(self._current_threshold + action))

    def get_current_threshold(self) -> float:
        return float(self._current_threshold)

    def get_last_decision_threshold(self) -> float:
        return float(self._last_decision_threshold)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _now_ms(self, now_seconds: Optional[float]) -> int:
        now = time.time() if now_seconds is None else float(now_seconds)
        return int(now * 1000)

    def _time_since_last(self, now_ms: int) -> float:
        if self._last_notification_ms is None:
            return float(self._config.default_time_since_last)
        elapsed = max(0.0, (now_ms - self._last_notification_ms) / 1000.0)
        return float(min(elapsed, self._config.max_time_since_last))

    def _build_context(self, bad_prob: float, tsl: float) -> np.ndarray:
        return np.array(
            [1.0, bad_prob, tsl, bad_prob**2, tsl**2, bad_prob * tsl],
            dtype=np.float64,
        )

    def _select_action(self, context: np.ndarray) -> float:
        best_score = float("-inf")
        best_action = 0.0
        for action in self._actions:
            model = self._models[action]
            theta = model.theta()
            pred = float(context @ theta)
            var = float(context @ (model.A_inv @ context))
            var = max(var, 0.0)
            score = pred + self._config.alpha * var**0.5
            if score > best_score:
                best_score = score
                best_action = action
        return float(best_action)

    def _clip(self, value: float) -> float:
        return float(max(self._config.tau_min, min(self._config.tau_max, value)))

    def _refresh_history(self) -> None:
        try:
            history = HistoryService.get_instance().get_notification_history()
        except Exception:
            return

        if self._history_seen == 0 and history:
            try:
                self._last_notification_ms = int(history[-1].timestamp_ms)
            except Exception:
                pass

        # Assign new history entries to pending decisions
        while self._history_seen < len(history):
            record = history[self._history_seen]
            if self._pending:
                decision = self._pending.popleft()
                decision.history_index = self._history_seen
                self._awaiting_delta[self._history_seen] = decision
            try:
                self._last_notification_ms = int(record.timestamp_ms)
            except Exception:
                pass
            self._history_seen += 1

        # Apply rewards where delta is ready
        for idx, decision in list(self._awaiting_delta.items()):
            if idx >= len(history):
                continue
            record = history[idx]
            delta = getattr(record, "delta", None)
            if delta is None:
                continue
            reward = self._compute_reward(decision, delta)
            self._update_model(decision, reward)
            self._last_reward = reward
            self._awaiting_delta.pop(idx, None)

    def _compute_reward(self, decision: DecisionState, delta_value: float) -> float:
        delta_val = max(float(delta_value or 0.0), 0.0)
        penalty_notif = self._config.penalty_notification
        penalty_freq = self._config.penalty_frequency / max(decision.time_since_last, 1e-3)
        reward = delta_val - penalty_notif - penalty_freq
        return float(
            max(self._config.reward_clip_min, min(self._config.reward_clip_max, reward))
        )

    def _update_model(self, decision: DecisionState, reward: float) -> None:
        action = decision.action
        model = self._models[action]
        gamma = float(self._config.gamma)

        if 0.0 < gamma < 1.0:
            model.A = gamma * model.A + (1.0 - gamma) * np.eye(self._dim, dtype=np.float64)
            model.b = gamma * model.b

        model.A = model.A + np.outer(decision.context, decision.context)
        model.b = model.b + reward * decision.context
        try:
            model.A_inv = np.linalg.inv(model.A)
        except np.linalg.LinAlgError:
            model.A_inv = np.linalg.pinv(model.A)
        self._models[action] = model

