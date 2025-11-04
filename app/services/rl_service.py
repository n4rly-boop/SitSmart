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

    alpha: float = float(os.getenv("RL_ALPHA", 0.3)) # exploration-exploitation trade-off
    eta: float = float(os.getenv("RL_THRESHOLD_STEP", 0.03)) # threshold step size
    lambda_reg: float = float(os.getenv("RL_LAMBDA_REG", 1e-3)) # regularization parameter
    tau_min: float = float(os.getenv("RL_TAU_MIN", 0.5)) # minimum threshold
    tau_max: float = float(os.getenv("RL_TAU_MAX", 0.95)) # maximum threshold
    gamma: float = float(os.getenv("RL_FORGETTING_GAMMA", 1.0)) # forgetting factor
    penalty_notification: float = float(os.getenv("RL_PENALTY_NOTIF", 0.05)) # penalty for notification
    penalty_frequency: float = float(os.getenv("RL_PENALTY_FREQUENCY", 0.1)) # penalty for frequency
    reward_clip_min: float = -1.0 # minimum reward
    reward_clip_max: float = 1.0 # maximum reward
    initial_threshold: float = float(os.getenv("ML_BAD_PROB_THRESHOLD", 0.6)) # initial threshold
    # Mild inertia/guardrails
    switch_margin: float = float(os.getenv("RL_SWITCH_MARGIN", 0.02)) # min score gain to switch
    switch_cost: float = float(os.getenv("RL_SWITCH_COST", 0.01)) # penalty for non-zero/flip actions

    def actions(self) -> tuple[float, float, float]:
        step = abs(float(self.eta))
        return (0.0, -step, step)


@dataclass
class DecisionState:
    tick_id: int
    timestamp_ms: int
    bad_prob: float
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
        # Base context features (without action):
        self._base_dim = 6  # [1, bad_prob, bad_prob^2, thr, thr^2, bad_prob-thr]
        self._actions = self._config.actions()
        # Joint model over (base_context, action, action^2, base_context * action)
        self._joint_dim = 2 * self._base_dim + 2
        self._model: _ModelState = _ModelState.create(self._joint_dim, self._config.lambda_reg)
        # Per-action usage counters for annealing exploration
        self._action_counts: Dict[float, int] = {a: 0 for a in self._actions}

        thr = float(self._config.initial_threshold)
        self._current_threshold: float = float(self._clip(thr))
        self._tick: int = 0
        self._history_seen: int = 0
        self._last_reward: Optional[float] = None
        self._last_decision_threshold: float = self._current_threshold

        self._staged: Dict[int, DecisionState] = {}
        self._pending: Deque[DecisionState] = deque()
        self._awaiting_delta: Dict[int, DecisionState] = {}
        self._last_applied_action: float = 0.0

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

        context = self._build_context(bad_prob)
        # Do not change threshold here; notification uses current threshold only
        current_threshold = float(self._current_threshold)
        notify = bad_prob >= current_threshold

        tick_id = self._tick
        self._tick += 1
        decision = DecisionState(
            tick_id=tick_id,
            timestamp_ms=now_ms,
            bad_prob=bad_prob,
            context=context,
            action=self._last_applied_action,
            threshold_after=current_threshold,
            notify=notify,
        )
        self._staged[tick_id] = decision

        return {
            "tick_id": tick_id,
            "notify": notify,
            "new_threshold": float(current_threshold),
            "chosen_action": 0.0,
            "last_reward": None if self._last_reward is None else float(self._last_reward),
        }

    def commit_decision(self, tick_id: int, *, sent: bool, timestamp_ms: Optional[int] = None) -> None:
        """Finalize a staged decision once the notification pipeline acts."""
        decision = self._staged.pop(int(tick_id), None)
        if decision is None or not sent:
            return
        if timestamp_ms is not None:
            decision.timestamp_ms = int(timestamp_ms)
        self._pending.append(decision)
        self._refresh_history()

    def estimate_threshold(self, bad_prob: float, now_seconds: Optional[float] = None) -> float:
        """Return the next threshold adjustment without mutating state."""
        self._refresh_history()
        prob = float(np.clip(bad_prob, 0.0, 1.0))
        context = self._build_context(prob)
        action = self._select_action(context)
        return float(self._clip(self._current_threshold + action))

    def get_current_threshold(self) -> float:
        return float(self._current_threshold)

    def get_last_decision_threshold(self) -> float:
        return float(self._last_decision_threshold)

    def get_delta_baseline(self) -> Optional[float]:
        """Return the current EWMA delta baseline (mean delta)."""
        return self._delta_baseline

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _now_ms(self, now_seconds: Optional[float]) -> int:
        now = time.time() if now_seconds is None else float(now_seconds)
        return int(now * 1000)

    def _build_context(self, bad_prob: float) -> np.ndarray:
        thr = float(self._current_threshold)
        return np.array(
            [
                1.0,
                bad_prob,
                bad_prob**2,
                thr,
                thr**2,
                bad_prob - thr,
            ],
            dtype=np.float64,
        )

    def _build_joint_context(self, base_context: np.ndarray, action: float) -> np.ndarray:
        a = float(action)
        return np.concatenate([
            base_context,
            np.array([a, a * a], dtype=np.float64),
            base_context * a,
        ])

    def _select_action(self, base_context: np.ndarray) -> float:
        theta = self._model.theta()
        action_scores: Dict[float, float] = {}
        for action in self._actions:
            joint_ctx = self._build_joint_context(base_context, action)
            pred = float(joint_ctx @ theta)
            var = float(joint_ctx @ (self._model.A_inv @ joint_ctx))
            var = max(var, 0.0)
            # Anneal exploration by action usage count
            alpha_eff = float(self._config.alpha) / (self._action_counts.get(action, 0) + 1) ** 0.5
            score = pred + alpha_eff * var**0.5
            # Mild inertia: penalize non-zero actions and direction flips slightly
            if action != 0.0:
                score -= float(self._config.switch_cost)
                if self._last_applied_action != 0.0 and np.sign(action) != np.sign(self._last_applied_action):
                    score -= float(self._config.switch_cost)
            action_scores[action] = score

        # Choose best by score
        best_action = max(self._actions, key=lambda a: action_scores[a])
        best_score = action_scores[best_action]

        # Hysteresis: prefer staying (0.0) unless clear advantage over no-change
        zero_score = action_scores.get(0.0, float("-inf"))
        if best_action != 0.0 and (best_score - zero_score) < float(self._config.switch_margin):
            return 0.0

        # Hysteresis: avoid flipping unless clearly better than current direction
        cur = float(self._last_applied_action)
        if cur != 0.0 and np.sign(best_action) != np.sign(cur):
            cur_score = action_scores.get(cur, -1e18)
            if (best_score - cur_score) < float(self._config.switch_margin):
                return cur

        return float(best_action)

    def _clip(self, value: float) -> float:
        return float(max(self._config.tau_min, min(self._config.tau_max, value)))

    def _refresh_history(self) -> None:
        try:
            history = HistoryService.get_instance().get_notification_history()
        except Exception:
            return

        # Assign new history entries to pending decisions
        while self._history_seen < len(history):
            if self._pending:
                decision = self._pending.popleft()
                decision.history_index = self._history_seen
                self._awaiting_delta[self._history_seen] = decision
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

            # After applying reward, adapt the threshold now (post-delta)
            try:
                next_action = self._select_action(decision.context)
                new_thr = self._clip(self._current_threshold + next_action)
                self._current_threshold = new_thr
                self._last_decision_threshold = new_thr
                self._last_applied_action = float(next_action)
            except Exception:
                # If anything goes wrong, keep current threshold unchanged
                pass

    def _compute_reward(self, decision: DecisionState, delta_value: float) -> float:
        delta_val = max(float(delta_value or 0.0), 0.0)
        # Use user-specific meaningful delta as a baseline; if unavailable, fall back
        try:
            meaningful = HistoryService.get_instance().get_meaningful_delta_threshold()
        except Exception:
            meaningful = None
        if meaningful is None:
            meaningful = 0.1

        # Positive signal when delta is low (we want to move threshold up)
        signal = float(meaningful) - float(delta_val)
        action_sign = 0.0 if decision.action == 0.0 else (1.0 if decision.action > 0.0 else -1.0)

        shaped = signal * action_sign

        penalty_notif = self._config.penalty_notification
        penalty_freq = self._config.penalty_frequency
        reward = shaped - penalty_notif - penalty_freq
        return float(
            max(self._config.reward_clip_min, min(self._config.reward_clip_max, reward))
        )

    def _update_model(self, decision: DecisionState, reward: float) -> None:
        action = decision.action
        gamma = float(self._config.gamma)

        joint_ctx = self._build_joint_context(decision.context, action)

        if 0.0 < gamma < 1.0:
            self._model.A = gamma * self._model.A + (1.0 - gamma) * np.eye(self._joint_dim, dtype=np.float64)
            self._model.b = gamma * self._model.b

        self._model.A = self._model.A + np.outer(joint_ctx, joint_ctx)
        self._model.b = self._model.b + reward * joint_ctx
        try:
            self._model.A_inv = np.linalg.inv(self._model.A)
        except np.linalg.LinAlgError:
            self._model.A_inv = np.linalg.pinv(self._model.A)
        # Count usage for annealing
        try:
            self._action_counts[action] = int(self._action_counts.get(action, 0)) + 1
        except Exception:
            pass
