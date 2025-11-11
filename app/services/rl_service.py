import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import numpy as np

from app.services.history_service import HistoryService


@dataclass
class TSConfig:
    """Hyper-parameters for the threshold Thompson Sampling agent."""

    eta: float = float(os.getenv("RL_THRESHOLD_STEP", 0.03)) # threshold step size
    lambda_reg: float = float(os.getenv("RL_LAMBDA_REG", 1e-3)) # regularization parameter
    tau_min: float = float(os.getenv("RL_TAU_MIN", 0.5)) # minimum threshold
    tau_max: float = float(os.getenv("RL_TAU_MAX", 0.95)) # maximum threshold
    gamma: float = float(os.getenv("RL_FORGETTING_GAMMA", 1.0)) # forgetting factor
    penalty_notification: float = float(os.getenv("RL_PENALTY_NOTIF", 0.05)) # penalty for notification
    reward_clip_min: float = -1.0 # minimum reward
    reward_clip_max: float = 1.0 # maximum reward
    initial_threshold: float = float(os.getenv("ML_BAD_PROB_THRESHOLD", 0.6)) # initial threshold
    # Thompson Sampling noise scale
    ts_sigma: float = float(os.getenv("RL_TS_SIGMA", 0.05))
    # Small cost to changing threshold to discourage jitter
    change_cost: float = float(os.getenv("RL_CHANGE_COST", 0.002))
    # EWMA step for delta baseline (per-user)
    delta_baseline_beta: float = float(os.getenv("RL_DELTA_BASELINE_BETA", 0.1))

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


class ThresholdTSAgent:
    """Online Thompson Sampling agent that adapts the global notification threshold."""

    _instance: Optional["ThresholdTSAgent"] = None

    def __init__(self, config: Optional[TSConfig] = None) -> None:
        self._config = config or TSConfig()
        # Base context features (without action):
        # [1, bad_prob, bad_prob^2, thr, thr^2, bad_prob-thr, mu]
        self._base_dim = 7
        self._actions = self._config.actions()
        # Joint model over (base_context, action, action^2, base_context * action)
        self._joint_dim = 2 * self._base_dim + 2
        self._model: _ModelState = _ModelState.create(self._joint_dim, self._config.lambda_reg)

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
        self._delta_baseline: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @classmethod
    def get_instance(cls) -> "ThresholdTSAgent":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def decide(self, bad_prob: float, now_seconds: Optional[float] = None) -> Dict[str, float | bool | int | None]:
        """Stage a decision without adapting threshold. Notification uses current threshold only."""
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

    def _build_base_context(self, bad_prob: float, thr: float, mu: float) -> np.ndarray:
        return np.array([
            1.0,
            bad_prob,
            bad_prob**2,
            thr,
            thr**2,
            bad_prob - thr,
            mu,
        ], dtype=np.float64)

    def _build_context(self, bad_prob: float) -> np.ndarray:
        thr = float(self._current_threshold)
        mu = float(self._delta_baseline if self._delta_baseline is not None else 0.1)
        return self._build_base_context(bad_prob, thr, mu)

    def _build_joint_context(self, base_context: np.ndarray, action: float) -> np.ndarray:
        a = float(action)
        return np.concatenate([
            base_context,
            np.array([a, a * a], dtype=np.float64),
            base_context * a,
        ])

    def _select_action(self, base_context: np.ndarray) -> float:
        # Thompson Sampling: sample theta ~ N(θ̂, σ² A_inv)
        theta = self._model.theta()
        cov = float(self._config.ts_sigma) ** 2 * self._model.A_inv
        # Ensure numerical stability of covariance
        try:
            theta_sample = np.random.multivariate_normal(theta, cov)
        except Exception:
            jitter = 1e-8
            theta_sample = np.random.multivariate_normal(theta, cov + jitter * np.eye(len(theta)))

        best_action = 0.0
        best_score = float("-inf")
        for action in self._actions:
            joint_ctx = self._build_joint_context(base_context, action)
            score = float(joint_ctx @ theta_sample)
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
            # Rebuild base context at decision time using stored bad_prob, threshold, and current mu
            mu_val = float(self._delta_baseline if self._delta_baseline is not None else 0.1)
            base_ctx = self._build_base_context(float(decision.bad_prob), float(decision.threshold_after), mu_val)

            reward = self._compute_reward(decision, delta)
            self._update_model_with(base_ctx, float(decision.action), reward)
            self._last_reward = reward
            self._awaiting_delta.pop(idx, None)

            # After applying reward, adapt the threshold now (post-delta)
            try:
                next_action = self._select_action(base_ctx)
                new_thr = self._clip(self._current_threshold + next_action)
                self._current_threshold = new_thr
                self._last_decision_threshold = new_thr
                self._last_applied_action = float(next_action)
            except Exception:
                # If anything goes wrong, keep current threshold unchanged
                pass

            # Update EWMA delta baseline after using it for reward
            try:
                beta = float(self._config.delta_baseline_beta)
                d = float(delta if delta is not None else 0.0)
                if self._delta_baseline is None:
                    self._delta_baseline = d
                else:
                    self._delta_baseline = (1.0 - beta) * float(self._delta_baseline) + beta * d
                if self._delta_baseline < 0.0:
                    self._delta_baseline = 0.0
                if self._delta_baseline > 1.0:
                    self._delta_baseline = 1.0
            except Exception:
                pass

    def _compute_reward(self, decision: DecisionState, delta_value: float) -> float:
        delta_val = max(float(delta_value or 0.0), 0.0)
        mu = float(self._delta_baseline if self._delta_baseline is not None else 0.1)
        # Directional reward: sign(a)*(mu - delta)
        action_sign = 0.0 if decision.action == 0.0 else (1.0 if decision.action > 0.0 else -1.0)
        directional = (mu - delta_val) * action_sign
        # Small penalties
        penalty_notif = float(self._config.penalty_notification)
        change_penalty = float(self._config.change_cost) if decision.action != 0.0 else 0.0
        reward = directional - penalty_notif - change_penalty
        return float(
            max(self._config.reward_clip_min, min(self._config.reward_clip_max, reward))
        )

    def _update_model_with(self, base_context: np.ndarray, action: float, reward: float) -> None:
        gamma = float(self._config.gamma)

        joint_ctx = self._build_joint_context(base_context, action)

        if 0.0 < gamma < 1.0:
            self._model.A = gamma * self._model.A + (1.0 - gamma) * np.eye(self._joint_dim, dtype=np.float64)
            self._model.b = gamma * self._model.b

        self._model.A = self._model.A + np.outer(joint_ctx, joint_ctx)
        self._model.b = self._model.b + reward * joint_ctx
        try:
            self._model.A_inv = np.linalg.inv(self._model.A)
        except np.linalg.LinAlgError:
            self._model.A_inv = np.linalg.pinv(self._model.A)
