import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import numpy as np

from app.services.history_service import HistoryService


@dataclass
class TSConfig:
    """Hyper-parameters for the adaptive threshold agent.
    
    Core RL Parameters:
    - eta: Step size for threshold adjustments (actions: -eta, 0, +eta). Larger = faster adaptation but more jitter.
    
    Threshold Bounds:
    - tau_min, tau_max: Hard limits for notification threshold. Prevents extreme values.
    - initial_threshold: Starting threshold value. Should match user's initial ML threshold setting.
    
    Reward Shaping:
    - penalty_notification: Per-notification penalty (always applied). Reduces spam, encourages selectivity.
    - change_cost: Small penalty for changing threshold (action != 0). Reduces jitter, encourages stability.
    - reward_clip_min/max: Hard bounds on reward values. Prevents extreme updates, stabilizes learning.
    
    Delta Baseline (User Adaptation):
    - delta_baseline_beta: EWMA step for tracking user's mean delta. Higher = faster adaptation to user changes.
    
    Band-Based Boundaries (Delta Band):
    - band_q_low/high: Target quantiles for adaptive band boundaries. Low delta < L → raise threshold, delta > H → lower threshold.
    - band_quantile_lr: Learning rate for online quantile estimation (Robbins-Monro). Lower = slower adaptation.
    """

    # Core RL
    eta: float = float(os.getenv("RL_THRESHOLD_STEP", 0.03))
    
    # Threshold bounds
    tau_min: float = float(os.getenv("RL_TAU_MIN", 0.5))
    tau_max: float = float(os.getenv("RL_TAU_MAX", 0.95))
    initial_threshold: float = float(os.getenv("ML_BAD_PROB_THRESHOLD", 0.6))
    
    # Reward shaping
    penalty_notification: float = float(os.getenv("RL_PENALTY_NOTIF", 0.05))
    change_cost: float = float(os.getenv("RL_CHANGE_COST", 0.002))
    reward_clip_min: float = float(os.getenv("RL_REWARD_CLIP_MIN", -1.0))
    reward_clip_max: float = float(os.getenv("RL_REWARD_CLIP_MAX", 1.0))
    
    # Delta baseline
    delta_baseline_beta: float = float(os.getenv("RL_DELTA_BASELINE_BETA", 0.1))
    
    # Band boundaries (learned via quantiles)
    band_q_low: float = float(os.getenv("RL_BAND_Q_LOW", 0.3))
    band_q_high: float = float(os.getenv("RL_BAND_Q_HIGH", 0.7))
    band_quantile_lr: float = float(os.getenv("RL_BAND_Q_LR", 0.02))

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


class ThresholdTSAgent:
    """Adaptive threshold agent with deterministic action selection.
    
    Actions are rule-based based on delta bounds (L/H):
    - delta < L → raise threshold
    - delta > H → lower threshold
    - L ≤ delta ≤ H → hold
    
    The agent learns to adapt L/H boundaries via online quantile estimation.
    """

    _instance: Optional["ThresholdTSAgent"] = None

    def __init__(self, config: Optional[TSConfig] = None) -> None:
        self._config = config or TSConfig()
        self._actions = self._config.actions()

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
        # Online quantile trackers for band boundaries
        self._q_low_val: float = 0.2
        self._q_high_val: float = 0.8

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

    def get_band_bounds(self) -> tuple[Optional[float], Optional[float]]:
        """Return the current delta band boundaries (L, H)."""
        L, H = self._get_band_bounds()
        return (L, H)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _now_ms(self, now_seconds: Optional[float]) -> int:
        now = time.time() if now_seconds is None else float(now_seconds)
        return int(now * 1000)

    def _build_base_context(self, bad_prob: float, thr: float, mu: float) -> np.ndarray:
        """Build context for tracking (not used for action selection anymore)."""
        L, H = self._get_band_bounds()
        return np.array([1.0, bad_prob, bad_prob**2, thr, thr**2, bad_prob - thr, mu, L, H], dtype=np.float64)

    def _build_context(self, bad_prob: float) -> np.ndarray:
        mu = float(self._delta_baseline if self._delta_baseline is not None else 0.1)
        return self._build_base_context(bad_prob, float(self._current_threshold), mu)


    def _select_action(self, base_context: Optional[np.ndarray], delta: Optional[float] = None) -> float:
        """Select action deterministically based on delta bounds.
        
        Actions are now rule-based:
        - delta < L → raise threshold (+eta)
        - delta > H → lower threshold (-eta)  
        - L ≤ delta ≤ H → hold (0)
        
        The RL agent only learns to adapt L/H boundaries via online quantile estimation.
        """
        if delta is not None:
            d = float(max(0.0, min(1.0, delta)))
        elif base_context is not None and len(base_context) > 6:
            # Use mu (delta baseline) as proxy when delta not available
            d = float(base_context[6])
        else:
            d = float(self._delta_baseline if self._delta_baseline is not None else 0.1)
        
        L, H = self._get_band_bounds()
        
        if d < L:
            return float(self._config.eta)  # Raise threshold
        elif d > H:
            return -float(self._config.eta)  # Lower threshold
        else:
            return 0.0  # Hold

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
            mu_val = float(self._delta_baseline if self._delta_baseline is not None else 0.1)
            reward = self._compute_reward(decision, delta)
            self._last_reward = reward
            self._awaiting_delta.pop(idx, None)

            d = float(max(0.0, min(1.0, delta if delta is not None else 0.0)))
            
            try:
                # Use actual delta for deterministic action selection
                next_action = self._select_action(None, delta=d)
                new_thr = self._clip(self._current_threshold + next_action)
                self._current_threshold = new_thr
                self._last_decision_threshold = new_thr
                self._last_applied_action = float(next_action)
            except Exception:
                pass

            self._update_delta_baseline(d)
            self._update_band_quantiles(d)

    def _compute_reward(self, decision: DecisionState, delta_value: float) -> float:
        """Compute reward for learning L/H boundaries.
        
        Since actions are now deterministic, reward is used only to validate
        that the chosen action was appropriate, which helps adapt L/H boundaries.
        """
        d = float(max(0.0, min(1.0, delta_value)))
        L, H = self._get_band_bounds()
        
        # Simple reward: positive if action matches desired behavior, negative otherwise
        if decision.action > 0.0:  # +eta (raise threshold)
            # Good if delta < L, bad otherwise
            reward = 1.0 if d < L else -0.5
        elif decision.action < 0.0:  # -eta (lower threshold)
            # Good if delta > H, bad otherwise
            reward = 1.0 if d > H else -0.5
        else:  # hold (action = 0)
            # Good if L ≤ delta ≤ H, bad otherwise
            reward = 1.0 if (L <= d <= H) else -0.5
        
        penalty = float(self._config.penalty_notification)
        change_penalty = float(self._config.change_cost) if decision.action != 0.0 else 0.0
        reward = reward - penalty - change_penalty
        return float(max(self._config.reward_clip_min, min(self._config.reward_clip_max, reward)))

    def _get_band_bounds(self) -> tuple[float, float]:
        L = float(max(0.0, min(1.0, self._q_low_val)))
        H = float(max(0.0, min(1.0, self._q_high_val)))
        if H < L:
            mid = 0.5 * (L + H)
            L, H = max(0.0, mid - 1e-3), min(1.0, mid + 1e-3)
        return (L, H)

    def _update_delta_baseline(self, delta_val: float) -> None:
        beta = float(self._config.delta_baseline_beta)
        if self._delta_baseline is None:
            self._delta_baseline = delta_val
        else:
            self._delta_baseline = (1.0 - beta) * float(self._delta_baseline) + beta * delta_val
        self._delta_baseline = float(max(0.0, min(1.0, self._delta_baseline)))

    def _update_band_quantiles(self, delta_val: float) -> None:
        lr = float(self._config.band_quantile_lr)
        qL, qH = float(self._config.band_q_low), float(self._config.band_q_high)
        self._q_low_val += lr * (qL - (1.0 if delta_val <= self._q_low_val else 0.0))
        self._q_high_val += lr * (qH - (1.0 if delta_val <= self._q_high_val else 0.0))
        self._q_low_val = float(max(0.0, min(1.0, self._q_low_val)))
        self._q_high_val = float(max(0.0, min(1.0, self._q_high_val)))
        if self._q_high_val < self._q_low_val:
            mid = 0.5 * (self._q_low_val + self._q_high_val)
            self._q_low_val, self._q_high_val = max(0.0, mid - 1e-3), min(1.0, mid + 1e-3)
