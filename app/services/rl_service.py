from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import asdict
from typing import Dict, List, Optional

from app.api.schemas import ModelAnalysisRequest, ModelAnalysisResponse


# ---------------- RLConfig ----------------
class RLConfig:
    """Configuration for the posture notification RL agent.

    Notes:
    - State is persisted to JSON to avoid losing learning progress across runs.
    - Epsilon is decayed gradually; persistence ensures it doesn't reset each run.
    """

    # Exploration
    epsilon_start: float = 0.2
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.999

    # Learning
    learning_rate: float = 0.3

    # Reward window
    reward_window_seconds: int = 5
    # Minimum seconds between agent decisions (throttling)
    decision_interval_seconds: int = 6

    # Considered a success if badness reduces by this fraction OR falls under thresholds
    improvement_ratio_threshold: float = 0.2

    # Feature scaling to keep values in a reasonable range for linear models
    scale_shoulder_angle_deg: float = 10.0
    scale_head_tilt_deg: float = 15.0
    scale_head_drop_ratio: float = 0.5
    scale_time_since_notify_s: float = 60.0
    scale_moving_avg_reward: float = 1.0

    # Persistence
    state_file_path: str = os.getenv(
        "RL_AGENT_STATE_PATH",
        os.path.join(os.getcwd(), "train", "rl_agent_state.json"),
    )


# ---------------- PostureRLAgent ----------------
class PostureRLAgent:
    """A simple epsilon-greedy contextual bandit for posture notifications.

    - Binary action space: 0 = do not notify, 1 = notify
    - Linear value model per action: Q(a, x) = w_a · x
    - Online update with learning rate on observed rewards
    - Persistent state to JSON file so learning survives restarts
    """

    def __init__(self, config: Optional[RLConfig] = None):
        self.config = config or RLConfig()
        self.weights: Dict[int, List[float]] = {0: [], 1: []}
        self.epsilon: float = self.config.epsilon_start
        self.total_updates: int = 0
        # Per-action reward accounting
        self.action_reward_sum: Dict[int, float] = {0: 0.0, 1: 0.0}
        self.action_reward_count: Dict[int, int] = {0: 0, 1: 0}
        self.action_last_reward: Dict[int, Optional[float]] = {0: None, 1: None}
        self._state_loaded: bool = False
        self._ensure_state_dir()
        self._load_state()

    # ---------- Public API ----------
    def select_action(self, state_vector: List[float]) -> int:
        """Choose an action with epsilon-greedy exploration."""
        self._maybe_resize_weights(len(state_vector))
        eps = max(self.config.epsilon_min, self.epsilon)
        # Random exploration
        if random.random() < eps:
            action = 1 if random.random() < 0.5 else 0
        else:
            q0 = _dot(self.weights[0], state_vector)
            q1 = _dot(self.weights[1], state_vector)
            action = 1 if q1 >= q0 else 0
        # Epsilon decay per decision
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
        self._save_state()
        return action

    def learn(self, state_vector: List[float], action: int, reward: float) -> None:
        """Online update for the chosen action's weights using gradient step."""
        self._maybe_resize_weights(len(state_vector))
        action = 1 if action else 0
        current_q = _dot(self.weights[action], state_vector)
        td_error = reward - current_q
        lr = self.config.learning_rate
        for i in range(len(self.weights[action])):
            self.weights[action][i] += lr * td_error * state_vector[i]
        # Per-action reward tracking
        self.action_reward_sum[action] = float(self.action_reward_sum.get(action, 0.0)) + float(reward)
        self.action_reward_count[action] = int(self.action_reward_count.get(action, 0)) + 1
        self.action_last_reward[action] = float(reward)
        self.total_updates += 1
        self._save_state()

    def get_persistent_features(self) -> Dict[str, float]:
        return {
            "epsilon": float(self.epsilon),
            "total_updates": float(self.total_updates),
            "rewards": {
                "sum": {"0": float(self.action_reward_sum.get(0, 0.0)), "1": float(self.action_reward_sum.get(1, 0.0))},
                "count": {"0": int(self.action_reward_count.get(0, 0)), "1": int(self.action_reward_count.get(1, 0))},
                "last": {"0": self.action_last_reward.get(0), "1": self.action_last_reward.get(1)},
            },
        }

    # ---------- Persistence ----------
    def _ensure_state_dir(self) -> None:
        state_dir = os.path.dirname(self.config.state_file_path)
        if state_dir and not os.path.exists(state_dir):
            try:
                os.makedirs(state_dir, exist_ok=True)
            except Exception:
                pass

    def _load_state(self) -> None:
        if self._state_loaded:
            return
        path = self.config.state_file_path
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                w0 = data.get("weights", {}).get("0", [])
                w1 = data.get("weights", {}).get("1", [])
                self.weights = {0: list(map(float, w0)), 1: list(map(float, w1))}
                self.epsilon = float(data.get("epsilon", self.config.epsilon_start))
                self.total_updates = int(data.get("total_updates", 0))
                # Backward compatible: reward stats may be absent
                r = data.get("rewards", {})
                self.action_reward_sum = {0: float((r.get("sum", {}) or {}).get("0", 0.0)), 1: float((r.get("sum", {}) or {}).get("1", 0.0))}
                self.action_reward_count = {0: int((r.get("count", {}) or {}).get("0", 0)), 1: int((r.get("count", {}) or {}).get("1", 0))}
                last = r.get("last", {}) or {}
                lr0 = last.get("0", None)
                lr1 = last.get("1", None)
                self.action_last_reward = {0: (None if lr0 is None else float(lr0)), 1: (None if lr1 is None else float(lr1))}
        except Exception:
            # If state is corrupt, reset to defaults
            self.weights = {0: [], 1: []}
            self.epsilon = self.config.epsilon_start
            self.total_updates = 0
        finally:
            self._state_loaded = True

    def _save_state(self) -> None:
        try:
            payload = {
                "weights": {"0": self.weights[0], "1": self.weights[1]},
                "epsilon": self.epsilon,
                "total_updates": self.total_updates,
                "rewards": {
                    "sum": {"0": self.action_reward_sum.get(0, 0.0), "1": self.action_reward_sum.get(1, 0.0)},
                    "count": {"0": self.action_reward_count.get(0, 0), "1": self.action_reward_count.get(1, 0)},
                    "last": {"0": self.action_last_reward.get(0), "1": self.action_last_reward.get(1)},
                },
                "config": asdict(self.config),
            }
            with open(self.config.state_file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception:
            pass

    # ---------- Internal ----------
    def _maybe_resize_weights(self, dim: int) -> None:
        for a in (0, 1):
            if len(self.weights[a]) < dim:
                self.weights[a].extend([0.0] * (dim - len(self.weights[a])))


def _dot(w: List[float], x: List[float]) -> float:
    return float(sum((w[i] * x[i]) for i in range(min(len(w), len(x)))))


class RLService:
    """Service encapsulating RL decision logic and state endpoints.

    Responsibilities:
    - Convert features into a model state vector
    - Select action via epsilon-greedy agent
    - Schedule reward evaluation based on post-notification features
    - Expose current agent state for inspection
    """

    _instance: Optional["RLService"] = None

    def __init__(self):
        self._agent = PostureRLAgent()
        self._recent_features: List[Dict[str, object]] = []
        self._pending_eval: Optional[Dict[str, object]] = None
        self._last_decision_at_ms: int = 0
        self._last_notified_at_ms: int = 0
        self._training_enabled: bool = True

    # ---------- Singleton ----------
    @classmethod
    def get_instance(cls) -> "RLService":
        if cls._instance is None:
            cls._instance = RLService()
        return cls._instance

    # ---------- Public API ----------
    def set_training_enabled(self, enabled: bool) -> None:
        self._training_enabled = bool(enabled)
        if not self._training_enabled:
            # Cancel any pending evaluation to avoid learning during ML mode
            self._pending_eval = None

    def analyze(self, req: ModelAnalysisRequest) -> ModelAnalysisResponse:
        now_ms = int(time.time() * 1000)
        self._record_features(req.features.model_dump(), now_ms)

        # Throttle decisions
        interval_ms = max(0, int(self._agent.config.decision_interval_seconds) * 1000)
        if (now_ms - self._last_decision_at_ms) < interval_ms:
            return ModelAnalysisResponse(should_notify=False, reason="throttled")

        state_vector = self._build_state_vector(req.features.model_dump(), now_ms)
        action = self._agent.select_action(state_vector)
        self._last_decision_at_ms = now_ms

        should_notify = action == 1
        details = {"action": action, "epsilon": self._agent.epsilon}

        if should_notify and self._training_enabled:
            self._last_notified_at_ms = now_ms
            # schedule reward evaluation
            reward_window_ms = max(0, int(self._agent.config.reward_window_seconds) * 1000)
            deadline_ms = now_ms + reward_window_ms
            self._pending_eval = {
                "deadline_ms": deadline_ms,
                "baseline_badness": self._compute_badness(req.features.model_dump()),
                "state_vector": state_vector,
            }
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._delayed_evaluate(deadline_ms))
            except Exception:
                # Fallback to background thread running asyncio if loop missing
                try:
                    import threading

                    def _runner():
                        asyncio.run(self._delayed_evaluate(deadline_ms))

                    t = threading.Thread(target=_runner, daemon=True)
                    t.start()
                except Exception:
                    pass

        return ModelAnalysisResponse(
            should_notify=should_notify,
            score=None,
            reason=None if should_notify else "no-notify",
            details=details,
        )

    def get_state(self) -> Dict[str, object]:
        try:
            agent = self._agent
            cfg = agent.config
            return {
                "epsilon": float(agent.epsilon),
                "total_updates": int(agent.total_updates),
                "weights": {
                    "action_0": list(agent.weights.get(0, [])),
                    "action_1": list(agent.weights.get(1, [])),
                },
                "rewards": {
                    "sum": {"0": float(agent.action_reward_sum.get(0, 0.0)), "1": float(agent.action_reward_sum.get(1, 0.0))},
                    "count": {"0": int(agent.action_reward_count.get(0, 0)), "1": int(agent.action_reward_count.get(1, 0))},
                    "last": {"0": agent.action_last_reward.get(0), "1": agent.action_last_reward.get(1)},
                },
                "config": {
                    "epsilon_start": float(cfg.epsilon_start),
                    "epsilon_min": float(cfg.epsilon_min),
                    "epsilon_decay": float(cfg.epsilon_decay),
                    "learning_rate": float(cfg.learning_rate),
                    "reward_window_seconds": int(cfg.reward_window_seconds),
                    "decision_interval_seconds": int(cfg.decision_interval_seconds),
                    "improvement_ratio_threshold": float(cfg.improvement_ratio_threshold),
                    "state_file_path": str(cfg.state_file_path),
                },
                "cooldown_seconds": 0,  # cooldown moved out of notification layer
                "last_notified_at_ms": int(self._last_notified_at_ms),
            }
        except Exception:
            return {"error": "unavailable"}

    # ---------- Internals ----------
    def _record_features(self, features: Optional[Dict[str, float]], now_ms: int) -> None:
        if not features:
            return
        try:
            self._recent_features.append({"t": now_ms, "features": features})
            if len(self._recent_features) > 500:
                self._recent_features = self._recent_features[-300:]
        except Exception:
            pass

    def _latest_features_at_or_after(self, ts_ms: int) -> Optional[Dict[str, float]]:
        try:
            chosen = None
            for item in self._recent_features:
                t = int(item.get("t", 0))  # type: ignore[arg-type]
                if t >= ts_ms:
                    chosen = item
                    break
            if chosen is None and self._recent_features:
                chosen = self._recent_features[-1]
            if chosen is not None:
                f = chosen.get("features")
                if isinstance(f, dict):
                    return f  # type: ignore[return-value]
        except Exception:
            pass
        return None

    async def _delayed_evaluate(self, deadline_ms: int) -> None:
        try:
            if not self._training_enabled:
                return
            delay = max(0.0, (deadline_ms - int(time.time() * 1000)) / 1000.0)
            if delay > 0:
                await asyncio.sleep(delay)
            if not self._pending_eval:
                return
            features_after = self._latest_features_at_or_after(deadline_ms)
            if not features_after:
                reward = -1.0
            else:
                baseline = float(self._pending_eval.get("baseline_badness", 0.0))  # type: ignore[arg-type]
                current = self._compute_badness(features_after)
                improved = False
                if baseline > 1e-6:
                    reduction = (baseline - current) / max(baseline, 1e-6)
                    improved = reduction >= self._agent.config.improvement_ratio_threshold
                else:
                    improved = current <= baseline + 0.05
                reward = 1.0 if improved else -1.0
            self._agent.learn(list(self._pending_eval.get("state_vector", [])), 1, reward)  # type: ignore[arg-type]
        except Exception:
            pass
        finally:
            self._pending_eval = None

    def _compute_badness(self, features: Dict[str, float]) -> float:
        try:
            s = abs(float(features.get("shoulder_line_angle_deg", 0.0)))
            h_val = features.get("head_tilt_deg")
            h = 0.0 if h_val is None else abs(float(h_val))
            r = float(features.get("head_to_shoulder_distance_ratio", 0.0))
        except Exception:
            return 0.0
        cfg = self._agent.config
        bs = s / max(cfg.scale_shoulder_angle_deg, 1e-6)
        bt = h / max(cfg.scale_head_tilt_deg, 1e-6)
        br = max(0.0, (cfg.scale_head_drop_ratio - r) / max(cfg.scale_head_drop_ratio, 1e-6))
        return max(bs, bt, br)

    def _build_state_vector(self, features: Dict[str, float], now_ms: int) -> List[float]:
        try:
            s = abs(float(features.get("shoulder_line_angle_deg", 0.0)))
            hv = features.get("head_tilt_deg")
            h = 0.0 if hv is None else abs(float(hv))
            r = float(features.get("head_to_shoulder_distance_ratio", 0.0))
        except Exception:
            s, h, r = 0.0, 0.0, 0.0
        cfg = self._agent.config
        return [
            s / max(cfg.scale_shoulder_angle_deg, 1e-6),
            h / max(cfg.scale_head_tilt_deg, 1e-6),
            r / max(cfg.scale_head_drop_ratio, 1e-6),
            1.0,
        ]


