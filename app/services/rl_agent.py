from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict
from typing import Dict, List, Optional

from app.services.rl_config import RLConfig


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
            self.moving_avg_reward = 0.0
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
