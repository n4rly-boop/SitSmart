from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
import asyncio
from typing import Callable, Dict, List, Optional

from app.api.schemas import Notification, NotificationSeverity
from app.services.rl_agent import PostureRLAgent


ConditionFn = Callable[[Dict[str, float]], bool]


@dataclass
class NotificationOptions:
    cooldown_seconds: int = 4
    webhook_url: Optional[str] = None  # Default provided at runtime
    # Default thresholds; scalable for future custom conditions
    max_shoulder_angle_abs_deg: float = 8.0
    max_head_tilt_abs_deg: float = 10.0
    max_head_drop_ratio: float = 0.5


class NotificationService:
    _instance: Optional["NotificationService"] = None

    def __init__(self, options: Optional[NotificationOptions] = None, conditions: Optional[List[ConditionFn]] = None):
        self.options = options or NotificationOptions()
        # Compose default condition with any provided custom conditions
        default_condition = self._default_condition
        self.conditions: List[ConditionFn] = [default_condition]
        if conditions:
            self.conditions.extend(conditions)
        self._last_notified_at_ms: int = 0
        self._last_decision_at_ms: int = 0
        # RL agent and pending reward eval state
        self._agent = PostureRLAgent()
        self._pending_eval: Optional[Dict[str, object]] = None
        self._eval_seq: int = 0
        self._recent_features: List[Dict[str, object]] = []  # list of {"t": ms, "features": dict}

        if not self.options.webhook_url:
            # Default to internal webhook endpoint; can be overridden via env
            base = os.getenv("NOTIFICATION_WEBHOOK_URL", "http://127.0.0.1:8000/api/notifications/webhook")
            self.options.webhook_url = base

    @classmethod
    def get_instance(cls) -> "NotificationService":
        if cls._instance is None:
            cls._instance = NotificationService()
        return cls._instance

    def _default_condition(self, features: Dict[str, float]) -> bool:
        try:
            s = float(features.get("shoulder_line_angle_deg", 0.0))
            h_val = features.get("head_tilt_deg")
            h = None if h_val is None else float(h_val)
            r = float(features.get("head_to_shoulder_distance_ratio", 0.0))
        except Exception:
            return False

        if abs(s) > self.options.max_shoulder_angle_abs_deg:
            return True
        if h is not None and abs(h) > self.options.max_head_tilt_abs_deg:
            return True
        if r < self.options.max_head_drop_ratio:
            return True
        return False

    def _can_notify(self, now_ms: int) -> bool:
        cooldown_ms = max(0, int(self.options.cooldown_seconds) * 1000)
        return (now_ms - self._last_notified_at_ms) >= cooldown_ms

    def _mark_notified(self, now_ms: int) -> None:
        self._last_notified_at_ms = now_ms

    def evaluate(self, features: Optional[Dict[str, float]]) -> bool:
        if not features:
            return False
        for cond in self.conditions:
            try:
                if cond(features):
                    return True
            except Exception:
                # Ignore faulty condition
                continue
        return False

    def _compute_badness(self, features: Dict[str, float]) -> float:
        try:
            s = abs(float(features.get("shoulder_line_angle_deg", 0.0)))
            h = abs(float(features.get("head_tilt_deg", 0.0)))
            r = float(features.get("head_to_shoulder_distance_ratio", 0.0))
        except Exception:
            return 0.0

        # Threshold-free deviation using configurable scales
        cfg = self._agent.config
        bs = s / max(cfg.scale_shoulder_angle_deg, 1e-6)
        bt = h / max(cfg.scale_head_tilt_deg, 1e-6)
        # For head drop, smaller ratio implies worse; normalize by scale_head_drop_ratio
        br = max(0.0, (cfg.scale_head_drop_ratio - r) / max(cfg.scale_head_drop_ratio, 1e-6))
        return max(bs, bt, br)

    def _build_state_vector(self, features: Dict[str, float], now_ms: int) -> List[float]:
        # Scaled raw magnitudes
        try:
            s = abs(float(features.get("shoulder_line_angle_deg", 0.0)))
            h= abs(float(features.get("head_tilt_deg", 0.0)))
            r = float(features.get("head_to_shoulder_distance_ratio", 0.0))
        except Exception:
            s, h, r = 0.0, 0.0, 0.0

        cfg = self._agent.config
        t_since_s = max(0.0, (now_ms - self._last_notified_at_ms) / 1000.0)
        # Features
        x = [
            s / max(cfg.scale_shoulder_angle_deg, 1e-6),
            h / max(cfg.scale_head_tilt_deg, 1e-6),
            r / max(cfg.scale_head_drop_ratio, 1e-6),
            1.0,  # bias
        ]
        return x

    def _record_features(self, features: Optional[Dict[str, float]], now_ms: int) -> None:
        if not features:
            return
        try:
            self._recent_features.append({"t": now_ms, "features": features})
            # Cap buffer size
            if len(self._recent_features) > 500:
                self._recent_features = self._recent_features[-300:]
        except Exception:
            pass

    def _latest_features_at_or_after(self, ts_ms: int) -> Optional[Dict[str, float]]:
        try:
            # Find the first sample with t >= ts_ms; if none, return latest sample
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

    def get_rl_state(self) -> Dict[str, object]:
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
                "cooldown_seconds": int(self.options.cooldown_seconds),
                "last_notified_at_ms": int(self._last_notified_at_ms),
            }
        except Exception:
            return {
                "error": "unavailable",
            }

    def build_notification(self, features: Dict[str, float]) -> Notification:
        suggested = "Sit upright, relax shoulders, and keep screen at eye level."
        return Notification(
            title="Posture Check",
            message="Please straighten your back and align your head with your shoulders.",
            severity=NotificationSeverity.warning,
            suggested_action=suggested,
            ttl_seconds=10,
        )

    def send_via_webhook(self, notification: Notification) -> None:
        if not self.options.webhook_url:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(asyncio.to_thread(self._send_webhook_sync, notification))
        except RuntimeError:
            # No running loop (e.g., in tests) – send synchronously
            self._send_webhook_sync(notification)
        except Exception:
            pass

    def _send_webhook_sync(self, notification: Notification) -> None:
        try:
            data = notification.model_dump_json().encode("utf-8")
            req = urllib.request.Request(
                self.options.webhook_url, data=data, headers={"Content-Type": "application/json"}, method="POST"
            )
            urllib.request.urlopen(req, timeout=3.0)  # nosec - internal or trusted URL
        except Exception:
            pass

    def _try_finish_pending_evaluation(self, features: Optional[Dict[str, float]], now_ms: int) -> None:
        # Disabled: we now update strictly once per notification via the delayed task
        return

    async def _delayed_evaluate(self, eval_id: int, notified_at_ms: int, deadline_ms: int, baseline_badness: float, state_vector: List[float]) -> None:
        try:
            # Sleep until deadline; if event loop isn't running, this will raise and we skip
            delay = max(0.0, (deadline_ms - int(time.time() * 1000)) / 1000.0)
            if delay > 0:
                await asyncio.sleep(delay)
            # If another path already handled this evaluation, skip
            if not self._pending_eval or int(self._pending_eval.get("id", -1)) != eval_id:
                return
            # Use latest features at or after deadline; if none, treat as failure
            features_after = self._latest_features_at_or_after(deadline_ms)
            if not features_after:
                reward = -1.0
            else:
                current_badness = self._compute_badness(features_after)
                # Reward definition without hard thresholds: +1 if badness dropped sufficiently; else -1
                improved = False
                if baseline_badness > 1e-6:
                    reduction = (baseline_badness - current_badness) / max(baseline_badness, 1e-6)
                    if reduction >= self._agent.config.improvement_ratio_threshold:
                        improved = True
                else:
                    # Baseline near-zero; reward only if still near-zero (no regression)
                    improved = current_badness <= baseline_badness + 0.05
                reward = 1.0 if improved else -1.0
            self._agent.learn(state_vector, 1, reward)
        except Exception:
            pass
        finally:
            self._pending_eval = None

    def maybe_notify(self, features: Optional[Dict[str, float]]) -> bool:
        now_ms = int(time.time() * 1000)
        # Record incoming features for reward evaluation buffer
        try:
            self._record_features(features, now_ms)
        except Exception:
            pass
        # Check if we can complete any pending reward evaluation
        try:
            self._try_finish_pending_evaluation(features, now_ms)
        except Exception:
            pass

        # Remove hardcoded-threshold gate: agent learns when to notify per-user
        if not features:
            return False

        # RL agent decides whether to notify; throttle decision frequency
        state_vector = self._build_state_vector(features, now_ms)
        # throttle by decision interval
        interval_ms = max(0, int(self._agent.config.decision_interval_seconds) * 1000)
        if (now_ms - self._last_decision_at_ms) < interval_ms:
            return False
        if not self._can_notify(now_ms):
            return False

        action = self._agent.select_action(state_vector)
        self._last_decision_at_ms = now_ms
        if action != 1:
            return False

        if not self._can_notify(now_ms):
            return False

        notif = self.build_notification(features)
        self.send_via_webhook(notif)
        self._mark_notified(now_ms)

        # Schedule reward evaluation in config-defined window
        reward_window_ms = max(0, int(self._agent.config.reward_window_seconds) * 1000)
        self._eval_seq += 1
        eval_id = self._eval_seq
        deadline_ms = now_ms + reward_window_ms
        self._pending_eval = {
            "id": eval_id,
            "notified_at_ms": now_ms,
            "deadline_ms": deadline_ms,
            "baseline_badness": self._compute_badness(features),
            "baseline_is_bad": True,  # gating removed; treat baseline as "needs action" when we notify
            "state_vector": state_vector,
        }
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._delayed_evaluate(eval_id, now_ms, deadline_ms, self._pending_eval["baseline_badness"], state_vector))
        except Exception:
            # If no loop or scheduling fails, fallback to timer via threading
            try:
                import threading
                delay_s = max(0.0, (deadline_ms - int(time.time() * 1000)) / 1000.0)
                def _thread_eval():
                    try:
                        asyncio.run(self._delayed_evaluate(eval_id, now_ms, deadline_ms, self._pending_eval["baseline_badness"], state_vector))
                    except Exception:
                        pass
                t = threading.Timer(delay_s, _thread_eval)
                t.daemon = True
                t.start()
            except Exception:
                pass
        try:
            print("notification sent (RL agent)")
        except Exception:
            pass
        return True

