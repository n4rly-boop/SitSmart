from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
import asyncio
from typing import Callable, Dict, List, Optional

from app.api.schemas import Notification, NotificationSeverity


ConditionFn = Callable[[Dict[str, float]], bool]


@dataclass
class NotificationOptions:
    cooldown_seconds: int = 5
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

    def maybe_notify(self, features: Optional[Dict[str, float]]) -> bool:
        now_ms = int(time.time() * 1000)
        if not self._can_notify(now_ms):
            return False
        if not self.evaluate(features):
            return False
        notif = self.build_notification(features or {})
        self.send_via_webhook(notif)
        self._mark_notified(now_ms)
        print("notification sent")
        return True

