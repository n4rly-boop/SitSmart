from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
import asyncio
from typing import Dict, Optional

from app.api.schemas import Notification, NotificationSeverity, ModelAnalysisResponse


@dataclass
class NotificationOptions:
    cooldown_seconds: int = max(int(os.getenv("FEATURE_BUFFER_SECONDS", "5")), int(os.getenv("NOTIFICATION_COOLDOWN_SECONDS", "5")))
    webhook_url: Optional[str] = None  # Default provided at runtime
    analyze_base_url: Optional[str] = None  # Base URL for model analyze routes
    ml_bad_prob_threshold: float = float(os.getenv("ML_BAD_PROB_THRESHOLD", "0.6"))

class NotificationService:
    _instance: Optional["NotificationService"] = None

    def __init__(self, options: Optional[NotificationOptions] = None):
        self.options = options or NotificationOptions()
        self._last_notified_at_ms: int = 0
        if not self.options.webhook_url:
            base = os.getenv("NOTIFICATION_WEBHOOK_URL", "http://127.0.0.1:8000/api/notifications/webhook")
            self.options.webhook_url = base

    @classmethod
    def get_instance(cls) -> "NotificationService":
        if cls._instance is None:
            cls._instance = NotificationService()
        return cls._instance

    def _can_notify(self, now_ms: int) -> bool:
        cooldown_ms = max(0, int(self.options.cooldown_seconds) * 1000)
        return (now_ms - self._last_notified_at_ms) >= cooldown_ms

    def _mark_notified(self, now_ms: int) -> None:
        self._last_notified_at_ms = now_ms

    # RL is intentionally not exposed here anymore

    def build_notification(self) -> Notification:
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

    def maybe_notify_from_ml_response(self, response: ModelAnalysisResponse) -> bool:
        now_ms = int(time.time() * 1000)
        bad_prob = response.bad_posture_prob or 0.0
        if bad_prob < float(self.options.ml_bad_prob_threshold):
            return False
        if not self._can_notify(now_ms):
            return False
        notif = self.build_notification()
        self.send_via_webhook(notif)
        self._mark_notified(now_ms)
        return True