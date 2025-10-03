from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
import asyncio
from typing import Dict, Optional

from app.api.schemas import Notification, NotificationSeverity, ModelAnalysisRequest, ModelAnalysisResponse, FeatureVector
from app.services.rl_service import RLService


@dataclass
class NotificationOptions:
    cooldown_seconds: int = max(int(os.getenv("FEATURE_BUFFER_SECONDS", "5")), int(os.getenv("NOTIFICATION_COOLDOWN_SECONDS", "5")))
    webhook_url: Optional[str] = None  # Default provided at runtime
    analyze_base_url: Optional[str] = None  # Base URL for model analyze routes

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

    def get_rl_state(self) -> Dict[str, object]:
        # Proxy RL state to RL service
        return RLService.get_instance().get_state()

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


    def decide_and_notify(self, features: Optional[Dict[str, float]], method: str = "rl") -> bool:
        """Call either RL or future ML service to get decision, then send webhook if allowed by cooldown.

        method: "rl" or "ml"
        """
        if not features:
            return False
        try:
            fv = FeatureVector(**features)  # type: ignore[arg-type]
        except Exception:
            return False

        if method == "rl":
            # Enable RL learning when RL method is active
            RLService.get_instance().set_training_enabled(True)
            response = RLService.get_instance().analyze(ModelAnalysisRequest(features=fv))
        else:
            # Pause RL training while ML drives decisions
            RLService.get_instance().set_training_enabled(False)
            # Placeholder ML path with same contract (stub decision false)
            from app.services.ml_service import MLService
            response = MLService.get_instance().analyze(
                ModelAnalysisRequest(features=FeatureVector(**features))
            )

        if not isinstance(response, ModelAnalysisResponse):
            return False

        now_ms = int(time.time() * 1000)
        if not response.should_notify:
            return False
        if not self._can_notify(now_ms):
            return False

        notif = self.build_notification(features)
        self.send_via_webhook(notif)
        self._mark_notified(now_ms)
        return True