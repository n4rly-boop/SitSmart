from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
import asyncio
from typing import Dict, Optional

from app.api.schemas import Notification, NotificationSeverity, ModelAnalysisResponse
from app.config import get_config
from app.services.rl_service import ThresholdLinUCBAgent
from app.services.history_service import HistoryService


_CONFIG = get_config()


@dataclass
class NotificationOptions:
    cooldown_seconds: int = int(_CONFIG.effective_cooldown_seconds)
    webhook_url: Optional[str] = None  # Default provided at runtime
    analyze_base_url: Optional[str] = None  # Base URL for model analyze routes
    ml_bad_prob_threshold: float = float(_CONFIG.ml_bad_prob_threshold)

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

    def maybe_notify_from_ml_response(self, response: ModelAnalysisResponse, f1_features: Optional[Dict[str, float]] = None) -> bool:
        now_ms = int(time.time() * 1000)
        bad_prob = float(response.bad_posture_prob or 0.0)

        agent = ThresholdLinUCBAgent.get_instance()
        try:
            decision = agent.decide(bad_prob, now_seconds=now_ms / 1000.0)
        except Exception:
            # RL unavailable – use static policy
            threshold = float(self.options.ml_bad_prob_threshold)
            should_notify = bad_prob >= threshold and self._can_notify(now_ms)
            if not should_notify:
                return False
            notif = self.build_notification()
            self.send_via_webhook(notif)
            self._mark_notified(now_ms)
            try:
                HistoryService.get_instance().on_notification(bad_prob, threshold, now_ms, f1_features=f1_features)
            except Exception:
                pass
            return True

        rl_threshold = float(decision["new_threshold"])
        should_notify = bool(decision["notify"]) and self._can_notify(now_ms)

        try:
            agent.commit_decision(decision["tick_id"], sent=should_notify, timestamp_ms=now_ms)
        except Exception:
            should_notify = False

        if not should_notify:
            return False

        notif = self.build_notification()
        self.send_via_webhook(notif)
        self._mark_notified(now_ms)

        try:
            HistoryService.get_instance().on_notification(bad_prob, rl_threshold, now_ms, f1_features=f1_features)
        except Exception:
            pass
        return True
