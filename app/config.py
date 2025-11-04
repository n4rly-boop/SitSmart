

import os
from dataclasses import dataclass
from typing import Optional


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _get_str(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        value = os.getenv(name)
        if value is None:
            return default
        return value
    except Exception:
        return default


@dataclass
class AppConfig:
    """Central application configuration (excludes RL service settings)."""

    # Feature aggregation window in seconds
    feature_buffer_seconds: int = _get_int("FEATURE_BUFFER_SECONDS", 5)

    # Notification cooldown in seconds
    notification_cooldown_seconds: int = _get_int("NOTIFICATION_COOLDOWN_SECONDS", 8)

    # ML probability threshold to trigger notification
    ml_bad_prob_threshold: float = _get_float("ML_BAD_PROB_THRESHOLD", 0.6)

    # Notification delivery webhook
    notification_webhook_url: str = _get_str(
        "NOTIFICATION_WEBHOOK_URL", "http://127.0.0.1:8000/api/notifications/webhook"
    ) or "http://127.0.0.1:8000/api/notifications/webhook"

    # Optional: external analyze service base URL (unused by default)
    notification_analyze_base_url: Optional[str] = _get_str("NOTIFICATION_ANALYZE_BASE_URL")

    @property
    def effective_cooldown_seconds(self) -> int:
        """Effective cooldown equals max(buffer window, explicit cooldown)."""
        return max(int(self.feature_buffer_seconds), int(self.notification_cooldown_seconds))


# Singleton-style accessor
_CONFIG = AppConfig()


def get_config() -> AppConfig:
    return _CONFIG


