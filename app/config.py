

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
class RLConfig:
    """Reinforcement Learning service configuration.
    
    Core RL Parameters:
    - eta: Step size for threshold adjustments (actions: -eta, 0, +eta). 
      Larger = faster adaptation but more jitter.
    
    Threshold Bounds:
    - tau_min, tau_max: Hard limits for notification threshold. Prevents extreme values.
    - initial_threshold: Starting threshold value. Should match user's initial ML threshold setting.
    
    Band-Based Boundaries (Delta Band):
    - band_q_low/high: Target quantiles for adaptive band boundaries. 
      Low delta < L → raise threshold, delta > H → lower threshold.
    - band_quantile_lr: Learning rate for online quantile estimation (Robbins-Monro). 
      Lower = slower adaptation.
    """

    # Core RL: step size for threshold adjustments
    eta: float = _get_float("RL_THRESHOLD_STEP", 0.03)
    
    # Threshold bounds: hard limits for notification threshold
    tau_min: float = _get_float("RL_TAU_MIN", 0.5)
    tau_max: float = _get_float("RL_TAU_MAX", 0.95)
    initial_threshold: float = _get_float("ML_BAD_PROB_THRESHOLD", 0.6)
    
    # Band boundaries: target quantiles for adaptive band boundaries
    band_q_low: float = _get_float("RL_BAND_Q_LOW", 0.3)
    band_q_high: float = _get_float("RL_BAND_Q_HIGH", 0.6)
    band_quantile_lr: float = _get_float("RL_BAND_Q_LR", 0.03)


@dataclass
class HistoryConfig:
    """History service configuration for RL feature tracking."""

    # Delta computation window in seconds (time to wait before computing delta)
    delta_range_seconds: float = _get_float("RL_DELTA_RANGE_SECONDS", 8.0)


@dataclass
class AppConfig:
    """Central application configuration."""

    # Feature aggregation window in seconds
    feature_buffer_seconds: int = _get_int("FEATURE_BUFFER_SECONDS", 12)

    # Notification cooldown in seconds
    notification_cooldown_seconds: int = _get_int("NOTIFICATION_COOLDOWN_SECONDS", 15)

    # ML probability threshold to trigger notification
    ml_bad_prob_threshold: float = _get_float("ML_BAD_PROB_THRESHOLD", 0.6)

    # Notification delivery webhook URL
    notification_webhook_url: str = _get_str(
        "NOTIFICATION_WEBHOOK_URL", "http://127.0.0.1:8000/api/notifications/webhook"
    ) or "http://127.0.0.1:8000/api/notifications/webhook"

    # Optional: external analyze service base URL (unused by default)
    notification_analyze_base_url: Optional[str] = _get_str("NOTIFICATION_ANALYZE_BASE_URL")

    @property
    def effective_cooldown_seconds(self) -> int:
        """Effective cooldown equals max(buffer window, explicit cooldown)."""
        return max(int(self.feature_buffer_seconds), int(self.notification_cooldown_seconds))


# Singleton-style accessors
_CONFIG = AppConfig()
_RL_CONFIG = RLConfig()
_HISTORY_CONFIG = HistoryConfig()


def get_config() -> AppConfig:
    return _CONFIG


def get_rl_config() -> RLConfig:
    return _RL_CONFIG


def get_history_config() -> HistoryConfig:
    return _HISTORY_CONFIG


