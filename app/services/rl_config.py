from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
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
    reward_window_seconds: int = 3
    # Minimum seconds between agent decisions (throttling)
    decision_interval_seconds: int = 1

    # Considered a success if badness reduces by this fraction OR falls under thresholds
    improvement_ratio_threshold: float = 0.3

    # Feature scaling to keep values in a reasonable range for linear models
    scale_shoulder_angle_deg: float = 30.0
    scale_head_tilt_deg: float = 30.0
    scale_head_drop_ratio: float = 1.0
    scale_time_since_notify_s: float = 60.0
    scale_moving_avg_reward: float = 1.0

    # Persistence
    state_file_path: str = os.getenv(
        "RL_AGENT_STATE_PATH",
        os.path.join(os.getcwd(), "train", "rl_agent_state.json"),
    )


