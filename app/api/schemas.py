from typing import Any, Dict, Optional
from enum import Enum
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class AnalyzeResponse(BaseModel):
    label: str = Field(description="Posture label, e.g., 'good' or 'bad' or 'unknown'")
    score: float = Field(ge=0.0, le=1.0, description="Confidence/quality score in [0,1]")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional details like angles, keypoints, thresholds"
    )


class FeatureVector(BaseModel):
    shoulder_line_angle_deg: float = Field(description="Angle of shoulder line vs horizontal; + means left shoulder higher")
    head_tilt_deg: Optional[float] = Field(
        default=None,
        description="Head roll angle from ear-to-ear line vs horizontal; may be None if unavailable",
    )
    head_to_shoulder_distance_px: float = Field(description="Perpendicular distance from head center to shoulder line in pixels")
    head_to_shoulder_distance_ratio: float = Field(
        description="Head-to-shoulder distance normalized by shoulder width"
    )
    shoulder_width_px: float = Field(description="Pixel distance between shoulders")


class FeatureExtractionResponse(BaseModel):
    features: Optional[FeatureVector] = Field(default=None, description="Computed posture features")
    landmarks: Optional[list[list[float]]] = Field(
        default=None, description="Normalized landmarks [x,y] in [0,1] from MediaPipe Pose"
    )
    reason: Optional[str] = Field(default=None, description="Reason when features are None")


class NotificationSeverity(str, Enum):
    info = "info"
    warning = "warning"
    critical = "critical"


class Notification(BaseModel):
    title: str = Field(description="Short title for the notification")
    message: str = Field(description="User-facing message prompting posture correction")
    severity: NotificationSeverity = Field(description="Severity level for UI styling/priority")
    suggested_action: Optional[str] = Field(default=None, description="Optional suggested action to resolve the alert")
    ttl_seconds: Optional[int] = Field(default=None, description="Optional display duration in seconds for clients")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="UTC timestamp when the notification was generated")
