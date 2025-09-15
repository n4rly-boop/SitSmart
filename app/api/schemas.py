from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AnalyzeResponse(BaseModel):
    label: str = Field(description="Posture label, e.g., 'good' or 'bad' or 'unknown'")
    score: float = Field(ge=0.0, le=1.0, description="Confidence/quality score in [0,1]")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional details like angles, keypoints, thresholds"
    )
