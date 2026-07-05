from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

DEFAULT_INCLUDE = ("*.mp4", "*.mkv", "*.mov", "*.webm")


class SegmentationConfig(BaseModel):
    segment_seconds: float = Field(default=30.0, gt=0.0)
    overlap_seconds: float = Field(default=0.0, ge=0.0)
    merge_short_tail: bool = True
    min_tail_seconds: float = Field(default=5.0, ge=0.0)
    sentence_snap_tolerance_seconds: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _validate_step(self) -> SegmentationConfig:
        step = self.segment_seconds - self.overlap_seconds
        if step <= 0:
            raise ValueError(
                f"step must be > 0; segment_seconds ({self.segment_seconds}) - "
                f"overlap_seconds ({self.overlap_seconds}) = {step}"
            )
        return self


class FrameSamplingConfig(BaseModel):
    position: float = 0.5
    jpeg_quality: int = Field(default=85, ge=1, le=100)
    max_long_edge: int = Field(default=512, gt=0)

    @field_validator("position", mode="before")
    @classmethod
    def _clamp_position(cls, value: float) -> float:
        v = float(value)
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v


class Config(BaseModel):
    """Top-level pipeline configuration.

    Bundles segmentation and frame-sampling knobs with the model identifiers,
    device, IO paths, and filter globs the CLI needs. The CLI constructs this
    directly from its flags; the field defaults below are the fallback.
    """

    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    frames: FrameSamplingConfig = Field(default_factory=FrameSamplingConfig)

    whisper_model: str = "small.en"
    text_embed_model: str = "intfloat/multilingual-e5-large-instruct"
    vision_embed_model: str = "google/siglip2-so400m-patch14-384"

    device: str = "auto"
    db_path: Path = Path("./video-lance.db")

    include: tuple[str, ...] = DEFAULT_INCLUDE
    exclude: tuple[str, ...] = ()

    workers: int = Field(default=1, ge=1)
