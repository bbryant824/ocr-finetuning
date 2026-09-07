"""Small validated configuration for the local and remote processes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Self

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LabelStudioSettings(ConfigModel):
    url: str = "http://localhost:8080"
    project_id: int | None = None
    token: str | None = None


class GPUSettings(ConfigModel):
    url: str = "http://localhost:9000"
    token: str | None = None
    model: str = "Qwen/Qwen3-VL-4B-Instruct"


class ExperimentSettings(ConfigModel):
    batch_size: int = Field(default=50, gt=0)
    rounds: int = Field(default=6, gt=0)
    seed: int = 824
    train_ratio: float = Field(default=0.7, ge=0, le=1)
    validation_ratio: float = Field(default=0.1, ge=0, le=1)
    test_ratio: float = Field(default=0.2, ge=0, le=1)

    @model_validator(mode="after")
    def ratios_sum_to_one(self) -> Self:
        if abs(self.train_ratio + self.validation_ratio + self.test_ratio - 1) > 1e-9:
            raise ValueError("data split ratios must sum to 1")
        return self


class Settings(ConfigModel):
    database: Path = Path("var/state.sqlite3")
    artifacts: Path = Path("var/artifacts")
    documents: Path = Path("var/documents")
    label_studio: LabelStudioSettings = LabelStudioSettings()
    gpu: GPUSettings = GPUSettings()
    experiment: ExperimentSettings = ExperimentSettings()
    training: dict[str, Any] = Field(
        default_factory=lambda: {
            "epochs": 3,
            "batch_size": 1,
            "gradient_accumulation": 16,
            "bf16": True,
            "freeze_vision_encoder": True,
        }
    )
    poll_seconds: int = Field(default=30, gt=0)


def load_settings(path: Path | None = None) -> Settings:
    """Load YAML and override secrets from environment variables."""

    load_dotenv()
    config_path = path or Path(os.getenv("ACTIVE_OCR_CONFIG", "config/default.yaml"))
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")

    label_token = os.getenv("ACTIVE_OCR_LABEL_STUDIO_TOKEN")
    gpu_token = os.getenv("ACTIVE_OCR_GPU_TOKEN")
    if label_token:
        raw.setdefault("label_studio", {})["token"] = label_token
    if gpu_token:
        raw.setdefault("gpu", {})["token"] = gpu_token
    return Settings.model_validate(raw)
