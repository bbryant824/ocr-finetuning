"""GPU-only Qwen implementation boundary.

This module stays separate because it runs remotely and will eventually own
Transformers, PEFT, CUDA, image tiling, prediction parsing, and LoRA training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_output(raw: str) -> dict[str, Any]:
    """Parse the structured OCR response expected from the model prompt."""

    try:
        output = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Qwen output is not valid JSON") from exc
    if not isinstance(output, dict) or not isinstance(output.get("regions"), list):
        raise ValueError("Qwen output must contain a regions list")
    return output


class QwenRunner:
    """Placeholder for GPU model loading, prediction, and LoRA training."""

    def __init__(self, model_name: str, output_root: Path) -> None:
        self.model_name = model_name
        self.output_root = output_root

    def load(self) -> None:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("install the gpu dependencies to load Qwen") from exc
        raise NotImplementedError("Qwen loading will be implemented with the OCR model work")

    def train(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Qwen fine-tuning is not implemented yet")

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Qwen prediction is not implemented yet")

    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Qwen evaluation is not implemented yet")
