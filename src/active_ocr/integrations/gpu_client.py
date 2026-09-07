"""Lightweight HTTP client for the remote Qwen process."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx


class GPUClient:
    """Submit asynchronous jobs without importing any machine-learning packages."""

    def __init__(self, base_url: str, token: str | None) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token

    @property
    def headers(self) -> dict[str, str]:
        return {} if not self.token else {"Authorization": f"Bearer {self.token}"}

    def submit(self, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if kind not in {"train", "predict", "evaluate"}:
            raise ValueError(f"unsupported GPU job: {kind}")
        response = httpx.post(
            f"{self.base_url}/jobs/{kind}", headers=self.headers, json=payload, timeout=60
        )
        response.raise_for_status()
        return dict(response.json())

    def job(self, job_id: str) -> dict[str, Any]:
        response = httpx.get(f"{self.base_url}/jobs/{job_id}", headers=self.headers, timeout=30)
        response.raise_for_status()
        return dict(response.json())
