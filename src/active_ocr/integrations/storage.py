"""Minimal SQLite state store plus content-addressed local artifacts."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

Record = TypeVar("Record", bound=BaseModel)
SAFE_NAME = re.compile(r"^[a-zA-Z0-9_-]+$")


class SQLiteStore:
    """Persist validated models in one small, flexible SQLite table."""

    def __init__(self, database_path: Path, artifact_root: Path) -> None:
        self.database_path = database_path
        self.artifact_root = artifact_root

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS records ("
                "kind TEXT NOT NULL, key TEXT NOT NULL, payload TEXT NOT NULL, "
                "PRIMARY KEY (kind, key))"
            )

    def save(self, kind: str, key: str, value: BaseModel) -> None:
        """Insert or replace a Pydantic model as JSON."""

        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "INSERT INTO records(kind, key, payload) VALUES (?, ?, ?) "
                "ON CONFLICT(kind, key) DO UPDATE SET payload=excluded.payload",
                (kind, key, value.model_dump_json()),
            )

    def load(self, kind: str, key: str, model: type[Record]) -> Record | None:
        """Load one record and validate it against the requested model type."""

        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT payload FROM records WHERE kind=? AND key=?", (kind, key)
            ).fetchone()
        return None if row is None else model.model_validate_json(row[0])

    def list(self, kind: str, model: type[Record]) -> tuple[Record, ...]:
        """Load all records of one kind in stable key order."""

        with sqlite3.connect(self.database_path) as connection:
            rows = connection.execute(
                "SELECT payload FROM records WHERE kind=? ORDER BY key", (kind,)
            ).fetchall()
        return tuple(model.model_validate_json(row[0]) for row in rows)

    def save_artifact(self, namespace: str, data: bytes, suffix: str = "") -> Path:
        """Store immutable bytes by SHA-256 and return their local path."""

        if not SAFE_NAME.fullmatch(namespace):
            raise ValueError("unsafe artifact namespace")
        if suffix and (not suffix.startswith(".") or not suffix[1:].isalnum()):
            raise ValueError("artifact suffix must be a simple extension")
        digest = hashlib.sha256(data).hexdigest()
        path = self.artifact_root / namespace / digest[:2] / f"{digest}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        return path
