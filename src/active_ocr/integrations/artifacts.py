"""Native-file inventory, hashes and environment provenance shared by adapter and transport.

Deliberately model agnostic and dependency light: the local coordinator, the CLI and the GPU
worker all import this module, so nothing here may require Torch or LLaMA-Factory.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from importlib.metadata import distribution, distributions
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from active_ocr.models import SHA256, Model


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(text: str | bytes) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def safe_path(root: Path, path: str | Path) -> Path:
    """Lexical containment AND no symlink in any component, including ancestors."""
    root = Path(os.path.abspath(root))
    candidate = Path(path)
    candidate = candidate if candidate.is_absolute() else root / candidate
    if ".." in candidate.parts:
        raise ValueError("parent traversal")
    candidate = Path(os.path.abspath(candidate))
    if not candidate.is_relative_to(root):
        raise ValueError("path escapes root")
    if any(p.is_symlink() for p in (candidate, *candidate.parents)):
        raise ValueError("symlink path")
    return candidate


class FileEntry(Model):
    filename: str = Field(min_length=1)
    bytes: int = Field(ge=0, strict=True)
    sha256: SHA256

    @field_validator("filename")
    @classmethod
    def relative_filename(cls, value: str) -> str:
        if (
            Path(value).is_absolute()
            or ".." in Path(value).parts
            or "\\" in value
            or Path(value).as_posix() != value
            or value in ("", ".")
        ):
            raise ValueError("inventory requires normalized relative filenames")
        return value


class Packages(Model):
    python: str
    packages: tuple[tuple[str, str], ...]


class WorkerManifest(Model):
    schema_version: Literal[1]
    source_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    files: tuple[FileEntry, ...]
    environment: Packages
    build_spec_sha256: SHA256 | None = None
    deployment_reference: str | None = None


def installed_packages() -> Packages:
    # Provider bootstrap paths can expose shadowed copies. Resolve each name as
    # Python metadata does, rather than choosing by enumeration order or version.
    names = {
        re.sub(r"[-_.]+", "-", d.metadata["Name"].lower())
        for d in distributions()
        if d.metadata["Name"]
    }
    return Packages(
        python=platform.python_version(),
        packages=tuple((name, distribution(name).version) for name in sorted(names)),
    )


def inventory(root: Path, names) -> list[dict]:
    return [
        FileEntry(
            filename=name, bytes=(p := safe_path(root, name)).stat().st_size, sha256=file_hash(p)
        ).model_dump()
        for name in sorted(names)
    ]


def verify_files(root: Path, files: tuple[FileEntry, ...]) -> None:
    if inventory(root, [f.filename for f in files]) != [f.model_dump() for f in files]:
        raise ValueError("declared file inventory does not match bytes on disk")
