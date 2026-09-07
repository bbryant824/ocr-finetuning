"""Import pre-rendered page images from a small JSONL or CSV manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections.abc import Iterator, Mapping
from pathlib import Path

from PIL import Image

from active_ocr.models import Page, Split

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff"})


def _entries(manifest: Path) -> Iterator[Mapping[str, object]]:
    if manifest.suffix.lower() == ".jsonl":
        for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on manifest line {line_number}") from exc
            if not isinstance(entry, dict):
                raise ValueError(f"manifest line {line_number} must contain an object")
            yield entry
        return
    if manifest.suffix.lower() == ".csv":
        with manifest.open(newline="", encoding="utf-8") as handle:
            yield from csv.DictReader(handle)
        return
    raise ValueError("page manifest must use .jsonl or .csv")


def _split(document_id: str, train: float, validation: float) -> Split:
    digest = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:16], 16) / 0xFFFFFFFFFFFFFFFF
    if bucket < train:
        return Split.TRAIN
    if bucket < train + validation:
        return Split.VALIDATION
    return Split.TEST


def load_image_pages(
    manifest: Path,
    *,
    document_root: Path,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> tuple[Page, ...]:
    """Load manifest pages and split them by document, never by page.

    Each row requires ``path`` and ``document_id``. Relative paths are resolved
    from the manifest directory. Images are copied into ``document_root`` and
    referenced using Label Studio's local-file URL. Duplicate paths are rejected.
    """

    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if min(train_ratio, validation_ratio, test_ratio) < 0:
        raise ValueError("split ratios must be non-negative")
    if abs(train_ratio + validation_ratio + test_ratio - 1) > 1e-9:
        raise ValueError("split ratios must sum to 1")

    pages: list[Page] = []
    seen_paths: set[Path] = set()
    for row_number, entry in enumerate(_entries(manifest), 1):
        path_value = entry.get("path")
        document_value = entry.get("document_id")
        if not isinstance(path_value, str) or not isinstance(document_value, str):
            raise ValueError(f"manifest row {row_number} requires path and document_id")
        relative_path = path_value.strip()
        document_id = document_value.strip()
        if not relative_path or not document_id:
            raise ValueError(f"manifest row {row_number} requires path and document_id")
        image_path = Path(relative_path)
        if not image_path.is_absolute():
            image_path = manifest.parent / image_path
        image_path = image_path.resolve()
        if image_path in seen_paths:
            raise ValueError(f"duplicate page path in manifest: {relative_path}")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported page image: {image_path.suffix}")
        seen_paths.add(image_path)

        content_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        page_identity = f"{document_id}\0{relative_path}\0{content_hash}".encode()
        page_id = hashlib.sha256(page_identity).hexdigest()
        with Image.open(image_path) as image:
            width, height = image.size
        relative_copy = Path("pages") / page_id[:2] / f"{page_id}{image_path.suffix.lower()}"
        stored_image = document_root / relative_copy
        stored_image.parent.mkdir(parents=True, exist_ok=True)
        if not stored_image.exists():
            shutil.copy2(image_path, stored_image)
        pages.append(
            Page(
                id=page_id,
                document_id=document_id,
                image_uri=f"/data/local-files/?d={relative_copy.as_posix()}",
                width=width,
                height=height,
                split=_split(document_id, train_ratio, validation_ratio),
            )
        )
    if not pages:
        raise ValueError("page manifest is empty")
    return tuple(pages)
