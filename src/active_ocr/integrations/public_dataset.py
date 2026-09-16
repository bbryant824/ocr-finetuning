"""Pinned READ2016 engineering conversion. Original labels/files remain oracle-side."""

from __future__ import annotations

import hashlib
import io
import json
import os
import posixpath
import re
import shutil
import subprocess
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Literal

from PIL import Image
from pydantic import Field

from active_ocr.models import (
    Box,
    CommitSHA,
    Model,
    SourceArtifact,
    SourcePolicy,
    SourceRegion,
    Split,
)

ARCHIVE_SHA256 = "f4748c58af757e06804e638a6e84c2150daab19f50d55e10aac3115e6bfc1756"
ARCHIVE_BYTES = 493223531
NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
SPLITS = {"Training": Split.TRAIN, "Validation": Split.VALIDATION}


class SourceFile(SourceArtifact):
    bytes: int = Field(ge=0)


class PageMapping(Model):
    id: str
    split: Split
    image_uri: str
    xml_uri: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    region_count: int = Field(ge=0)
    line_count: int = Field(ge=0)
    empty_lines: int = Field(ge=0)
    missing_baselines: int = Field(ge=0)


class ReadProvenance(Model):
    schema_version: Literal[1]
    mapping_version: Literal[1]
    source_policy: Literal[SourcePolicy.READ2016]
    dataset_record: Literal["https://zenodo.org/records/1297399"]
    dataset_version: Literal["1.2.0"]
    archive_sha256: Literal[ARCHIVE_SHA256]
    converter_code_sha: CommitSHA
    grouping: Literal["unknown"]
    engineering_only: Literal[True]
    test_included: Literal[False]
    files: tuple[SourceFile, ...]
    pages: tuple[PageMapping, ...]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_uri(uri: str) -> PurePosixPath:
    path = PurePosixPath(uri)
    if (
        not uri
        or not path.parts
        or "\\" in uri
        or "\x00" in uri
        or path.is_absolute()
        or any(p in (".", "..") for p in path.parts)
        or path.as_posix() != uri
    ):
        raise ValueError(f"unsafe relative source path: {uri}")
    return path


def source_path(root: Path, uri: str) -> Path:
    """No symlink component beneath (or at) the explicitly supplied dataset root."""
    path = root
    if root.is_symlink():
        raise ValueError("symlinked source root")
    for part in relative_uri(uri).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"symlinked source path: {uri}")
    if not path.is_file():
        raise ValueError(f"missing regular source file: {uri}")
    return path


def verify_inventory(root: Path, files: tuple[SourceFile, ...], subtree: str) -> None:
    names = [f.uri for f in files]
    if names != sorted(set(names)):
        raise ValueError("inventory must be sorted with unique paths")
    base = root
    if root.is_symlink():
        raise ValueError("symlinked source root")
    for part in relative_uri(subtree).parts:
        base /= part
        if base.is_symlink():
            raise ValueError("symlinked source subtree")
    actual = set()
    for path in base.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("symlink or special source file")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(names):
        raise ValueError("source inventory membership mismatch")
    for item in files:
        path = source_path(root, item.uri)
        if path.stat().st_size != item.bytes or file_digest(path) != item.sha256:
            raise ValueError(f"source file changed: {item.uri}")


def _one(element: ET.Element, tag: str) -> ET.Element:
    found = element.findall(f"{{{NAMESPACE}}}{tag}")
    if len(found) != 1:
        raise ValueError(f"expected exactly one {tag}")
    return found[0]


def _index(value: str | None) -> int:
    if value is None or not re.fullmatch(r"[0-9]+", value):
        raise ValueError("reading order requires a nonnegative integer")
    return int(value)


def _custom_index(element: ET.Element) -> int:
    custom = element.get("custom", "")
    blocks = list(re.finditer(r"([A-Za-z]+)\s*\{([^{}]*)\}", custom))
    if re.sub(r"([A-Za-z]+)\s*\{([^{}]*)\}", "", custom).strip():
        raise ValueError("malformed custom blocks")
    if any(
        m[1] not in {"readingOrder", "structure", "textStyle", "unclear", "sic", "abbrev"}
        for m in blocks
    ):
        raise ValueError("unreviewed custom annotation block")
    order = [m[2] for m in blocks if m[1] == "readingOrder"]
    if len(order) != 1 or not re.fullmatch(r"\s*index\s*:\s*[0-9]+\s*;\s*", order[0]):
        raise ValueError("missing or malformed readingOrder block")
    return int(re.search(r"[0-9]+", order[0])[0])


def parse_page(data: bytes, image_name: str) -> tuple[dict, tuple[SourceRegion, ...]]:
    """One fixed PAGE mapping; no source mutation, semantic text repair or image transforms."""
    # UTF-8 is the admitted archive encoding; disallow declarations before XML parsing.
    text = data.decode("utf-8")
    if re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", text, re.I):
        raise ValueError("DTD/entity declarations are forbidden")
    root = ET.fromstring(text)
    if root.tag != f"{{{NAMESPACE}}}PcGts":
        raise ValueError("unsupported PAGE namespace/root")
    allowed = {
        "PcGts": {"Metadata", "Page"},
        "Metadata": {"Creator", "Created", "LastChange"},
        "Page": {"ReadingOrder", "TextRegion", "PrintSpace"},
        "ReadingOrder": {"OrderedGroup"},
        "OrderedGroup": {"RegionRefIndexed"},
        "TextRegion": {"Coords", "TextLine", "TextEquiv"},
        "TextLine": {"Coords", "Baseline", "TextEquiv", "TextStyle"},
        "TextEquiv": {"Unicode"},
        "PrintSpace": {"Coords"},
    }
    attributes = {
        "PcGts": {"pcGtsId", "{http://www.w3.org/2001/XMLSchema-instance}schemaLocation"},
        "Page": {"imageFilename", "imageWidth", "imageHeight", "orientation", "rotation"},
        "OrderedGroup": {"id", "caption"},
        "RegionRefIndexed": {"index", "regionRef"},
        "TextRegion": {"type", "id", "custom", "orientation", "rotation"},
        "TextLine": {"id", "custom", "orientation", "rotation"},
        "Coords": {"points"},
        "Baseline": {"points"},
        "TextStyle": {"strikethrough", "monospace"},
    }
    for element in root.iter():
        tag = element.tag.removeprefix(f"{{{NAMESPACE}}}")
        if set(element.attrib) - attributes.get(tag, set()):
            raise ValueError("unsupported PAGE attributes/annotation flags")
        if any(
            child.tag not in {f"{{{NAMESPACE}}}{t}" for t in allowed.get(tag, ())}
            for child in element
        ):
            raise ValueError("unsupported PAGE structure or nested text")
        for attr, value in element.attrib.items():
            if attr.lower() in ("orientation", "rotation") and value not in ("0", "0.0"):
                raise ValueError("nonidentity declared rotation")
            if "illegib" in attr.lower() or "unclear" in attr.lower():
                raise ValueError("unreviewed whole-line annotation flag")
    page = _one(root, "Page")
    if page.get("imageFilename") != image_name or PurePosixPath(image_name).name != image_name:
        raise ValueError("PAGE/image basename mismatch")
    width, height = _index(page.get("imageWidth")), _index(page.get("imageHeight"))
    if not width or not height:
        raise ValueError("invalid PAGE dimensions")
    group = _one(_one(page, "ReadingOrder"), "OrderedGroup")
    regions = page.findall(f"{{{NAMESPACE}}}TextRegion")
    by_id = {r.get("id"): r for r in regions}
    refs = [(_index(r.get("index")), r.get("regionRef")) for r in group]
    if (
        None in by_id
        or "" in by_id
        or len(by_id) != len(regions)
        or len({i for i, _ in refs}) != len(refs)
        or len({r for _, r in refs}) != len(refs)
        or {r for _, r in refs} != set(by_id)
    ):
        raise ValueError("reading order must cover every unique region exactly once")
    lines = []
    missing_baselines = 0
    for index, region_id in sorted(refs):
        region = by_id[region_id]
        if _custom_index(region) != index:
            raise ValueError("region reading order mismatch")
        ordered = [
            (_custom_index(line), line) for line in region.findall(f"{{{NAMESPACE}}}TextLine")
        ]
        ordered.sort(key=lambda item: item[0])
        if [i for i, _ in ordered] != list(range(len(ordered))):
            raise ValueError("line reading order must be contiguous and unique")
        for _, line in ordered:
            unicode = _one(_one(line, "TextEquiv"), "Unicode")
            if len(unicode):
                raise ValueError("nested Unicode text")
            points = _one(line, "Coords").get("points", "").split()
            if any(not re.fullmatch(r"[0-9]+,[0-9]+", point) for point in points):
                raise ValueError("polygon requires finite integer coordinate pairs")
            vertices = [tuple(map(int, point.split(","))) for point in points]
            if len(set(vertices)) < 3:
                raise ValueError("polygon needs at least three distinct vertices")
            xs, ys = zip(*vertices, strict=True)
            if max(xs) > width or max(ys) > height:
                raise ValueError("polygon outside original pixel bounds")
            box = Box(x=min(xs), y=min(ys), width=max(xs) - min(xs), height=max(ys) - min(ys))
            lines.append(
                SourceRegion(
                    id=line.get("id", ""), box=box, text=unicode.text or "", illegible=False
                )
            )
            missing_baselines += int(not line.findall(f"{{{NAMESPACE}}}Baseline"))
    if len({line.id for line in lines}) != len(lines):
        raise ValueError("line IDs must be unique across the page")
    return {
        "width": width,
        "height": height,
        "region_count": len(regions),
        "line_count": len(lines),
        "empty_lines": sum(r.text == "" for r in lines),
        "missing_baselines": missing_baselines,
    }, tuple(lines)


def validate_image(data: bytes, width: int, height: int) -> None:
    with Image.open(io.BytesIO(data)) as image:
        if (
            image.format != "JPEG"
            or image.mode != "RGB"
            or getattr(image, "n_frames", 1) != 1
            or image.getexif().get(274, 1) != 1
            or image.size != (width, height)
        ):
            raise ValueError("READ image format/mode/frame/orientation/dimensions mismatch")
        image.load()


def _page_paths(files: tuple[SourceFile, ...], prefix: str = "") -> list[tuple[str, str, Split]]:
    names = {f.uri for f in files}
    pairs = []
    expected = set()
    for folder, split in SPLITS.items():
        base = f"{prefix}PublicData/{folder}"
        xmls = sorted(n for n in names if n.startswith(base + "/page/page/") and n.endswith(".xml"))
        if len(xmls) != (350 if split is Split.TRAIN else 50):
            raise ValueError("READ requires official 350/50 page coverage")
        for xml in xmls:
            stem = PurePosixPath(xml).stem
            image = f"{base}/Images/{stem}.JPG"
            if xml != f"{base}/page/page/{stem}.xml":
                raise ValueError("unexpected nested PAGE path")
            pairs.append((image, xml, split))
            expected.update((image, xml))
        expected.update((f"{base}/page/doc.xml", f"{base}/page/list"))
    if names != expected or len(files) != 804:
        raise ValueError("READ regular inventory/pairing mismatch")
    return pairs


def validate_provenance(root: Path, reference: SourceArtifact, pages: tuple) -> None:
    """Verify original files and literal row mapping, without exposing provenance to adapters."""
    if reference.uri != "provenance.json":
        raise ValueError("READ rows require relative provenance.json")
    data = source_path(root, reference.uri).read_bytes()
    if digest(data) != reference.sha256:
        raise ValueError("source provenance changed")
    provenance = ReadProvenance.model_validate_json(data)
    verify_inventory(root, provenance.files, "source/PublicData")
    pairs = _page_paths(provenance.files, "source/")
    if len(pages) != len(pairs) or len(provenance.pages) != len(pairs):
        raise ValueError("manifest/provenance page coverage mismatch")
    inventory = {f.uri: f for f in provenance.files}
    for row, mapping, (image, xml, split) in zip(pages, provenance.pages, pairs, strict=True):
        observed, regions = parse_page(
            source_path(root, xml).read_bytes(), PurePosixPath(image).name
        )
        expected = PageMapping(
            id=f"read2016-v1.2.0:{split}:{PurePosixPath(xml).stem}",
            split=split,
            image_uri=image,
            xml_uri=xml,
            **observed,
        )
        if mapping != expected or (
            row.id,
            row.split,
            row.image_uri,
            row.width,
            row.height,
            row.image_sha256,
            row.regions,
        ) != (
            mapping.id,
            split,
            image,
            mapping.width,
            mapping.height,
            inventory[image].sha256,
            regions,
        ):
            raise ValueError("manifest/provenance literal page mapping mismatch")


def _archive_inventory(archive: Path) -> tuple[SourceFile, ...]:
    if (
        archive.is_symlink()
        or archive.stat().st_size != ARCHIVE_BYTES
        or file_digest(archive) != ARCHIVE_SHA256
    ):
        raise ValueError("archive is not the pinned READ2016 1.2.0 archive")
    files = _tar_inventory(archive)
    if file_digest(archive) != ARCHIVE_SHA256:
        raise ValueError("archive changed during verification")
    return files


def _tar_inventory(archive: Path) -> tuple[SourceFile, ...]:
    """Validate archive members without extracting; caller enforces the pinned archive bytes."""
    files, links, seen = [], {}, set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = member.name.rstrip("/") if member.isdir() else member.name
            relative_uri(name)
            if name in seen:
                raise ValueError("duplicate archive member")
            seen.add(name)
            if member.isdir():
                if name != "PublicData" and not any(
                    (name == f"PublicData/{s}" or name.startswith(f"PublicData/{s}/"))
                    for s in SPLITS
                ):
                    raise ValueError("unexpected archive directory")
            elif member.isfile():
                with tar.extractfile(member) as stream:
                    checksum = hashlib.file_digest(stream, "sha256").hexdigest()
                files.append(SourceFile(uri=name, bytes=member.size, sha256=checksum))
            elif member.issym():
                links[name] = member.linkname
            else:
                raise ValueError("hard links and special archive members are forbidden")
    files = tuple(sorted(files, key=lambda f: f.uri))
    pairs = _page_paths(files)
    expected_links = {image.replace("/Images/", "/page/"): image for image, _, _ in pairs}
    if set(links) != set(expected_links):
        raise ValueError("unexpected archive image symlinks")
    for name, target in links.items():
        if (
            PurePosixPath(target).is_absolute()
            or "\\" in target
            or posixpath.normpath(posixpath.join(posixpath.dirname(name), target))
            != expected_links[name]
        ):
            raise ValueError("unsafe archive image symlink target")
    return files


def _write_synced(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _publish(extracted: Path, output: Path, files: tuple[SourceFile, ...], code_sha: str) -> Path:
    """Build from verified files; reserve once, publish manifest last, never overwrite a dataset."""
    from active_ocr.integrations.simulation import _read_source

    output.mkdir(exist_ok=False)
    owned = []
    try:
        with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as tmp:
            stage = Path(tmp)
            copied = []
            for file in files:
                data = source_path(extracted, file.uri).read_bytes()
                if len(data) != file.bytes or digest(data) != file.sha256:
                    raise ValueError("staged file changed during copy")
                uri = f"source/{file.uri}"
                _write_synced(stage / uri, data)
                copied.append(SourceFile(uri=uri, bytes=file.bytes, sha256=file.sha256))
            copied = tuple(copied)
            mappings, rows = [], []
            by_uri = {f.uri: f for f in copied}
            for image, xml, split in _page_paths(copied, "source/"):
                observed, regions = parse_page(
                    source_path(stage, xml).read_bytes(), PurePosixPath(image).name
                )
                mapping = PageMapping(
                    id=f"read2016-v1.2.0:{split}:{PurePosixPath(xml).stem}",
                    split=split,
                    image_uri=image,
                    xml_uri=xml,
                    **observed,
                )
                mappings.append(mapping)
                rows.append(
                    dict(
                        id=mapping.id,
                        document_id=None,
                        image_uri=image,
                        split=split,
                        width=mapping.width,
                        height=mapping.height,
                        image_sha256=by_uri[image].sha256,
                        source_policy=SourcePolicy.READ2016,
                        regions=[r.model_dump() for r in regions],
                    )
                )
            provenance = ReadProvenance(
                schema_version=1,
                mapping_version=1,
                source_policy=SourcePolicy.READ2016,
                dataset_record="https://zenodo.org/records/1297399",
                dataset_version="1.2.0",
                archive_sha256=ARCHIVE_SHA256,
                grouping="unknown",
                engineering_only=True,
                test_included=False,
                converter_code_sha=code_sha,
                files=copied,
                pages=tuple(mappings),
            )
            data = provenance.model_dump_json(indent=2).encode() + b"\n"
            _write_synced(stage / "provenance.json", data)
            for row in rows:
                row["source_provenance"] = {"uri": "provenance.json", "sha256": digest(data)}
            manifest = "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
            ).encode()
            _write_synced(stage / "pending.jsonl", manifest)
            _read_source(manifest, stage, source_policy=SourcePolicy.READ2016)
            for name in ("source", "provenance.json", "pages.jsonl"):
                target = output / name
                if target.exists() or target.is_symlink():
                    raise FileExistsError(target)
            os.replace(stage / "source", output / "source")
            owned.append(output / "source")
            os.replace(stage / "provenance.json", output / "provenance.json")
            owned.append(output / "provenance.json")
            verify_inventory(output, copied, "source/PublicData")
            if (output / "provenance.json").read_bytes() != data:
                raise ValueError("persisted provenance changed")
            if (stage / "pending.jsonl").read_bytes() != manifest:
                raise ValueError("pending manifest changed")
            directory_fd = os.open(output, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            os.replace(stage / "pending.jsonl", output / "pages.jsonl")
        return output / "pages.jsonl"
    except Exception as error:
        if (output / "pages.jsonl").exists():
            # Publication can succeed before a response/temporary-cleanup failure.
            # Never turn a completed dataset into an unreadable partial one.
            error.add_note(f"Completion manifest present at {output}; verify before retrying")
            raise
        # Remove only paths installed by this invocation; preserve any unrelated contents.
        for path in reversed(owned):
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        try:
            output.rmdir()
        except OSError:
            error.add_note(f"Incomplete output retained at {output}; use a fresh destination")
        raise


def convert_read2016(
    archive: Path, extracted: Path, output: Path, *, source_policy: SourcePolicy
) -> Path:
    """Convert only the pinned admitted archive; no extraction, network or resume/overwrite mode."""
    if SourcePolicy(source_policy) is not SourcePolicy.READ2016:
        raise ValueError("conversion requires explicit READ engineering policy")
    if output.resolve().is_relative_to(extracted.resolve()):
        raise ValueError("output must not be inside staged source")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    files = _archive_inventory(archive)
    verify_inventory(extracted, files, "PublicData")
    root = Path(__file__).resolve().parents[3]
    code_sha = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("conversion requires a clean committed checkout")
    if shutil.disk_usage(output.parent).free < 2 * 1024**3:
        raise ValueError("conversion requires 2 GiB scratch headroom")
    return _publish(extracted, output, files, code_sha)
