"""Pinned, synchronous Qwen page OCR. GPU libraries are imported only on use.

See docs/qwen-runtime.md for the worker manifest and staged verification gates.
No oracle, selection, transport, automatic downloads or CPU runtime fallback lives here.
"""

from __future__ import annotations

import gc
import hashlib
import io
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
from array import array
from contextlib import suppress
from importlib.metadata import distributions
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import Field, field_validator

from active_ocr.models import (
    SHA256,
    Box,
    ExecutionTelemetry,
    Model,
    Prediction,
    PredictionPurpose,
    PredictionStatus,
    RealOCRConfig,
    Region,
    RevealedExample,
    RunKind,
    SimulationPage,
    Split,
)

REPOSITORY = "Qwen/Qwen3-VL-4B-Instruct"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
RECIPE = "qwen3-vl-read-engineering-v1"
TRAIN_POLICY = "qwen3-vl-page-lora-v1"
DECODE_POLICY = "qwen3-vl-page-greedy-v1"
EOS, PAD, IMAGE = 151645, 151643, 151655
PROMPT = "\n".join(
    (
        "Transcribe every text line in reading order, "
        "including headings, page numbers and marginalia.",
        "Return only JSON with exactly this shape: "
        '{"regions":[{"text":"...","bbox":[x1,y1,x2,y2]}]}.',
        "Preserve spelling, punctuation, spacing and empty text. "
        "Give one box per line. Coordinates are",
        "numbers from 0 to 1000 relative to the original full page, "
        "with x2>x1 and y2>y1. Do not explain.",
    )
)
PINS = {
    "torch": "2.14.0",
    "torchvision": "0.29.0",
    "transformers": "5.16.1",
    "peft": "0.20.0",
    "accelerate": "1.14.0",
    "tokenizers": "0.23.2",
    "safetensors": "0.8.0",
    "pillow": "12.3.0",
}
TARGETS = tuple(
    f"model.language_model.layers.{i}.self_attn.{projection}_proj"
    for i in range(36)
    for projection in ("q", "v")
)
BASE_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
)
PROCESSOR_FILES = (
    "config.json",
    "chat_template.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
)
# Public staged provenance, experiments/assets/read2016-qwen3vl4b.json. No raw labels.
ASSETS: dict[str, tuple[int, str]] = {
    "chat_template.json": (
        5502,
        "6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4",
    ),
    "config.json": (1505, "edac7703329133edfc53e46ac0081835144c99d7eebf28b71c732694d435224d"),
    "generation_config.json": (
        269,
        "8469742d1fce0de951c8909b26a2c0c0d8490837ce476efb114da9e0cefc4d44",
    ),
    "merges.txt": (1671839, "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
    "model-00001-of-00002.safetensors": (
        4967229296,
        "30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9",
    ),
    "model-00002-of-00002.safetensors": (
        3908490048,
        "046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6",
    ),
    "model.safetensors.index.json": (
        64742,
        "58a7841d7bff2548dd91577d216274a83cf1b500bc6a534b809d6c1b1707cf2b",
    ),
    "preprocessor_config.json": (
        390,
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    ),
    "tokenizer.json": (7032403, "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7"),
    "tokenizer_config.json": (
        10868,
        "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399",
    ),
    "video_preprocessor_config.json": (
        385,
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    ),
    "vocab.json": (2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
}


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
    return Packages(
        python=platform.python_version(),
        packages=tuple(
            sorted(
                (re.sub(r"[-_.]+", "-", d.metadata["Name"].lower()), d.version)
                for d in distributions()
                if d.metadata["Name"]
            )
        ),
    )


def inventory(root: Path, names) -> list[dict]:
    return [
        FileEntry(
            filename=name, bytes=(p := safe_path(root, name)).stat().st_size, sha256=file_hash(p)
        ).model_dump()
        for name in sorted(names)
    ]


def asset_manifest(names) -> dict:
    return {
        "schema_version": 1,
        "repository": REPOSITORY,
        "revision": REVISION,
        "files": [dict(filename=n, bytes=ASSETS[n][0], sha256=ASSETS[n][1]) for n in sorted(names)],
    }


def resize_geometry(height: int, width: int) -> tuple[int, int, int]:
    if type(height) is not int or type(width) is not int or min(height, width) < 1:
        raise ValueError("invalid dimensions")
    budget = math.ceil(1024**2 * min(height, width) / max(height, width))
    if budget < 65536:
        raise ValueError("aspect ratio cannot meet pixel limits")
    h, w = round(height / 32) * 32, round(width / 32) * 32
    if h * w > budget:
        beta = math.sqrt(height * width / budget)
        h, w = (max(32, math.floor(d / beta / 32) * 32) for d in (height, width))
    elif h * w < 65536:
        beta = math.sqrt(65536 / (height * width))
        h, w = (math.ceil(d * beta / 32) * 32 for d in (height, width))
    if not 65536 <= h * w <= budget or max(h, w) > 1024:
        raise ValueError("rounded geometry cannot meet pixel limits")
    return h, w, budget


def serialize_target(example: RevealedExample) -> str:
    if example.page.split is not Split.TRAIN:
        raise ValueError("targets must be selected TRAIN")
    width, height = example.page.width, example.page.height
    regions = []
    seen = set()
    for region in example.regions:
        if region.id in seen:
            raise ValueError("duplicate target region")
        seen.add(region.id)
        x, y, w, h = (region.box.x, region.box.y, region.box.width, region.box.height)
        if (
            not all(math.isfinite(v) for v in (x, y, w, h))
            or min(x, y) < 0
            or min(w, h) <= 0
            or x + w > width
            or y + h > height
        ):
            raise ValueError("target outside original page")
        region.text.encode("utf-8", errors="strict")
        regions.append(
            {
                "text": region.text,
                "bbox": [
                    1000 * x / width,
                    1000 * y / height,
                    1000 * (x + w) / width,
                    1000 * (y + h) / height,
                ],
            }
        )
    return json.dumps(
        {"regions": regions}, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).replace("<", r"\u003c")


def parse_regions(raw: str, width: int, height: int) -> tuple[Region, ...]:
    raw.encode("utf-8", errors="strict")
    try:
        data = strict_json(raw)
    except RecursionError as exc:
        raise ValueError("generated JSON nesting exceeds parser depth") from exc
    if not isinstance(data, dict) or set(data) != {"regions"}:
        raise ValueError("expected only regions")
    if not isinstance(data["regions"], list):
        raise ValueError("regions must be an array")
    result = []
    for i, region in enumerate(data["regions"], 1):
        if not isinstance(region, dict) or set(region) != {"text", "bbox"}:
            raise ValueError("expected only text and bbox")
        if not isinstance(region["text"], str):
            raise ValueError("text must be a string")
        region["text"].encode("utf-8", errors="strict")
        box = region["bbox"]
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1000
                for v in box
            )
        ):
            raise ValueError("invalid normalized box")
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            raise ValueError("degenerate box")
        result.append(
            Region(
                id=f"line-{i:04d}",
                text=region["text"],
                illegible=False,
                box=Box(
                    x=x1 * width / 1000,
                    y=y1 * height / 1000,
                    width=x2 * width / 1000 - x1 * width / 1000,
                    height=y2 * height / 1000 - y1 * height / 1000,
                ),
            )
        )
    return tuple(result)


def epoch_groups(ids: tuple[str, ...], seed: int, epoch: int) -> list[list[str]]:
    ordered = sorted(ids)
    random.Random(seed + epoch).shuffle(ordered)
    return [ordered[i : i + 4] for i in range(0, len(ordered), 4)]


def training_span(
    prompt: list[int], full: list[int], target: list[int], newline: list[int], special_ids: set[int]
) -> tuple[list[int], list[int]]:
    if not target or special_ids.intersection(target):
        raise ValueError("target contains control token")
    if full != prompt + target + [EOS] + newline:
        raise ValueError("prompt/full prefix or assistant framing mismatch")
    p, t = len(prompt), len(target) + 1
    if p > 2048 or t > 4096 or p + t > 6144:
        raise ValueError("training token overflow")
    ids = full[: p + t]
    return ids, [-100] * p + ids[p:]


def generation_config():
    from transformers import GenerationConfig

    return GenerationConfig(
        do_sample=False,
        num_beams=1,
        num_return_sequences=1,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        max_new_tokens=2048,
        eos_token_id=EOS,
        pad_token_id=PAD,
        bos_token_id=None,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        stop_strings=None,
        use_cache=True,
    )


def decode_result(ids: list[int], tokenizer, width: int, height: int):
    # Preserve all control tokens in the evidence; strip only a terminal EOS for parsing.
    raw = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    terminated = bool(ids) and ids[-1] == EOS
    if not terminated:
        if len(ids) == 2048:
            return PredictionStatus.TRUNCATED, (), raw, "length"
        raise RuntimeError("generation stopped without EOS or declared token limit")
    content = ids[:-1]
    try:
        if set(content).intersection(tokenizer.all_special_ids):
            raise ValueError("unexpected generated control token")
        text = tokenizer.decode(
            content, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        regions = parse_regions(text, width, height)
    except (ValueError, UnicodeError, OverflowError):
        return PredictionStatus.INVALID_OUTPUT, (), raw, "eos"
    return PredictionStatus.OK, regions, raw, "eos"


def load_processor(model_root: Path):
    from transformers import AutoTokenizer, Qwen3VLProcessor, Qwen3VLVideoProcessor
    from transformers.image_processing_backends import TorchvisionBackend
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    kwargs = {"local_files_only": True, "trust_remote_code": False}
    image = Qwen2VLImageProcessor.from_pretrained(model_root, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_root, use_fast=True, **kwargs)
    video = Qwen3VLVideoProcessor.from_pretrained(model_root, **kwargs)
    template = strict_json((model_root / "chat_template.json").read_bytes())["chat_template"]
    processor = Qwen3VLProcessor(
        image_processor=image, tokenizer=tokenizer, video_processor=video, chat_template=template
    )
    if (
        type(image) is not Qwen2VLImageProcessor
        or not isinstance(image, TorchvisionBackend)
        or type(processor) is not Qwen3VLProcessor
        or not tokenizer.is_fast
    ):
        raise ValueError("unexpected processor backend")
    expected = {
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "do_resize": True,
        "do_rescale": True,
        "do_normalize": True,
        "rescale_factor": 1 / 255,
        "image_mean": [0.5] * 3,
        "image_std": [0.5] * 3,
        "resample": Image.Resampling.BICUBIC,
    }
    if any(getattr(image, k) != v for k, v in expected.items()):
        raise ValueError("processor settings changed")
    for token, expected_id in (
        ("<|im_end|>", EOS),
        ("<|endoftext|>", PAD),
        ("<|image_pad|>", IMAGE),
    ):
        if tokenizer.convert_tokens_to_ids(token) != expected_id:
            raise ValueError("tokenizer special ID mismatch")
    return processor


def encode_page(processor, image: Image.Image, target: str | None = None):
    """No source metadata can enter the template; this accepts pixels and selected text only."""
    h, w, budget = resize_geometry(image.height, image.width)
    user = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt_text = processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    if not prompt_text.endswith("<|im_start|>assistant\n"):
        raise ValueError("assistant framing changed")
    kwargs = dict(
        images=[image],
        images_kwargs={"size": {"shortest_edge": 65536, "longest_edge": budget}, "device": "cpu"},
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_tensors="pt",
        return_mm_token_type_ids=True,
    )
    prompt = processor(text=[prompt_text], **kwargs)
    pids = prompt["input_ids"][0].tolist()
    grid = prompt["image_grid_thw"].tolist()
    if grid != [[1, h // 16, w // 16]] or pids.count(IMAGE) != h * w // 1024:
        raise ValueError("actual image grid/token count mismatch")
    if len(pids) > 2048 or len(pids) + 2048 > 6144:
        raise ValueError("prompt token overflow")
    receipt = {
        "height": h,
        "width": w,
        "grid": grid[0],
        "prompt_tokens": len(pids),
        "target_tokens": 0,
        "target_exceeds_decode_budget": False,
    }
    if target is None:
        return dict(prompt), receipt
    full_text = processor.apply_chat_template(
        user + [{"role": "assistant", "content": target}],
        tokenize=False,
        add_generation_prompt=False,
    )
    full = processor(text=[full_text], **kwargs)
    tokenizer = processor.tokenizer
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    ids, labels = training_span(
        pids,
        full["input_ids"][0].tolist(),
        target_ids,
        tokenizer.encode("\n", add_special_tokens=False),
        set(tokenizer.all_special_ids),
    )
    if full["image_grid_thw"].tolist() != grid:
        raise ValueError("full/prompt grids differ")
    n = len(ids)
    for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
        full[key] = full[key][:, :n]
    full["labels"] = full["input_ids"].new_tensor([labels])
    full["labels"][full["attention_mask"] == 0] = -100
    receipt.update(
        target_tokens=len(target_ids) + 1, target_exceeds_decode_budget=len(target_ids) + 1 > 2048
    )
    return dict(full), receipt


def lora_config():
    from peft import LoraConfig

    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(TARGETS),
        init_lora_weights=True,
        use_rslora=False,
        use_dora=False,
        modules_to_save=None,
        base_model_name_or_path=REPOSITORY,
        revision=REVISION,
    )


def adapter_shapes() -> dict[str, tuple[int, int]]:
    shapes = {}
    for name in TARGETS:
        prefix = "base_model.model." + name
        shapes[prefix + ".lora_A.weight"] = (16, 2560)
        shapes[prefix + ".lora_B.weight"] = (4096 if name.endswith("q_proj") else 1024, 16)
    return shapes


def check_base_modules(model) -> None:
    modules = dict(model.named_modules())
    for name in TARGETS:
        expected = (4096 if name.endswith("q_proj") else 1024, 2560)
        if name not in modules or tuple(modules[name].weight.shape) != expected:
            raise ValueError("base language projection mismatch: " + name)


def tensor_hash(tensor) -> str:
    import torch

    h = hashlib.sha256()
    flat = tensor.detach().reshape(-1)
    h.update(canonical({"shape": list(tensor.shape), "dtype": str(tensor.dtype)}))
    for chunk in flat.split(1024 * 1024):
        h.update(chunk.contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def frozen_hashes(model) -> dict[str, str]:
    return {name: tensor_hash(p) for name, p in model.named_parameters() if not p.requires_grad}


def check_trainables(model) -> list:
    import torch

    expected = {
        name.replace(".weight", ".default.weight"): shape
        for name, shape in adapter_shapes().items()
    }
    actual = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(actual) != set(expected):
        raise ValueError("unexpected trainable names")
    if any(
        tuple(p.shape) != expected[name] or p.dtype != torch.float32 for name, p in actual.items()
    ):
        raise ValueError("unexpected trainable shape/dtype")
    if sum(p.numel() for p in actual.values()) != 5898240:
        raise ValueError("unexpected trainable count")
    return list(actual.values())


def check_adapter(path: Path) -> dict[str, str]:
    """Verify configuration and all tensors before loading any base weights."""
    import torch
    from safetensors import safe_open

    expected = lora_config().to_dict()
    expected.update(inference_mode=True)
    expected["target_modules"] = sorted(expected["target_modules"])
    config = strict_json(safe_path(path, "adapter_config.json").read_bytes())
    if isinstance(config.get("target_modules"), list):
        config["target_modules"] = sorted(config["target_modules"])
    # Convert enums/sets exactly as PEFT's JSON writer does.
    if canonical(config) != canonical(expected):
        raise ValueError("adapter configuration mismatch")
    shapes = adapter_shapes()
    hashes = {}
    with safe_open(safe_path(path, "adapter_model.safetensors"), framework="pt", device="cpu") as f:
        if set(f.keys()) != set(shapes):
            raise ValueError("adapter tensor keys mismatch")
        for key, shape in shapes.items():
            tensor = f.get_tensor(key)
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype != torch.float32
                or not torch.isfinite(tensor).all().item()
            ):
                raise ValueError("invalid adapter tensor: " + key)
            hashes[key] = tensor_hash(tensor)
    return hashes


class Checkpoint(Model):
    schema_version: Literal[1] = 1
    kind: Literal["base", "adapter"]
    binding: dict[str, Any]
    experiment_id: str | None = None
    round_number: int | None = Field(default=None, ge=1, strict=True)
    selected: tuple[tuple[str, SHA256], ...] = ()
    target_sha256: SHA256 | None = None
    seed: int | None = Field(default=None, ge=0, lt=2**32, strict=True)
    training: dict[str, Any] | None = None
    files: tuple[FileEntry, ...] = ()


def check_training_record(checkpoint: Checkpoint) -> None:
    training = checkpoint.training
    keys = {
        "updates",
        "epoch_orders",
        "losses",
        "gradient_norms",
        "processing",
        "supervised_tokens",
        "changed_tensors",
        "frozen_sha256",
    }
    if training is None or set(training) != keys:
        raise ValueError("invalid training record")
    ids = tuple(p[0] for p in checkpoint.selected)
    expected_orders = [
        [p for group in epoch_groups(ids, checkpoint.seed, e) for p in group] for e in range(3)
    ]
    updates = 3 * math.ceil(len(ids) / 4)
    if (
        training["epoch_orders"] != expected_orders
        or training["updates"] != updates
        or len(training["losses"]) != 3 * len(ids)
        or len(training["gradient_norms"]) != updates
        or type(training["changed_tensors"]) is not int
        or not 1 <= training["changed_tensors"] <= 144
        or re.fullmatch(r"[0-9a-f]{64}", training["frozen_sha256"]) is None
    ):
        raise ValueError("training count/order mismatch")
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or v < 0
        for v in training["losses"] + training["gradient_norms"]
    ):
        raise ValueError("invalid loss/gradient diagnostics")
    processing = training["processing"]
    if set(processing) != set(ids):
        raise ValueError("processing coverage mismatch")
    for record in processing.values():
        if set(record) != {
            "height",
            "width",
            "grid",
            "prompt_tokens",
            "target_tokens",
            "target_exceeds_decode_budget",
        }:
            raise ValueError("invalid processing record")
        h, w, p, t = (record[k] for k in ("height", "width", "prompt_tokens", "target_tokens"))
        if (
            any(type(v) is not int for v in (h, w, p, t))
            or h % 32
            or w % 32
            or min(h, w) < 32
            or max(h, w) > 1024
            or h * w < 65536
            or record["grid"] != [1, h // 16, w // 16]
            or not 1 <= p <= 2048
            or not 1 <= t <= 4096
            or p + t > 6144
            or record["target_exceeds_decode_budget"] is not (t > 2048)
        ):
            raise ValueError("invalid processing geometry/token limits")
    if training["supervised_tokens"] != 3 * sum(r["target_tokens"] for r in processing.values()):
        raise ValueError("supervised token count mismatch")


def publish_checkpoint(root: Path, checkpoint: Checkpoint, staged: Path | None = None) -> str:
    data = canonical(checkpoint.model_dump(mode="json"))
    key = hashlib.sha256(data).hexdigest()
    directory = safe_path(root, "checkpoints/" + key)
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        verify_checkpoint_files(directory, checkpoint, data)
        return "checkpoint:sha256:" + key
    # Exclusive directory reservation: an interrupted directory is never considered complete.
    directory.mkdir()
    for entry in checkpoint.files:
        if staged is None:
            raise ValueError("missing staged adapter")
        source = safe_path(staged, entry.filename)
        if source.stat().st_size != entry.bytes or file_hash(source) != entry.sha256:
            raise ValueError("staged adapter changed")
        with safe_path(directory, entry.filename).open("xb") as dest, source.open("rb") as src:
            for chunk in iter(lambda: src.read(1024 * 1024), b""):
                dest.write(chunk)
            dest.flush()
            os.fsync(dest.fileno())
    # Manifest is the last file; incomplete publications deliberately remain failures.
    with (directory / "manifest.json").open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    verify_checkpoint_files(directory, checkpoint, data)
    return "checkpoint:sha256:" + key


def verify_adapter_files(directory: Path, files: tuple[FileEntry, ...]) -> None:
    if inventory(directory, [f.filename for f in files]) != [f.model_dump() for f in files]:
        raise ValueError("checkpoint bytes changed")


def verify_checkpoint_files(directory: Path, checkpoint: Checkpoint, data: bytes) -> None:
    actual_names = {p.name for p in directory.iterdir()}
    names = [f.filename for f in checkpoint.files]
    if len(set(names)) != len(names) or actual_names != {*names, "manifest.json"}:
        raise ValueError("checkpoint file inventory mismatch")
    if safe_path(directory, "manifest.json").read_bytes() != data:
        raise ValueError("checkpoint manifest bytes changed")
    verify_adapter_files(directory, checkpoint.files)


def train_epochs(model, batches: dict[str, dict], seed: int, parameters: list, context, guard=None):
    """Page means, including the actual final accumulation-group denominator."""
    import torch

    optimizer = torch.optim.AdamW(
        parameters,
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        foreach=False,
        fused=False,
    )
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer.zero_grad(set_to_none=True)
    losses, norms, orders = [], [], []
    for epoch in range(3):
        groups = epoch_groups(tuple(batches), seed, epoch)
        orders.append([page for group in groups for page in group])
        for group in groups:
            for page in group:
                if guard is not None:
                    guard()
                with context():
                    loss = model(**batches[page], use_cache=False).loss
                    if not torch.isfinite(loss).item():
                        raise RuntimeError("nonfinite training loss")
                    losses.append(float(loss.detach()))
                    (loss / len(group)).backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in parameters):
                raise RuntimeError("missing/nonfinite adapter gradients")
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            norms.append(float(norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return {
        "updates": len(norms),
        "epoch_orders": orders,
        "losses": losses,
        "gradient_norms": norms,
    }


def reset_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cuda_context():
    from contextlib import ExitStack

    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    stack = ExitStack()
    stack.enter_context(sdpa_kernel(SDPBackend.FLASH_ATTENTION))
    stack.enter_context(torch.autocast("cuda", dtype=torch.bfloat16))
    return stack


def reload_probe(
    model, inputs: dict, tokenizer, *, original_size: tuple[int, int], fixed_ids=None
) -> dict:
    """Selected TRAIN engineering probe. Never an unbiased evaluation result."""
    import torch
    from peft import get_peft_model_state_dict

    model.eval()
    with torch.inference_mode(), cuda_context():
        output = model.generate(**inputs, generation_config=generation_config())
        prefix = inputs["input_ids"].shape[1]
        if not torch.equal(output[:, :prefix], inputs["input_ids"]):
            raise RuntimeError("reload probe prompt prefix changed")
        ids = output[0, prefix:].tolist()
        status, regions, _, reason = decode_result(ids, tokenizer, *original_size)
        # Fixed generated prefix, full vocabulary, never sampling scores/warpers.
        probe = dict(inputs)
        reference_ids = ids if fixed_ids is None else fixed_ids
        continuation = inputs["input_ids"].new_tensor(
            [reference_ids[: min(8, len(reference_ids)) - 1]]
        )
        probe["input_ids"] = torch.cat((inputs["input_ids"], continuation), dim=1)
        added = probe["input_ids"].shape[1] - prefix
        for key, fill in (("attention_mask", 1), ("mm_token_type_ids", 0)):
            probe[key] = torch.cat((inputs[key], inputs[key].new_full((1, added), fill)), dim=1)
        logits = model(**probe, use_cache=False).logits[0, prefix - 1 :].float().cpu()
    if not torch.isfinite(logits).all().item():
        raise RuntimeError("nonfinite reload probe logits")
    return {
        "ids": ids,
        "status": status.value,
        "regions": [r.model_dump() for r in regions],
        "finish_reason": reason,
        "logits": logits,
        "tensors": {k: tensor_hash(v) for k, v in get_peft_model_state_dict(model).items()},
    }


def compare_probes(before: dict, after: dict) -> float:
    import torch

    for key in ("ids", "status", "regions", "finish_reason", "tensors"):
        if before[key] != after[key]:
            raise RuntimeError("reload probe mismatch: " + key)
    a, b = before["logits"], after["logits"]
    if (
        a.shape != b.shape
        or not torch.isfinite(a).all()
        or not torch.isfinite(b).all()
        or not torch.allclose(a, b, rtol=1e-3, atol=1e-2)
    ):
        raise RuntimeError("reload logits exceed predeclared tolerance")
    return float((a - b).abs().max())


class QwenModel:
    backend = "qwen3-vl-v1"
    fit_policy = "reset-fit-cumulative-v1"
    kind = RunKind.REAL

    def __init__(
        self,
        input_root: Path,
        output_root: Path,
        real_config: RealOCRConfig,
        *,
        runtime_manifest: Path,
        deadline_unix_seconds: float | None = None,
    ):
        if deadline_unix_seconds is not None and (
            type(deadline_unix_seconds) not in (int, float)
            or not math.isfinite(deadline_unix_seconds)
            or deadline_unix_seconds <= 0
        ):
            raise ValueError("invalid absolute deadline")
        self.deadline_unix_seconds = deadline_unix_seconds
        self.real_config = real_config
        self.identity = real_config.expected_identity
        self.input_root = safe_path(input_root, input_root.absolute())
        self.output_root = safe_path(output_root, output_root.absolute())
        self.runtime_manifest = safe_path(runtime_manifest.parent, runtime_manifest.absolute())
        if not self.input_root.is_dir() or not self.output_root.is_dir():
            raise ValueError("input and output roots must exist")
        if self.input_root.is_relative_to(self.output_root) or self.output_root.is_relative_to(
            self.input_root
        ):
            raise ValueError("input/output roots overlap")
        self.model_root = safe_path(self.input_root, "model")
        self.images_root = safe_path(self.input_root, "images")
        self._model = None
        self._processor = None
        self._adapter_identity: tuple[Path, tuple[FileEntry, ...], dict[str, str]] | None = None
        self.telemetry = None
        self._check_recipe()
        WorkerManifest.model_validate(strict_json(self.runtime_manifest.read_bytes()))

    def _deadline(self):
        if self.deadline_unix_seconds is not None and time.time() >= self.deadline_unix_seconds:
            raise TimeoutError("absolute run deadline reached")

    def _check_recipe(self):
        cfg = self.real_config
        expected = dict(
            backend=self.backend,
            recipe_version=RECIPE,
            model_repository=REPOSITORY,
            processor_repository=REPOSITORY,
            model_revision=REVISION,
            processor_revision=REVISION,
            training_policy_id=TRAIN_POLICY,
            decode_policy_id=DECODE_POLICY,
            evaluator_id="page-text-nfc-v1",
        )
        if any(getattr(cfg, k) != v for k, v in expected.items()):
            raise ValueError("unsupported Qwen recipe")
        if not self.identity.code_bundle_sha256 or not self.identity.remote_dependency_sha256:
            raise ValueError("worker requires explicit code and remote package identities")

    def _verify_runtime(self):
        self._check_recipe()
        manifest = WorkerManifest.model_validate(strict_json(self.runtime_manifest.read_bytes()))
        root = self.runtime_manifest.parent
        files = [f.model_dump() for f in manifest.files]
        names = [f.filename for f in manifest.files]
        if (
            len(names) != len(set(names))
            or names != sorted(names)
            or any(not n.endswith(".py") for n in names)
        ):
            raise ValueError("invalid code allowlist")
        actual_paths = {safe_path(root, name) for name in names}
        import active_ocr
        import active_ocr.integrations
        import active_ocr.models

        for module_path in (
            __file__,
            active_ocr.__file__,
            active_ocr.models.__file__,
            active_ocr.integrations.__file__,
        ):
            if Path(module_path).absolute() not in actual_paths:
                raise ValueError("executing module absent from code allowlist")
        code = {"source_sha": manifest.source_sha, "files": files}
        if (
            manifest.source_sha != self.identity.source_sha
            or digest(code) != self.identity.code_bundle_sha256
            or inventory(root, names) != files
        ):
            raise ValueError("worker code identity mismatch")
        observed = installed_packages()
        if (
            observed != manifest.environment
            or digest(observed.model_dump(mode="json")) != self.identity.remote_dependency_sha256
            or manifest.build_spec_sha256 != self.identity.build_spec_sha256
            or manifest.deployment_reference != self.identity.deployment_reference
        ):
            raise ValueError("worker package/build identity mismatch")
        packages = dict(observed.packages)
        if (
            platform.system() != "Linux"
            or platform.machine() != "x86_64"
            or not observed.python.startswith("3.11.")
            or any(packages.get(k) != v for k, v in PINS.items())
        ):
            raise ValueError("unsupported pinned Linux environment")
        self._verify_assets()

    def _verify_assets(self):
        for names, expected in (
            (BASE_FILES, self.identity.model_manifest_sha256),
            (PROCESSOR_FILES, self.identity.processor_manifest_sha256),
        ):
            manifest = asset_manifest(names)
            if (
                digest(manifest) != expected
                or inventory(self.model_root, names) != manifest["files"]
            ):
                raise ValueError("base/processor asset identity mismatch")

    def _binding(self):
        return {
            "recipe": self.real_config.model_dump(mode="json"),
            "prompt": PROMPT,
            "target_format": "ordered-regions-original-1000-escaped-less-than-v1",
            "limits": {"prompt": 2048, "target": 4096, "total": 6144, "new": 2048},
            "training": {
                "epochs": 3,
                "accumulation": 4,
                "rank": 16,
                "alpha": 32,
                "learning_rate": 1e-4,
                "targets": list(TARGETS),
            },
            "base": asset_manifest(BASE_FILES),
            "processor": asset_manifest(PROCESSOR_FILES),
        }

    @staticmethod
    def _ownership(experiment_id, round_number):
        if (
            not isinstance(experiment_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", experiment_id) is None
            or type(round_number) is not int
            or round_number < 0
        ):
            raise ValueError("invalid run/round ownership")

    def _images(self, pages, split):
        if len({p.id for p in pages}) != len(pages):
            raise ValueError("duplicate pages")
        images = {}
        for page in pages:
            if page.split is not split:
                raise ValueError("wrong page split")
            path = safe_path(self.images_root, page.image_uri)
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != page.image_sha256:
                raise ValueError("image bytes changed")
            with Image.open(io.BytesIO(raw)) as image:
                if (
                    image.size != (page.width, page.height)
                    or image.mode != "RGB"
                    or image.getexif().get(274, 1) != 1
                ):
                    raise ValueError("image dimension/mode/orientation mismatch")
                image.load()
                resize_geometry(page.height, page.width)
                images[page.id] = image.copy()
        return images

    def _processor_ready(self):
        if self._processor is None:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            self._processor = load_processor(self.model_root)
        return self._processor

    def _drop(self):
        if self._model is not None:
            self._model = None
            gc.collect()
            import torch

            torch.cuda.empty_cache()

    def _bind_adapter(self, adapter: Path, files: tuple[FileEntry, ...]):
        # One synchronous operation owns this snapshot; never derive a replacement on load.
        self._adapter_identity = None
        verify_adapter_files(adapter, files)
        tensors = check_adapter(adapter)
        verify_adapter_files(adapter, files)
        self._adapter_identity = (adapter, files, tensors)

    def _load(self, adapter: Path | None = None):
        self._deadline()
        import torch
        from peft import PeftModel, get_peft_model_state_dict
        from transformers import Qwen3VLForConditionalGeneration

        identity = self._adapter_identity
        if adapter is not None:
            if identity is None or identity[0] != adapter:
                raise ValueError("adapter load requires its verified identity")
            _, files, expected = identity
            verify_adapter_files(adapter, files)
        else:
            self._adapter_identity = None
        self._drop()
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(
            including_emulation=False
        ):
            raise RuntimeError("CUDA BF16 is required; no CPU fallback")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_root,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        check_base_modules(model)
        if any(
            p.device != torch.device("cuda:0") or p.dtype != torch.bfloat16
            for p in model.parameters()
        ):
            raise RuntimeError("base is not entirely BF16 on cuda:0")
        model.requires_grad_(False)
        if adapter is not None:
            model = PeftModel.from_pretrained(
                model,
                adapter,
                is_trainable=False,
                autocast_adapter_dtype=True,
                local_files_only=True,
            )
            actual = {k: tensor_hash(v) for k, v in get_peft_model_state_dict(model).items()}
            if actual != expected:
                raise ValueError("loaded adapter values differ from verified file")
            verify_adapter_files(adapter, files)
        self._model = model
        return model

    def _checkpoint(self, model_id: str, experiment_id: str, round_number: int):
        if re.fullmatch(r"checkpoint:sha256:[0-9a-f]{64}", model_id) is None:
            raise ValueError("invalid checkpoint reference")
        directory = safe_path(self.output_root, "checkpoints/" + model_id.rsplit(":", 1)[1])
        raw = safe_path(directory, "manifest.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != model_id.rsplit(":", 1)[1]:
            raise ValueError("checkpoint digest mismatch")
        checkpoint = Checkpoint.model_validate(strict_json(raw))
        if (
            raw != canonical(checkpoint.model_dump(mode="json"))
            or checkpoint.binding != self._binding()
        ):
            raise ValueError("checkpoint recipe/manifest mismatch")
        verify_checkpoint_files(directory, checkpoint, raw)
        if checkpoint.kind == "base":
            if checkpoint != Checkpoint(kind="base", binding=self._binding()):
                raise ValueError("unexpected base metadata")
            if round_number != 0:
                raise ValueError("positive round requires its fitted adapter")
        else:
            if (
                round_number < 1
                or checkpoint.experiment_id != experiment_id
                or checkpoint.round_number != round_number
                or not checkpoint.selected
                or len({p[0] for p in checkpoint.selected}) != len(checkpoint.selected)
                or checkpoint.target_sha256 is None
                or checkpoint.seed is None
                or not 0 <= checkpoint.seed < 2**32
                or checkpoint.training is None
                or [f.filename for f in checkpoint.files]
                != ["adapter_config.json", "adapter_model.safetensors"]
            ):
                raise ValueError("adapter ownership/training metadata mismatch")
            check_training_record(checkpoint)
            self._bind_adapter(directory, checkpoint.files)
            verify_checkpoint_files(directory, checkpoint, raw)
        return checkpoint, directory

    def load_base(self, *, experiment_id: str) -> str:
        started = time.monotonic()
        self._ownership(experiment_id, 0)
        self._verify_runtime()
        self._processor_ready()
        self._load()
        self._verify_runtime()
        self._observe(started)
        return publish_checkpoint(
            self.output_root, Checkpoint(kind="base", binding=self._binding())
        )

    def predict(
        self,
        pages: tuple[SimulationPage, ...],
        *,
        experiment_id: str,
        round_number: int,
        model_id: str,
        purpose: PredictionPurpose = PredictionPurpose.POOL,
    ) -> tuple[Prediction, ...]:
        started = time.monotonic()
        self._ownership(experiment_id, round_number)
        purpose = PredictionPurpose(purpose)
        if (round_number == 0) != (purpose is PredictionPurpose.BASELINE_VALIDATION):
            raise ValueError("purpose/round mismatch")
        self._verify_runtime()
        checkpoint, directory = self._checkpoint(model_id, experiment_id, round_number)
        split = Split.TRAIN if purpose is PredictionPurpose.POOL else Split.VALIDATION
        images = self._images(pages, split)
        if not pages:
            return ()
        processor = self._processor_ready()
        prepared = [encode_page(processor, images[p.id]) for p in pages]
        manifest_bytes = canonical(checkpoint.model_dump(mode="json"))
        verify_checkpoint_files(directory, checkpoint, manifest_bytes)
        model = self._load(directory if checkpoint.kind == "adapter" else None)
        verify_checkpoint_files(directory, checkpoint, manifest_bytes)
        import torch

        model.eval()
        results, evidence = [], []
        for page, (batch, receipt) in zip(pages, prepared, strict=True):
            self._deadline()
            inputs = {k: v.to("cuda:0") for k, v in batch.items()}
            verify_checkpoint_files(directory, checkpoint, manifest_bytes)
            with torch.inference_mode(), cuda_context():
                generated = model.generate(**inputs, generation_config=generation_config())
            prefix = inputs["input_ids"].shape[1]
            if not torch.equal(generated[:, :prefix], inputs["input_ids"]):
                raise RuntimeError("generated prompt prefix changed")
            ids = generated[0, prefix:].tolist()
            status, regions, raw, reason = decode_result(
                ids, processor.tokenizer, page.width, page.height
            )
            evidence.append(
                {
                    "page_id": page.id,
                    "image_sha256": page.image_sha256,
                    "ids": ids,
                    "text": raw,
                    "finish_reason": reason,
                    "status": status.value,
                    "processing": receipt,
                }
            )
            results.append(
                dict(
                    page_id=page.id,
                    experiment_id=experiment_id,
                    round_number=round_number,
                    model_id=model_id,
                    purpose=purpose,
                    regions=regions,
                    status=status,
                    finish_reason=reason,
                )
            )
        self._verify_runtime()
        self._images(pages, split)
        self._observe(started)
        verify_checkpoint_files(directory, checkpoint, manifest_bytes)
        raw_ref = self._receipt(
            {
                "operation": "predict",
                "experiment_id": experiment_id,
                "round_number": round_number,
                "purpose": purpose.value,
                "model_id": model_id,
                "pages": evidence,
                "telemetry": self.telemetry.model_dump(mode="json"),
            }
        )
        return tuple(Prediction(**r, raw_output_artifact=raw_ref) for r in results)

    def _observe(self, started: float):
        import torch

        driver = None
        with suppress(OSError, subprocess.SubprocessError):
            driver = (
                subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "--id=0"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
                or None
            )
        self.telemetry = ExecutionTelemetry(
            device=torch.cuda.get_device_name(0),
            cuda_version=torch.version.cuda,
            driver_version=driver,
            elapsed_seconds=time.monotonic() - started,
            free_memory_bytes=torch.cuda.mem_get_info(0)[0],
            peak_memory_bytes=torch.cuda.max_memory_allocated(0),
        )

    def _receipt(self, record: dict) -> str:
        data = canonical(record)
        path = safe_path(self.output_root, "receipts/" + hashlib.sha256(data).hexdigest() + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError("receipt collision")
        else:
            with path.open("xb") as stream:
                stream.write(data)
        return str(path)

    def fit(
        self,
        examples: tuple[RevealedExample, ...],
        *,
        seed: int,
        experiment_id: str,
        round_number: int,
        external_reload: bool = False,
    ) -> str:
        if type(external_reload) is not bool:
            raise ValueError("external_reload must be boolean")
        self._deadline()
        started = time.monotonic()
        self._ownership(experiment_id, round_number)
        if round_number < 1 or type(seed) is not int or not 0 <= seed < 2**32 or not examples:
            raise ValueError("fit requires positive round, uint32 seed and selected examples")
        self._verify_runtime()
        pages = tuple(e.page for e in examples)
        images = self._images(pages, Split.TRAIN)
        targets = {e.page.id: serialize_target(e) for e in examples}
        processor = self._processor_ready()
        # Preflight EVERY selected target before a base load or the first optimizer update.
        prepared = {p.id: encode_page(processor, images[p.id], targets[p.id]) for p in pages}
        self._drop()
        reset_seed(seed)
        base = self._load()
        from peft import get_peft_model, get_peft_model_state_dict

        model = get_peft_model(base, lora_config(), autocast_adapter_dtype=True)
        del base
        self._model = model
        parameters = check_trainables(model)
        before_frozen = frozen_hashes(model)
        initial = {k: tensor_hash(v) for k, v in get_peft_model_state_dict(model).items()}
        batches = {
            key: {k: v.to("cuda:0") for k, v in batch.items()}
            for key, (batch, _) in prepared.items()
        }
        try:
            training = train_epochs(model, batches, seed, parameters, cuda_context, self._deadline)
            final = {k: tensor_hash(v) for k, v in get_peft_model_state_dict(model).items()}
            if initial == final or frozen_hashes(model) != before_frozen:
                raise RuntimeError("adapter unchanged or frozen base changed")
            if training["updates"] != 3 * math.ceil(len(examples) / 4):
                raise RuntimeError("incorrect update count")
            training.update(
                processing={key: receipt for key, (_, receipt) in prepared.items()},
                supervised_tokens=3 * sum(r["target_tokens"] for _, r in prepared.values()),
                changed_tensors=sum(final[k] != initial[k] for k in final),
                frozen_sha256=digest(before_frozen),
            )
            # Save in a private staging directory, validate, reload, then publish manifest last.
            with tempfile.TemporaryDirectory(prefix=".qwen-fit-", dir=self.output_root) as temp:
                stage = Path(temp)
                model.save_pretrained(stage, safe_serialization=True, save_embedding_layers=False)
                cfg_path = stage / "adapter_config.json"
                config = strict_json(cfg_path.read_bytes())
                config.update(base_model_name_or_path=REPOSITORY, revision=REVISION)
                cfg_path.write_bytes(canonical(config))
                adapter_files = tuple(
                    FileEntry(**f)
                    for f in inventory(stage, ("adapter_config.json", "adapter_model.safetensors"))
                )
                self._bind_adapter(stage, adapter_files)
                probe_id = min(targets)
                probe_inputs = {
                    k: v.to("cuda:0")
                    for k, v in encode_page(processor, images[probe_id])[0].items()
                }
                self._deadline()
                before_probe = reload_probe(
                    model, probe_inputs, processor.tokenizer, original_size=images[probe_id].size
                )
                del model, parameters, batches
                self._drop()
                difference = None
                if not external_reload:
                    reloaded = self._load(stage)
                    self._deadline()
                    after_probe = reload_probe(
                        reloaded,
                        probe_inputs,
                        processor.tokenizer,
                        original_size=images[probe_id].size,
                        fixed_ids=before_probe["ids"],
                    )
                    difference = compare_probes(before_probe, after_probe)
                self._verify_runtime()
                self._images(pages, Split.TRAIN)
                checkpoint = Checkpoint(
                    kind="adapter",
                    binding=self._binding(),
                    experiment_id=experiment_id,
                    round_number=round_number,
                    selected=tuple((p.id, p.image_sha256) for p in pages),
                    target_sha256=digest([(p.id, targets[p.id]) for p in pages]),
                    seed=seed,
                    training=training,
                    files=adapter_files,
                )
                check_training_record(checkpoint)
                self._observe(started)
                verify_adapter_files(stage, adapter_files)
                model_id = "checkpoint:sha256:" + digest(checkpoint.model_dump(mode="json"))
                self._receipt(
                    {
                        "operation": "fit",
                        "model_id": model_id,
                        "probe_page": probe_id,
                        "telemetry": self.telemetry.model_dump(mode="json"),
                        "reload": "pending-fresh-process"
                        if external_reload
                        else "fresh-model-same-process",
                        "max_logit_difference": difference,
                        "probe": {k: v for k, v in before_probe.items() if k != "logits"},
                    }
                )
                result = publish_checkpoint(self.output_root, checkpoint, stage)
                if external_reload:
                    probe_page = next(p for p in pages if p.id == probe_id)
                    self._write_probe(result, probe_page, before_probe, "before")
                return result
        except BaseException:
            self._drop()
            raise

    def _write_probe(self, model_id, page, probe, name):
        """Nested logits keys are relative to this Qwen output root, never the run root."""
        rows = min(8, len(probe["ids"]))
        if tuple(probe["logits"].shape) != (rows, 151936):
            raise ValueError("probe logits shape mismatch")
        values = array("f", probe["logits"].reshape(-1).tolist())
        if any(not math.isfinite(v) for v in values):
            raise ValueError("nonfinite probe logits")
        if sys.byteorder != "little":
            values.byteswap()
        key = "probes/" + model_id.rsplit(":", 1)[1] + "/" + name
        data = values.tobytes()
        metadata = dict(
            schema_version=1,
            model_id=model_id,
            page_id=page.id,
            image_sha256=page.image_sha256,
            original_width=page.width,
            original_height=page.height,
            process_id=os.getpid(),
            generated_ids=probe["ids"],
            status=probe["status"],
            regions=probe["regions"],
            finish_reason=probe["finish_reason"],
            adapter_tensors=probe["tensors"],
            logits=dict(
                key=key + ".f32le", bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
            ),
            logits_shape=[rows, 151936],
        )
        for suffix, content in ((".f32le", data), (".json", canonical(metadata))):
            path = safe_path(self.output_root, key + suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        return metadata

    def verify_reload(
        self,
        page: SimulationPage,
        *,
        experiment_id: str,
        round_number: int,
        model_id: str,
        before_metadata: Path,
        before_metadata_sha256: str,
        before_logits: Path,
        before_logits_sha256: str,
    ) -> dict:
        """One probe in a fresh interpreter; targets are neither accepted nor reconstructed."""
        self._deadline()
        self._ownership(experiment_id, round_number)
        self._verify_runtime()
        checkpoint, directory = self._checkpoint(model_id, experiment_id, round_number)
        if checkpoint.kind != "adapter" or (page.id, page.image_sha256) not in checkpoint.selected:
            raise ValueError("reload probe requires a selected training page")
        if page.id != min(p[0] for p in checkpoint.selected):
            raise ValueError("reload probe requires lexicographically first selected page")
        images = self._images((page,), Split.TRAIN)
        key = "probes/" + model_id.rsplit(":", 1)[1] + "/before"
        for path, suffix, expected in (
            (before_metadata, ".json", before_metadata_sha256),
            (before_logits, ".f32le", before_logits_sha256),
        ):
            if safe_path(self.output_root, path.absolute()) != safe_path(
                self.output_root, key + suffix
            ):
                raise ValueError("unexpected probe evidence path")
            if file_hash(path) != expected:
                raise ValueError("probe evidence digest mismatch")
        raw = before_metadata.read_bytes()
        before = strict_json(raw)
        ids = before["generated_ids"]
        rows = min(8, len(ids))
        if (
            raw != canonical(before)
            or before["schema_version"] != 1
            or before["model_id"] != model_id
            or before["page_id"] != page.id
            or before["image_sha256"] != page.image_sha256
            or (before["original_width"], before["original_height"]) != (page.width, page.height)
            or type(before["process_id"]) is not int
            or before["process_id"] <= 0
            or before["process_id"] == os.getpid()
            or not 1 <= len(ids) <= 2048
            or any(type(i) is not int or i < 0 for i in ids)
            or before["logits_shape"] != [rows, 151936]
            or before["logits"]
            != dict(key=key + ".f32le", bytes=rows * 151936 * 4, sha256=before_logits_sha256)
            or before_logits.stat().st_size != rows * 151936 * 4
            or set(before["adapter_tensors"]) != set(adapter_shapes())
        ):
            raise ValueError("probe metadata identity/shape mismatch")
        values = array("f")
        values.frombytes(before_logits.read_bytes())
        if sys.byteorder != "little":
            values.byteswap()
        if any(not math.isfinite(v) for v in values):
            raise ValueError("nonfinite before logits")
        import torch

        processor = self._processor_ready()
        inputs = {k: v.to("cuda:0") for k, v in encode_page(processor, images[page.id])[0].items()}
        model = self._load(directory)
        self._deadline()
        after = reload_probe(
            model, inputs, processor.tokenizer, original_size=images[page.id].size, fixed_ids=ids
        )
        # Retain the failed after diagnostic as well; only successful comparison returns.
        self._write_probe(model_id, page, after, "after")
        difference = compare_probes(
            dict(
                ids=ids,
                status=before["status"],
                regions=before["regions"],
                finish_reason=before["finish_reason"],
                tensors=before["adapter_tensors"],
                logits=torch.tensor(values).reshape(rows, 151936),
            ),
            after,
        )
        self._verify_runtime()
        self._images((page,), Split.TRAIN)
        verify_checkpoint_files(
            directory, checkpoint, canonical(checkpoint.model_dump(mode="json"))
        )
        self._deadline()
        return dict(
            max_absolute_difference=difference,
            train_process_id=before["process_id"],
            reload_process_id=os.getpid(),
        )
