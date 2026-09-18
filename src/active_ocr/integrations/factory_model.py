"""Thin LLaMA-Factory adapter for full-page joint box-and-text OCR.

Training and inference belong to the toolkit: `llamafactory-cli train <resolved.yaml>` performs
LoRA SFT and `llamafactory.chat.ChatModel` with the native HuggingFace backend generates. This
module only converts revealed examples into upstream rows, renders one YAML from the single
validated recipe, invokes the toolkit and maps raw responses onto the existing contracts.

GPU libraries are imported on use, so the coordinator and CLI can import this module without
them. No oracle, selection, transport or automatic download lives here.
"""

from __future__ import annotations

import gc
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
from contextlib import suppress
from functools import cache
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import Field

from active_ocr.integrations.artifacts import (
    FileEntry,
    WorkerManifest,
    canonical,
    digest,
    file_hash,
    installed_packages,
    inventory,
    safe_path,
    strict_json,
    verify_files,
)
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

CPU_TESTS = (
    "test_factory_model::test_actual_template_is_the_pinned_nothink_variant",
    "test_factory_model::test_actual_template_supervises_the_whole_target_and_masks_the_prompt",
    "test_factory_model::test_actual_encoded_lengths_are_measured_and_overlong_targets_fail",
    "test_factory_model::test_actual_native_lora_starts_with_zero_B_and_changes_after_training",
    "test_factory_model::test_actual_native_cli_trains_and_fresh_process_reloads",
)

RECIPE_NAME = "read2016-llamafactory-joint"
IMAGE_PLACEHOLDER = "<image>"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
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


def recipe_path() -> Path:
    """Locate the one validated recipe: explicit override, checkout, then deployed image."""
    override = os.environ.get("ACTIVE_OCR_RECIPE_FILE")
    if override:
        return Path(override)
    module = Path(__file__).resolve()
    # src/active_ocr/integrations/ in a checkout, /opt/ocr/active_ocr/integrations/ in the image.
    candidates = (
        module.parents[1] / "recipes" / f"{RECIPE_NAME}.json",
        module.parents[3] / "experiments" / "recipes" / f"{RECIPE_NAME}.json",
        module.parents[2] / "experiments" / "recipes" / f"{RECIPE_NAME}.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"cannot locate recipe {RECIPE_NAME}.json")


@cache
def recipe() -> dict:
    """Parse the single source of experiment settings; the YAML is rendered from this."""
    value = strict_json(recipe_path().read_bytes())
    if value.get("schema_version") != 1 or value.get("toolkit", {}).get("version") != "0.9.5":
        raise ValueError("unsupported recipe schema or toolkit version")
    missing = {"assets", "task", "limits", "image", "training", "decode"} - set(value)
    if missing:
        raise ValueError(f"recipe is missing sections: {sorted(missing)}")
    return value


def recipe_field(*keys: str) -> Any:
    value: Any = recipe()
    for key in keys:
        value = value[key]
    return value


BACKEND = "llamafactory-qwen3vl-v1"
BASE_FILES = tuple(recipe_field("assets", "base_files"))
PROCESSOR_FILES = tuple(recipe_field("assets", "processor_files"))
REPOSITORY = recipe_field("assets", "repository")
REVISION = recipe_field("assets", "revision")
PROMPT = recipe_field("task", "prompt")
TEMPLATE = recipe_field("toolkit", "template")
MAX_PROMPT_TOKENS = recipe_field("limits", "max_prompt_tokens")
MAX_OUTPUT_TOKENS = recipe_field("limits", "max_output_tokens")
MAX_SEQUENCE_TOKENS = recipe_field("limits", "max_sequence_tokens")


def binding(real_config: RealOCRConfig) -> dict:
    """The full resolved recipe travels with every checkpoint, so both sides must agree.

    This is a pure function of the frozen configuration and the local recipe file, so the
    coordinator can reproduce it without constructing a model or touching a GPU.
    """
    return {
        "recipe": real_config.model_dump(mode="json"),
        "settings": recipe(),
        "base": asset_manifest(BASE_FILES),
        "processor": asset_manifest(PROCESSOR_FILES),
    }


def asset_manifest(names) -> dict:
    return {
        "schema_version": 1,
        "repository": REPOSITORY,
        "revision": REVISION,
        "files": [dict(filename=n, bytes=ASSETS[n][0], sha256=ASSETS[n][1]) for n in sorted(names)],
    }


def serialize_target(example: RevealedExample) -> str:
    """Literal JSON target. Escaping `<` keeps `<image>` placeholders out of the response."""
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
    """Ordinary strict parsing plus semantic validation; never repair a malformed response."""
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


def conversation_row(image_path: Path, target: str | None) -> dict:
    """One upstream sharegpt multimodal row; only revealed targets ever reach the trainer."""
    conversations = [{"from": "human", "value": IMAGE_PLACEHOLDER + PROMPT}]
    if target is not None:
        conversations.append({"from": "gpt", "value": target})
    return {"conversations": conversations, "images": [str(image_path)]}


def write_dataset(directory: Path, rows: list[dict], name: str = "selected_train") -> Path:
    """Export train.jsonl and a minimal dataset_info.json; upstream package files are untouched."""
    if not rows:
        raise ValueError("training dataset requires at least one revealed example")
    directory.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    (directory / "train.jsonl").write_text(data, encoding="utf-8")
    (directory / "dataset_info.json").write_text(
        json.dumps(
            {
                name: {
                    "file_name": "train.jsonl",
                    "formatting": "sharegpt",
                    "columns": {"messages": "conversations", "images": "images"},
                    "tags": {
                        "role_tag": "from",
                        "content_tag": "value",
                        "user_tag": "human",
                        "assistant_tag": "gpt",
                    },
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return directory / "train.jsonl"


def training_yaml(model_root: Path, dataset_dir: Path, output_dir: Path, seed: int) -> dict:
    """Render the native configuration from the recipe; no second hand-maintained config."""
    training = recipe_field("training")
    image = recipe_field("image")
    return {
        "model_name_or_path": str(model_root),
        "trust_remote_code": False,
        "image_max_pixels": image["image_max_pixels"],
        "image_min_pixels": image["image_min_pixels"],
        "stage": training["stage"],
        "do_train": True,
        "finetuning_type": training["finetuning_type"],
        "lora_rank": training["lora_rank"],
        "lora_alpha": training["lora_alpha"],
        "lora_dropout": training["lora_dropout"],
        "lora_target": training["lora_target"],
        "freeze_vision_tower": training["freeze_vision_tower"],
        "dataset": "selected_train",
        "dataset_dir": str(dataset_dir),
        "template": TEMPLATE,
        # The upstream demo truncates at 2,048; every target is preflighted against this limit.
        "cutoff_len": MAX_SEQUENCE_TOKENS,
        "overwrite_cache": True,
        "preprocessing_num_workers": 1,
        "dataloader_num_workers": 0,
        "packing": training["packing"],
        "val_size": training["val_size"],
        "output_dir": str(output_dir),
        "overwrite_output_dir": False,
        "logging_steps": 1,
        "save_strategy": "no",
        "save_only_model": True,
        "plot_loss": False,
        "report_to": "none",
        "per_device_train_batch_size": training["per_device_train_batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "learning_rate": training["learning_rate"],
        "num_train_epochs": training["num_train_epochs"],
        "optim": training["optim"],
        "lr_scheduler_type": training["lr_scheduler_type"],
        "warmup_ratio": training["warmup_ratio"],
        "bf16": training["bf16"],
        "gradient_checkpointing": training["gradient_checkpointing"],
        "seed": seed,
        "data_seed": seed,
    }


def adapter_summary(path: Path) -> dict[str, Any]:
    """Read safetensors framing only; never deserialize an executable model object."""
    import struct

    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError("truncated adapter")
    size = struct.unpack("<Q", raw[:8])[0]
    if not 2 <= size <= 4 * 1024 * 1024 or size + 8 > len(raw):
        raise ValueError("invalid adapter header")
    header = strict_json(raw[8 : 8 + size])
    header.pop("__metadata__", None)
    if not header:
        raise ValueError("adapter contains no tensors")
    tensors, nonzero, changed_b, cursor = {}, 0, 0, 0
    ranges = []
    for name in sorted(header):
        entry = header[name]
        start, end = entry["data_offsets"]
        if (
            type(start) is not int
            or type(end) is not int
            or start < 0
            or end < start
            or end > len(raw) - size - 8
        ):
            raise ValueError("invalid adapter tensor framing")
        ranges.append((start, end))
        body = raw[8 + size + start : 8 + size + end]
        if len(body) != end - start:
            raise ValueError("adapter tensor body truncated")
        dtype = entry["dtype"]
        formats = {"F32": (4, "f"), "F16": (2, "e"), "BF16": (2, "H")}
        if dtype not in formats:
            raise ValueError("adapter must contain floating point tensors")
        width, code = formats[dtype]
        shape = entry["shape"]
        if any(type(n) is not int or n <= 0 for n in shape) or math.prod(shape) * width != len(
            body
        ):
            raise ValueError("adapter tensor size mismatch")
        values = (v[0] for v in struct.iter_unpack("<" + code, body))
        if dtype == "BF16":
            values = (struct.unpack("<f", struct.pack("<I", v << 16))[0] for v in values)
        nonzero_tensor = False
        for value in values:
            if not math.isfinite(value):
                raise ValueError("nonfinite adapter tensor")
            nonzero_tensor |= value != 0
        nonzero += nonzero_tensor
        if name.endswith(".lora_B.weight"):
            changed_b += nonzero_tensor
        tensors[name] = {
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        cursor = max(cursor, end)
    previous = 0
    for start, end in sorted(ranges):
        if start != previous:
            raise ValueError("overlapping or missing adapter bytes")
        previous = end
    if cursor != len(raw) - size - 8:
        raise ValueError("unreferenced adapter bytes")
    return {"tensors": tensors, "nonzero_tensors": nonzero, "changed_lora_B_tensors": changed_b}


def verify_adapter_config(config: dict) -> None:
    """The learned-B proof requires the native default (zero-B) initialization."""
    expected = {
        "peft_type": "LORA",
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "init_lora_weights": True,
        "use_dora": False,
        "use_rslora": False,
        "bias": "none",
        "modules_to_save": None,
    }
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("adapter differs from the native default-initialized recipe")


def verify_adapter(directory: Path, summary: dict | None = None) -> dict:
    """Default PEFT LoRA B starts at zero. Require finite learned B and native config."""
    verify_adapter_config(strict_json((directory / "adapter_config.json").read_bytes()))
    summary = summary or adapter_summary(directory / "adapter_model.safetensors")
    if not summary["changed_lora_B_tensors"]:
        raise ValueError("saved LoRA B is zero; no learned adapter change")
    return summary


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
    """Validate the recorded upstream summary structurally; the toolkit owns optimization."""
    training = checkpoint.training
    keys = {
        "toolkit_version",
        "examples",
        "epochs",
        "optimizer_steps",
        "losses",
        "train_runtime_seconds",
        "preflight",
        "changed_lora_B_tensors",
    }
    if training is None or set(training) != keys:
        raise ValueError("invalid training record")
    ids = tuple(p[0] for p in checkpoint.selected)
    if (
        training["toolkit_version"] != recipe_field("toolkit", "version")
        or training["examples"] != len(ids)
        or training["epochs"] != recipe_field("training", "num_train_epochs")
        or type(training["optimizer_steps"]) is not int
        or not 1 <= training["optimizer_steps"] <= 1000
        or not training["losses"]
        or type(training["changed_lora_B_tensors"]) is not int
        or training["changed_lora_B_tensors"] < 1
    ):
        raise ValueError("training count/identity mismatch")
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or v < 0
        for v in (*training["losses"], training["train_runtime_seconds"])
    ):
        raise ValueError("invalid loss/runtime diagnostics")
    preflight = training["preflight"]
    if set(preflight) != set(ids):
        raise ValueError("preflight coverage mismatch")
    for record in preflight.values():
        if set(record) != {"prompt_tokens", "target_tokens", "total_tokens"}:
            raise ValueError("invalid preflight record")
        p, t, total = (record[k] for k in ("prompt_tokens", "target_tokens", "total_tokens"))
        if (
            any(type(v) is not int for v in (p, t, total))
            or not 1 <= p <= MAX_PROMPT_TOKENS
            or not 1 <= t <= MAX_OUTPUT_TOKENS
            or total != p + t
            or total > MAX_SEQUENCE_TOKENS
        ):
            raise ValueError("preflight exceeds the declared token limits")


def verify_checkpoint_files(directory: Path, checkpoint: Checkpoint, data: bytes) -> None:
    actual_names = {p.name for p in directory.iterdir()}
    names = [f.filename for f in checkpoint.files]
    if len(set(names)) != len(names) or actual_names != {*names, "manifest.json"}:
        raise ValueError("checkpoint file inventory mismatch")
    if safe_path(directory, "manifest.json").read_bytes() != data:
        raise ValueError("checkpoint manifest bytes changed")
    verify_files(directory, checkpoint.files)


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


class FactoryModel:
    """Synchronous reset-fit adapter over LLaMA-Factory. Predict never receives labels."""

    backend = BACKEND
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
        self._chat = None
        self._chat_adapter: Path | None = None
        self.telemetry = None
        self._check_recipe()
        WorkerManifest.model_validate(strict_json(self.runtime_manifest.read_bytes()))

    # ---------------------------------------------------------------- identity

    def _deadline(self):
        if self.deadline_unix_seconds is not None and time.time() >= self.deadline_unix_seconds:
            raise TimeoutError("absolute run deadline reached")

    def _check_recipe(self):
        cfg = self.real_config
        expected = dict(
            backend=self.backend,
            recipe_version=recipe_field("recipe_version"),
            model_repository=REPOSITORY,
            processor_repository=REPOSITORY,
            model_revision=REVISION,
            processor_revision=REVISION,
            training_policy_id=recipe_field("training_policy_id"),
            decode_policy_id=recipe_field("decode_policy_id"),
            evaluator_id=recipe_field("evaluator_id"),
        )
        if any(getattr(cfg, k) != v for k, v in expected.items()):
            raise ValueError("unsupported LLaMA-Factory recipe")
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
        import platform

        if (
            platform.system() != "Linux"
            or platform.machine() != "x86_64"
            or not observed.python.startswith("3.11.")
            or packages.get("llamafactory") != recipe_field("toolkit", "version")
        ):
            raise ValueError("unsupported pinned Linux toolkit environment")
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

    def _binding(self) -> dict:
        return binding(self.real_config)

    @staticmethod
    def _ownership(experiment_id, round_number):
        if (
            not isinstance(experiment_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", experiment_id) is None
            or type(round_number) is not int
            or round_number < 0
        ):
            raise ValueError("invalid run/round ownership")

    def _images(self, pages, split) -> dict[str, Path]:
        """Return verified image paths; the toolkit performs all pixel processing."""
        if len({p.id for p in pages}) != len(pages):
            raise ValueError("duplicate pages")
        paths = {}
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
            paths[page.id] = path
        return paths

    # ------------------------------------------------------------ toolkit use

    def _infer_args(self, adapter: Path | None) -> dict:
        decode = recipe_field("decode")
        image = recipe_field("image")
        args = {
            "model_name_or_path": str(self.model_root),
            "template": TEMPLATE,
            "infer_backend": "huggingface",
            "finetuning_type": recipe_field("training", "finetuning_type"),
            "trust_remote_code": False,
            "cutoff_len": MAX_SEQUENCE_TOKENS,
            "image_max_pixels": image["image_max_pixels"],
            "image_min_pixels": image["image_min_pixels"],
            "do_sample": decode["do_sample"],
            "temperature": decode["temperature"],
            "top_p": decode["top_p"],
            "top_k": decode["top_k"],
            "repetition_penalty": decode["repetition_penalty"],
            "max_new_tokens": decode["max_new_tokens"],
            "skip_special_tokens": decode["skip_special_tokens"],
        }
        if adapter is not None:
            args["adapter_name_or_path"] = str(adapter)
        return args

    def _release(self):
        if self._chat is not None:
            self._chat = None
            self._chat_adapter = None
            gc.collect()
            with suppress(ImportError):
                import torch

                torch.cuda.empty_cache()

    def _chat_model(self, adapter: Path | None):
        """Reuse one loaded toolkit model across the pages of a single operation."""
        if self._chat is not None and self._chat_adapter == adapter:
            return self._chat
        self._release()
        self._deadline()
        import torch

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(
            including_emulation=False
        ):
            raise RuntimeError("CUDA BF16 is required; no CPU fallback")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from llamafactory.chat import ChatModel

        self._chat = ChatModel(self._infer_args(adapter))
        self._chat_adapter = adapter
        return self._chat

    def _generate(self, chat, image_path: Path, width: int, height: int) -> dict:
        """One greedy full-page generation; raw output and finish reason are always retained."""
        self._deadline()
        responses = chat.chat(
            [{"role": "user", "content": IMAGE_PLACEHOLDER + PROMPT}], images=[str(image_path)]
        )
        if len(responses) != 1:
            raise RuntimeError("expected exactly one greedy response")
        response = responses[0]
        raw, reason = response.response_text, response.finish_reason
        if reason == "length":
            status, regions = PredictionStatus.TRUNCATED, ()
        else:
            try:
                status, regions = PredictionStatus.OK, parse_regions(raw, width, height)
            except (ValueError, UnicodeError, OverflowError):
                status, regions = PredictionStatus.INVALID_OUTPUT, ()
        return {
            "text": raw,
            "finish_reason": reason,
            "status": status,
            "regions": regions,
            "prompt_tokens": int(response.prompt_length),
            "response_tokens": int(response.response_length),
        }

    def _preflight(self, examples, paths, targets) -> dict[str, dict[str, int]]:
        """Measure the actual toolkit-encoded lengths before any optimizer step runs."""
        from llamafactory.data import get_template_and_fix_tokenizer
        from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
        from llamafactory.extras.constants import IGNORE_INDEX
        from llamafactory.hparams import get_infer_args
        from llamafactory.model import load_tokenizer

        model_args, data_args, _, _ = get_infer_args(self._infer_args(None))
        module = load_tokenizer(model_args)
        tokenizer, processor = module["tokenizer"], module["processor"]
        template = get_template_and_fix_tokenizer(tokenizer, data_args)
        encoder = SupervisedDatasetProcessor(
            template=template, tokenizer=tokenizer, processor=processor, data_args=data_args
        )
        records = {}
        for example in examples:
            page = example.page
            messages = [
                {"role": "user", "content": IMAGE_PLACEHOLDER + PROMPT},
                {"role": "assistant", "content": targets[page.id]},
            ]
            messages = template.mm_plugin.process_messages(
                messages, [str(paths[page.id])], [], [], processor
            )
            prompt_ids, response_ids = template.encode_oneturn(tokenizer, messages, None, None)
            prompt, target = len(prompt_ids), len(response_ids)
            if prompt > MAX_PROMPT_TOKENS:
                raise ValueError(f"prompt overflow for {page.id}: {prompt} tokens")
            if target > MAX_OUTPUT_TOKENS:
                raise ValueError(f"target overflow for {page.id}: {target} tokens including EOS")
            if prompt + target > MAX_SEQUENCE_TOKENS:
                raise ValueError(f"sequence overflow for {page.id}: {prompt + target} tokens")
            # Exercise exactly the upstream SFT processor and reject silent truncation or
            # changed supervision, rather than trusting an independent length estimate.
            ids, labels = encoder._encode_data_example(
                prompt=[{"role": "user", "content": IMAGE_PLACEHOLDER + PROMPT}],
                response=[{"role": "assistant", "content": targets[page.id]}],
                system=None,
                tools=None,
                images=[str(paths[page.id])],
                videos=[],
                audios=[],
            )
            if (
                ids != prompt_ids + response_ids
                or labels != [IGNORE_INDEX] * prompt + response_ids
                or not response_ids
                or tokenizer.decode(
                    response_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                != targets[page.id] + "<|im_end|>\n"
            ):
                raise ValueError(f"upstream supervision changed or truncated for {page.id}")
            records[page.id] = {
                "prompt_tokens": prompt,
                "target_tokens": target,
                "total_tokens": prompt + target,
            }
        del template, tokenizer, processor, module
        gc.collect()
        return records

    def _train(self, config: dict, work: Path) -> dict:
        """Invoke the released CLI with an argument array; never through a shell."""
        import yaml

        config_path = work / "resolved.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
        log_path = work / "train.log"
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_DISABLED": "true",
            "PYTHONUNBUFFERED": "1",
        }
        remaining = None
        if self.deadline_unix_seconds is not None:
            remaining = self.deadline_unix_seconds - time.time()
            if remaining <= 0:
                raise TimeoutError("absolute run deadline reached before training")
        with log_path.open("wb") as log:
            # Same entry point as `llamafactory-cli train`, bound to this interpreter.
            completed = subprocess.run(
                [sys.executable, "-m", "llamafactory.cli", "train", str(config_path)],
                cwd=work,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=remaining,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"llamafactory-cli train failed with exit status {completed.returncode}; "
                f"log retained at {log_path}"
            )
        return {"config": config_path, "log": log_path}

    @staticmethod
    def _trainer_summary(output_dir: Path) -> tuple[int, list[float], float]:
        state = strict_json(safe_path(output_dir, "trainer_state.json").read_bytes())
        losses = [float(entry["loss"]) for entry in state.get("log_history", []) if "loss" in entry]
        results_path = output_dir / "all_results.json"
        runtime = 0.0
        if results_path.is_file():
            runtime = float(strict_json(results_path.read_bytes()).get("train_runtime", 0.0))
        return int(state.get("global_step", 0)), losses, runtime

    # ------------------------------------------------------------- operations

    def load_base(self, *, experiment_id: str) -> str:
        started = time.monotonic()
        self._ownership(experiment_id, 0)
        self._verify_runtime()
        # A base checkpoint identifies verified assets; fit loads them in the native CLI.
        self._observe(started)
        return publish_checkpoint(
            self.output_root, Checkpoint(kind="base", binding=self._binding())
        )

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
                or checkpoint.training is None
                or [f.filename for f in checkpoint.files] != list(ADAPTER_FILES)
            ):
                raise ValueError("adapter ownership/training metadata mismatch")
            check_training_record(checkpoint)
            summary = verify_adapter(directory)
            if summary["changed_lora_B_tensors"] != checkpoint.training["changed_lora_B_tensors"]:
                raise ValueError("adapter change evidence mismatch")
        return checkpoint, directory

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
        paths = self._images(pages, split)
        if not pages:
            return ()
        manifest_bytes = canonical(checkpoint.model_dump(mode="json"))
        chat = self._chat_model(directory if checkpoint.kind == "adapter" else None)
        results, evidence = [], []
        for page in pages:
            verify_checkpoint_files(directory, checkpoint, manifest_bytes)
            output = self._generate(chat, paths[page.id], page.width, page.height)
            evidence.append(
                {
                    "page_id": page.id,
                    "image_sha256": page.image_sha256,
                    "text": output["text"],
                    "finish_reason": output["finish_reason"],
                    "status": output["status"].value,
                    "prompt_tokens": output["prompt_tokens"],
                    "response_tokens": output["response_tokens"],
                }
            )
            results.append(
                dict(
                    page_id=page.id,
                    experiment_id=experiment_id,
                    round_number=round_number,
                    model_id=model_id,
                    purpose=purpose,
                    regions=output["regions"],
                    status=output["status"],
                    finish_reason=output["finish_reason"],
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
        paths = self._images(pages, Split.TRAIN)
        targets = {e.page.id: serialize_target(e) for e in examples}
        target_sha = digest([(p.id, targets[p.id]) for p in pages])
        # Preflight EVERY selected target before the trainer starts; never truncate or drop one.
        preflight = self._preflight(examples, paths, targets)
        self._release()
        work = safe_path(
            self.output_root,
            f"fits/{experiment_id}-{round_number}-{digest([target_sha, seed])[:16]}",
        )
        if work.exists():
            raise ValueError("fit output directory already exists; refusing to overwrite")
        work.mkdir(parents=True)
        dataset_dir = work / "data"
        write_dataset(dataset_dir, [conversation_row(paths[p.id], targets[p.id]) for p in pages])
        output_dir = work / "adapter"
        self._train(training_yaml(self.model_root, dataset_dir, output_dir, seed), work)
        self._verify_runtime()
        self._images(pages, Split.TRAIN)
        steps, losses, runtime = self._trainer_summary(output_dir)
        summary = adapter_summary(safe_path(output_dir, "adapter_model.safetensors"))
        verify_adapter(output_dir, summary)
        adapter_files = tuple(FileEntry(**f) for f in inventory(output_dir, ADAPTER_FILES))
        checkpoint = Checkpoint(
            kind="adapter",
            binding=self._binding(),
            experiment_id=experiment_id,
            round_number=round_number,
            selected=tuple((p.id, p.image_sha256) for p in pages),
            target_sha256=target_sha,
            seed=seed,
            training={
                "toolkit_version": recipe_field("toolkit", "version"),
                "examples": len(pages),
                "epochs": recipe_field("training", "num_train_epochs"),
                "optimizer_steps": steps,
                "losses": losses,
                "train_runtime_seconds": runtime,
                "preflight": preflight,
                "changed_lora_B_tensors": summary["changed_lora_B_tensors"],
            },
            files=adapter_files,
        )
        check_training_record(checkpoint)
        self._observe(started, training=True)
        model_id = publish_checkpoint(self.output_root, checkpoint, output_dir)
        probe_page = min(pages, key=lambda p: p.id)
        _, directory = self._checkpoint(model_id, experiment_id, round_number)
        before = self._probe(model_id, probe_page, paths[probe_page.id], directory, "before")
        self._receipt(
            {
                "operation": "fit",
                "model_id": model_id,
                "probe_page": probe_page.id,
                "optimizer_steps": steps,
                "telemetry": self.telemetry.model_dump(mode="json"),
                "reload": "pending-fresh-process" if external_reload else "same-process",
                "probe": before,
            }
        )
        if not external_reload:
            self._release()
            after = self._probe(model_id, probe_page, paths[probe_page.id], directory, "after")
            compare_probes(before, after)
        verify_checkpoint_files(
            directory, checkpoint, canonical(checkpoint.model_dump(mode="json"))
        )
        self._images(pages, Split.TRAIN)
        return model_id

    def _probe(self, model_id, page, image_path: Path, adapter: Path, stage: str) -> dict:
        """Selected TRAIN reload diagnostic. Never an unbiased evaluation result."""
        chat = self._chat_model(adapter)
        output = self._generate(chat, image_path, page.width, page.height)
        record = {
            "schema_version": 1,
            "model_id": model_id,
            "page_id": page.id,
            "image_sha256": page.image_sha256,
            "original_width": page.width,
            "original_height": page.height,
            "process_id": os.getpid(),
            "status": output["status"].value,
            "finish_reason": output["finish_reason"],
            "response_tokens": output["response_tokens"],
            "response_sha256": hashlib.sha256(output["text"].encode("utf-8")).hexdigest(),
            "regions": [r.model_dump(mode="json") for r in output["regions"]],
        }
        path = safe_path(self.output_root, f"probes/{model_id.rsplit(':', 1)[1]}/{stage}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(canonical(record))
            stream.flush()
            os.fsync(stream.fileno())
        return record

    def verify_reload(
        self,
        page: SimulationPage,
        *,
        experiment_id: str,
        round_number: int,
        model_id: str,
        before_metadata: Path,
        before_metadata_sha256: str,
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
        expected = safe_path(self.output_root, f"probes/{model_id.rsplit(':', 1)[1]}/before.json")
        if safe_path(self.output_root, before_metadata.absolute()) != expected:
            raise ValueError("unexpected probe evidence path")
        if file_hash(before_metadata) != before_metadata_sha256:
            raise ValueError("probe evidence digest mismatch")
        raw = before_metadata.read_bytes()
        before = strict_json(raw)
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
        ):
            raise ValueError("probe metadata identity mismatch")
        paths = self._images((page,), Split.TRAIN)
        after = self._probe(model_id, page, paths[page.id], directory, "after")
        compare_probes(before, after, distinct_process=True)
        return dict(
            train_process_id=before["process_id"],
            reload_process_id=after["process_id"],
        )

    # -------------------------------------------------------------- reporting

    def _observe(self, started: float, *, training: bool = False):
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
            # Training ran in the native CLI child; parent CUDA stats cannot measure its peak.
            peak_memory_bytes=None if training else torch.cuda.max_memory_allocated(0),
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


PROBE_COMPARED_KEYS = (
    "model_id",
    "page_id",
    "image_sha256",
    "original_width",
    "original_height",
    "status",
    "finish_reason",
    "response_tokens",
    "response_sha256",
    "regions",
)


def compare_probes(before: dict, after: dict, *, distinct_process: bool = False) -> None:
    """Greedy decoding must reproduce the saved adapter's output exactly after a reload."""
    for key in PROBE_COMPARED_KEYS:
        if before[key] != after[key]:
            raise RuntimeError("reload probe mismatch: " + key)
    if distinct_process and before["process_id"] == after["process_id"]:
        raise RuntimeError("fresh-process reload requires two distinct interpreters")
