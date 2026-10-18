"""Fixed OCR reference and shot comparisons using the native contrastive decoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import modal

from active_ocr.contrastive_diagnostic import file_hash, forced_states
from active_ocr.contrastive_icl import (
    _source,
    check_reference,
    contrastive_generate,
    contrastive_scores,
)
from active_ocr.evaluation import page_text_view
from active_ocr.icl import _receipt, check_input_capacity, line_key, read_cache, sample_context
from active_ocr.icl_controls import raw_metrics
from active_ocr.recognition_pipeline import _load_model, image, output_volume, sha_json
from active_ocr.region_pipeline import (
    MOUNTS,
    _check_records,
    _cuda,
    _inputs,
    _load_page,
    _record,
    _save,
    app,
)

SPEC = Path("experiments/recipes/read2016-known-box-references.json")
ARMS = ("wrong_reordered", "correct_reordered", "query_noise")


def support_ids(recipe, mode, key):
    shots = recipe["spec"].get("shot_counts", {}).get(mode, 8)
    return recipe["contexts"][key][:shots]


def reference_pairs(ids, arm):
    """The inverse label rotation keeps exactly the original incorrect pair multiset."""
    images, labels = {
        "wrong_reordered": (ids, ids[-1:] + ids[:-1]),
        "correct_reordered": (ids[1:] + ids[:1], ids[1:] + ids[:1]),
        "query_noise": (ids, ids),
    }[arm]
    return list(zip(images, labels, strict=True))


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def query_noise(batch, key, processor):
    """Official VCD schedule on normalized query pixels; preserve still-frame duplication."""
    import torch

    patch = processor.image_processor.patch_size
    temporal = processor.image_processor.temporal_patch_size
    count = int(batch["image_grid_thw"][-1].prod())
    pixels = batch["pixel_values"]
    assert pixels.dtype == torch.float32 and pixels.device.type == "cpu"
    query = pixels[-count:].reshape(count, 3, temporal, patch, patch)
    assert torch.equal(query, query[:, :, :1].expand_as(query))
    seed = int(sha_json([824, "exp020-vcd-query", key])[:16], 16) % (2**63)
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn((count, 3, 1, patch, patch), generator=generator).expand_as(query)
    betas = torch.sigmoid(torch.linspace(-6, 6, 1000)) * (0.005 - 1e-5) + 1e-5
    alpha = torch.cumprod(1 - betas, 0)[500]
    changed = pixels.clone()
    changed[-count:] = alpha.sqrt() * pixels[-count:] + (1 - alpha).sqrt() * noise.reshape(
        count, -1
    )
    assert torch.equal(changed[:-count], pixels[:-count]) and not torch.equal(changed, pixels)
    return dict(batch, pixel_values=changed), {
        "seed": seed,
        "step": 500,
        "alpha_bar": float(alpha),
        "query_patch_rows": count,
        "positive_pixels_sha256": tensor_hash(pixels),
        "negative_pixels_sha256": tensor_hash(changed),
        "support_pixels_sha256": tensor_hash(pixels[:-count]),
        "negative_support_pixels_sha256": tensor_hash(changed[:-count]),
    }


def cached_query_noise(batch, key, processor, directory):
    """Reuse a realized query perturbation exactly across contexts and resumed hosts."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    pixels = batch["pixel_values"]
    count = int(batch["image_grid_thw"][-1].prod())
    clean_hash = tensor_hash(pixels[-count:])
    path = directory / (sha_json(key) + ".safetensors")
    if not path.exists():
        negative, meta = query_noise(batch, key, processor)
        patch = processor.image_processor.patch_size
        epsilon = torch.randn(
            (count, 3, 1, patch, patch), generator=torch.Generator().manual_seed(meta["seed"])
        )
        info = {
            "key": key,
            "clean_query_sha256": clean_hash,
            "negative_query_sha256": tensor_hash(negative["pixel_values"][-count:]),
            "noise": {k: meta[k] for k in ("seed", "step", "alpha_bar", "query_patch_rows")},
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {"query": negative["pixel_values"][-count:].contiguous(), "epsilon": epsilon},
            str(path),
            metadata={"noise": json.dumps(info, sort_keys=True)},
        )
    with safe_open(str(path), framework="pt") as saved:
        info = json.loads(saved.metadata()["noise"])
    assert info["key"] == key and info["clean_query_sha256"] == clean_hash
    query = load_file(str(path))["query"]
    assert query.shape == pixels[-count:].shape and query.dtype == pixels.dtype
    assert tensor_hash(query) == info["negative_query_sha256"]
    changed = pixels.clone()
    changed[-count:] = query
    assert torch.equal(changed[:-count], pixels[:-count])
    return dict(batch, pixel_values=changed), info["noise"] | {
        "positive_pixels_sha256": tensor_hash(pixels),
        "negative_pixels_sha256": tensor_hash(changed),
        "support_pixels_sha256": tensor_hash(pixels[:-count]),
        "negative_support_pixels_sha256": tensor_hash(changed[:-count]),
        "clean_query_pixels_sha256": clean_hash,
        "negative_query_pixels_sha256": tensor_hash(query),
        "cache_file": str(path),
        "cache_sha256": file_hash(path),
        "cpu_capability": info["cpu_capability"],
    }


def reference_data(recipe, truth):
    result = {}
    for mode, ref in recipe["references"].items():
        assert file_hash(ref["path"]) == ref["file_sha256"]
        data = json.loads(Path(ref["path"]).read_text())
        data = data[ref["item"]] if ref["item"] is not None else data
        assert sha_json(data["receipt"]) == ref["receipt_sha256"]
        assert raw_metrics(data["receipt"], truth) == ref["raw_line"]
        check_reference(
            data["receipt"],
            recipe["queries"],
            recipe["config"],
            recipe["source_run"],
            recipe["contexts"] if mode != "zero" else None,
        )
        result[mode] = data["receipt"]
    return result


def prepare(spec):
    parent = Path(spec["parent_run"])
    original = json.loads((parent / "recipe.json").read_text())
    config, queries = original["config"], original["queries"]
    source, pages, truth, geometry = _source(config)
    bank = json.loads((parent / "bank.json").read_text())
    assert tuple(spec["arms"]) == ARMS
    assert (spec["shots"], spec["seed"], spec["alpha"], spec["beta"], spec["noise_step"]) == (
        8,
        824,
        1,
        0.1,
        500,
    )
    assert sha_json(bank) == original["bank_sha256"]
    assert len(bank) == 24 and sum(len(p["regions"]) for p in bank) == 589
    assert len(queries) == 8 and sum(len(p["regions"]) for p in queries) == 182
    for p in bank:
        assert pages[p["page_id"]].split.value == "train"
        assert p == _record(pages[p["page_id"]], geometry[p["page_id"]], truth[p["page_id"]])
    for p in queries:
        assert pages[p["page_id"]].split.value == "validation"
        assert p == _record(pages[p["page_id"]], geometry[p["page_id"]])
    _check_records(bank, labelled=True)
    _check_records(queries, labelled=False)
    assert not {p["image_sha256"] for p in bank} & {p["image_sha256"] for p in queries}
    targets = {line_key(p, r): r["target"] for p in bank for r in p["regions"]}
    contexts = {
        line_key(p, r): sample_context(list(targets), line_key(p, r), 8, 824, sha_json(bank))
        for p in queries
        for r in p["regions"]
    }
    for ids in contexts.values():
        old = list(zip(ids[1:] + ids[:1], ids, strict=True))
        assert Counter(reference_pairs(ids, ARMS[0])) == Counter(old)
        assert all(targets[i] != targets[j] for i, j in old)
    refs = {
        m: dict(original["references"][m])
        for m in ("zero", "paired-eight", "transcripts-only-eight", "mismatched-eight")
    }
    path = parent / "pilot/results.json"
    saved = json.loads(path.read_text())
    assert file_hash(path) == spec["contrast_results_sha256"]
    refs["contrastive"] = {
        "path": str(path),
        "item": None,
        "file_sha256": file_hash(path),
        "receipt_sha256": sha_json(saved["receipt"]),
        "raw_line": saved["raw_line"],
        "reused": True,
    }
    for name, expected in original["model_files_sha256"].items():
        assert file_hash(Path(spec["model_assets"]) / name) == expected
    names = set(original["code_sha256"]) | {
        "contrastive_references.py",
        "contrastive_diagnostic.py",
    }
    recipe = {
        "spec": spec,
        "spec_file_sha256": file_hash(SPEC),
        "config": config,
        "source_run": source.id,
        "queries": queries,
        "contexts": contexts,
        "cases": sorted(contexts),
        "bank_sha256": sha_json(bank),
        "references": refs,
        "native_tokens": original["native_check_tokens"],
        "parent_recipe_sha256": sha_json(original),
        "parent_recipe_file_sha256": file_hash(parent / "recipe.json"),
        "code_sha256": {n: file_hash(Path(__file__).with_name(n)) for n in sorted(names)},
        "model_files_sha256": original["model_files_sha256"],
        **{
            k: original[k] for k in ("input_bundle", "input_volume", "output_volume", "modal_image")
        },
    }
    receipts = reference_data(recipe, truth)
    rows = {line_key(p, r): r for p in receipts["contrastive"]["predictions"] for r in p["regions"]}
    recipe["contrast_tokens"] = {k: rows[k]["output_token_ids"] for k in recipe["native_tokens"]}
    root = Path(spec["output_directory"])
    _save(root / "recipe.json", recipe)
    _save(root / "bank.json", bank)
    for name in names:
        dest = root / "execution-source" / name
        if dest.exists():
            assert file_hash(dest) == recipe["code_sha256"][name]
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(Path(__file__).with_name(name).read_bytes())
    _save(
        root / "preflight.json",
        {
            "recipe_sha256": sha_json(recipe),
            "lines": 182,
            "new_outputs": 546,
            "readiness_outputs": 9,
            "reused_metrics": {k: v["raw_line"] for k, v in refs.items()},
        },
    )
    print(json.dumps({"prepared": 546, "recipe_sha256": sha_json(recipe)}), flush=True)


def prepare_shots(spec, spec_path):
    """Freeze the approved nested-shot extension without changing its source experiment."""
    parent = Path(spec["parent_run"])
    recipe = json.loads((parent / "recipe.json").read_text())
    assert sha_json(recipe) == spec["parent_recipe_sha256"]
    full = spec.get("evaluation_scope") == "full"
    expected_shots = {"zero_query_noise": 0, "two_ordinary": 2, "two_query_noise": 2}
    if full:
        expected_shots["eight_query_noise"] = 8
        previous = json.loads((Path(spec["full_source_run"]) / "recipe.json").read_text())
        assert sha_json(previous) == spec["full_source_recipe_sha256"]
        assert previous["bank_sha256"] == recipe["bank_sha256"]
        for name in (
            "models",
            "training",
            "inference",
            "prompt",
            "crop_padding_px",
            "source_run_id",
            "source_manifest_sha256",
        ):
            assert previous["config"][name] == recipe["config"][name]
        recipe["queries"] = previous["queries"]
        assert (
            len(recipe["queries"]) == 42
            and sum(len(p["regions"]) for p in recipe["queries"]) == 861
        )
        recipe["config"]["validation_page_ids"] = [p["page_id"] for p in recipe["queries"]]
        recipe["references"] = {n: previous["references"][n] for n in ("zero", "paired-eight")}
        recipe["full_source_recipe_sha256"] = sha_json(previous)
        recipe["full_source_recipe_file_sha256"] = file_hash(
            Path(spec["full_source_run"]) / "recipe.json"
        )
        recipe.pop("eight_noise_checks", None)
        recipe.pop("contrast_tokens", None)
        recipe["native_tokens"] = {line_key(p, p["regions"][0]): [] for p in recipe["queries"][:3]}
    assert spec["shot_counts"] == expected_shots
    assert list(spec["shot_counts"]) == spec["arms"]
    assert (spec["seed"], spec["alpha"], spec["beta"], spec["noise_step"]) == (824, 1, 0.1, 500)
    assert spec["model_assets"] == recipe["spec"]["model_assets"]
    bank = json.loads((parent / "bank.json").read_text())
    assert sha_json(bank) == recipe["bank_sha256"]
    source, pages, truth, geometry = _source(recipe["config"])
    assert source.id == recipe["source_run"]
    for records, split in ((bank, "train"), (recipe["queries"], "validation")):
        for record in records:
            key = record["page_id"]
            assert pages[key].split.value == split
            expected = (
                _record(pages[key], geometry[key], truth[key])
                if split == "train"
                else _record(pages[key], geometry[key])
            )
            assert record == expected
    targets = [line_key(p, r) for p in bank for r in p["regions"]]
    if full:
        recipe["contexts"] = {
            line_key(p, r): sample_context(targets, line_key(p, r), 8, 824, recipe["bank_sha256"])
            for p in recipe["queries"]
            for r in p["regions"]
        }
        recipe["cases"] = sorted(recipe["contexts"])
    for key, ids in recipe["contexts"].items():
        assert ids == sample_context(targets, key, 8, 824, recipe["bank_sha256"])
    assert not {p["image_sha256"] for p in bank} & {p["image_sha256"] for p in recipe["queries"]}
    for name, expected in recipe["model_files_sha256"].items():
        assert file_hash(Path(spec["model_assets"]) / name) == expected
    receipts = reference_data(recipe, truth)
    if full:
        recipe["paired_predictions"] = {
            line_key(p, r): r for p in receipts["paired-eight"]["predictions"] for r in p["regions"]
        }
        recipe["zero_predictions"] = {
            line_key(p, r): r for p in receipts["zero"]["predictions"] for r in p["regions"]
        }
    else:
        path = parent / "query_noise.json"
        data = json.loads(path.read_text())
        assert data["recipe_sha256"] == sha_json(recipe)
        check_reference(
            data["receipt"],
            recipe["queries"],
            recipe["config"],
            recipe["source_run"],
            recipe["contexts"],
        )
        assert raw_metrics(data["receipt"], truth) == data["raw_line"]
        noise_rows = {
            line_key(p, r): r for p in data["receipt"]["predictions"] for r in p["regions"]
        }
        recipe["references"]["query-noise-eight"] = {
            "path": str(path),
            "item": None,
            "file_sha256": file_hash(path),
            "receipt_sha256": sha_json(data["receipt"]),
            "raw_line": data["raw_line"],
            "reused": True,
        }
        recipe["zero_predictions"] = {
            line_key(p, r): r for p in receipts["zero"]["predictions"] for r in p["regions"]
        }
        recipe["eight_noise_checks"] = {
            k: {"metadata": r["metadata"], "noise": r["noise"]} for k, r in noise_rows.items()
        }
    recipe["parent_recipe_sha256"] = spec["parent_recipe_sha256"]
    recipe["parent_recipe_file_sha256"] = file_hash(parent / "recipe.json")
    recipe["spec"], recipe["spec_file_sha256"] = spec, file_hash(spec_path)
    recipe["code_sha256"] = {
        n: file_hash(Path(__file__).with_name(n)) for n in recipe["code_sha256"]
    }
    reference_data(recipe, truth)
    root = Path(spec["output_directory"])
    if (root / "recipe.json").exists():
        assert json.loads((root / "recipe.json").read_text()) == recipe
    _save(root / "recipe.json", recipe)
    _save(root / "bank.json", bank)
    for name, expected in recipe["code_sha256"].items():
        dest = root / "execution-source" / name
        if dest.exists():
            assert file_hash(dest) == expected
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(Path(__file__).with_name(name).read_bytes())
    _save(
        root / "preflight.json",
        {
            "recipe_sha256": sha_json(recipe),
            "lines": len(recipe["cases"]),
            "new_outputs": len(spec["arms"]) * len(recipe["cases"]),
            "reused_metrics": {k: v["raw_line"] for k, v in recipe["references"].items()},
        },
    )
    print(
        json.dumps(
            {
                "prepared": len(spec["arms"]) * len(recipe["cases"]),
                "recipe_sha256": sha_json(recipe),
            }
        ),
        flush=True,
    )


@app.function(
    image=image.add_local_python_source("active_ocr"),
    gpu="L4",
    cpu=2,
    memory=32768,
    timeout=7200,
    volumes=MOUNTS,
)
def run_reference_cases(payload):
    import torch
    import transformers
    from safetensors.torch import save_file

    recipe, bank = payload["recipe"], payload["bank"]
    digest, config = sha_json(recipe), recipe["config"]
    remote = Path("/output") / (
        recipe["spec"].get("remote_prefix", "exp020-references-") + digest[:12]
    )
    output_volume.reload()
    assert sha_json(bank) == recipe["bank_sha256"]
    _check_records(bank, labelled=True)
    _check_records(recipe["queries"], labelled=False)
    for name, expected in recipe["code_sha256"].items():
        assert file_hash(Path(__file__).with_name(name)) == expected
    for name, expected in recipe["model_files_sha256"].items():
        assert file_hash(Path("/input/model") / name) == expected
    records = bank + recipe["queries"]
    for p in records:
        assert file_hash(Path("/input/images") / p["image_sha256"]) == p["image_sha256"]
    full = recipe["spec"].get("evaluation_scope") == "full"
    if full:
        for name, value in (("recipe.json", recipe), ("bank.json", bank)):
            path = remote / name
            if path.exists():
                assert json.loads(path.read_text()) == value
            else:
                _save(path, value)
        for name, expected in recipe["code_sha256"].items():
            dest = remote / "execution-source" / name
            if dest.exists():
                assert file_hash(dest) == expected
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(Path(__file__).with_name(name).read_bytes())
    processor, model = _load_model(config["models"]["qwen"])
    model = model.to("cuda").eval()
    pages = {p["page_id"]: _load_page(p) for p in records}
    lines = {line_key(p, r): (p, r) for p in records for r in p["regions"]}
    settings = {
        k: config["inference"][k] for k in ("max_new_tokens", "do_sample", "repetition_penalty")
    }
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    rows = []
    study = bool(recipe["spec"].get("shot_counts"))
    dirty = False
    checkpoint_start = 0
    with torch.inference_mode():
        for mode, key in payload["tasks"]:
            identity = sha_json([digest, mode, key])
            row = read_cache(remote / f"{identity}.json", identity)
            if row is None:
                torch.cuda.reset_peak_memory_stats()
                p, r = lines[key]
                ids = support_ids(recipe, mode, key)
                pairs = reference_pairs(ids, "query_noise" if study else mode)
                checks = {}
                examples = [(pages[lines[k][0]["page_id"]], lines[k][1]) for k in ids]
                positive = _inputs(processor, pages[p["page_id"]], r, config, examples=examples)
                noise = None
                if mode == "query_noise" or study:
                    if full:
                        negative, noise = cached_query_noise(
                            positive, key, processor, remote / "noise"
                        )
                        noise["cache_file"] = str(Path(noise["cache_file"]).relative_to("/output"))
                    else:
                        negative, noise = query_noise(positive, key, processor)
                    assert torch.equal(positive["input_ids"], negative["input_ids"])
                    assert torch.equal(positive["image_grid_thw"], negative["image_grid_thw"])
                else:
                    examples_n = [
                        (
                            pages[lines[i][0]["page_id"]],
                            lines[i][1] | {"target": lines[j][1]["target"]},
                        )
                        for i, j in pairs
                    ]
                    negative = _inputs(
                        processor, pages[p["page_id"]], r, config, examples=examples_n
                    )
                    nquery = int(positive["image_grid_thw"][-1].prod())
                    assert torch.equal(
                        positive["pixel_values"][-nquery:], negative["pixel_values"][-nquery:]
                    )
                    assert torch.equal(
                        positive["image_grid_thw"][-1], negative["image_grid_thw"][-1]
                    )
                if study and mode == "zero_query_noise" and not full:
                    eight_examples = [
                        (pages[lines[k][0]["page_id"]], lines[k][1])
                        for k in recipe["contexts"][key]
                    ]
                    eight = _inputs(
                        processor, pages[p["page_id"]], r, config, examples=eight_examples
                    )
                    eight_negative, eight_meta = query_noise(eight, key, processor)
                    old = recipe["eight_noise_checks"][key]
                    assert eight_meta == old["noise"]
                    count = int(positive["image_grid_thw"][-1].prod())
                    assert torch.equal(
                        positive["pixel_values"][-count:], eight["pixel_values"][-count:]
                    )
                    assert torch.equal(
                        negative["pixel_values"][-count:], eight_negative["pixel_values"][-count:]
                    )
                    checks.update(eight_shot_query_match=True, eight_shot_noise_match=True)
                metadata = []
                for batch in (positive, negative):
                    check_input_capacity(batch, processor, model, len(ids) + 1, 64)
                    metadata.append(
                        {
                            "prompt_tokens": batch["input_ids"].shape[1],
                            "ids_sha256": sha_json(batch["input_ids"].tolist()),
                            "grid": batch["image_grid_thw"].tolist(),
                            "pixels_sha256": tensor_hash(batch["pixel_values"]),
                        }
                    )
                positive, negative = _cuda(positive), _cuda(negative)
                plen = positive["input_ids"].shape[1]
                tensors = {}
                native = None
                if key in recipe["native_tokens"]:
                    if study:
                        native = model.generate(**positive, **settings)[0, plen:].tolist()
                        if mode == "zero_query_noise":
                            old_zero = recipe["zero_predictions"][key]
                            assert (
                                processor.decode(native, skip_special_tokens=True)
                                .strip()
                                .replace("\n", " ")
                                == old_zero["text"]
                            )
                            assert len(native) == old_zero["tokens"]
                            if "input_tokens" in old_zero:
                                assert plen == old_zero["input_tokens"]
                        elif mode == "eight_query_noise":
                            old_eight = recipe["paired_predictions"][key]
                            assert (
                                processor.decode(native, skip_special_tokens=True)
                                .strip()
                                .replace("\n", " ")
                                == old_eight["text"]
                            )
                            assert (
                                len(native) == old_eight["tokens"]
                                and plen == old_eight["input_tokens"]
                            )
                        elif mode == "two_query_noise":
                            ordinary_id = sha_json([digest, "two_ordinary", key])
                            ordinary = read_cache(remote / f"{ordinary_id}.json", ordinary_id)
                            assert ordinary is not None and native == ordinary["output_token_ids"]
                        checks["native"] = True
                    elif mode == ARMS[0]:
                        native = model.generate(**positive, **settings)[0, plen:].tolist()
                        assert native == recipe["native_tokens"][key]
                        old = _cuda(
                            _inputs(
                                processor,
                                pages[p["page_id"]],
                                r,
                                config,
                                examples=examples,
                                demonstration_mode="mismatched",
                            )
                        )
                        got, _ = contrastive_generate(model, positive, old, settings)
                        assert got[0, plen:].tolist() == recipe["contrast_tokens"][key]
                        checks.update(native=True, original_contrast=True)
                    zero, _ = contrastive_generate(model, positive, negative, settings, alpha=0)
                    assert zero[0, plen:].tolist() == (
                        native if study else recipe["native_tokens"][key]
                    )
                    checks["alpha_zero"] = True
                torch.cuda.synchronize()
                start = time.perf_counter()
                if mode == "two_ordinary":
                    tokens = (
                        native
                        if native is not None
                        else model.generate(**positive, **settings)[0, plen:].tolist()
                    )
                    trace = []
                else:
                    got, trace = contrastive_generate(model, positive, negative, settings)
                    tokens = got[0, plen:].tolist()
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                vector = None
                if key in recipe["native_tokens"]:
                    values, processors = [], None
                    for role, batch in (("positive", positive), ("negative", negative)):
                        state = next(forced_states(model, batch, tokens[:1], settings))
                        tensors[role + ".logits"] = state["logits"].cpu().contiguous()
                        tensors[role + ".ids"] = state["ids"].cpu().contiguous()
                        values.append(state["logits"])
                        if role == "positive":
                            processors = state["processors"]
                    score = contrastive_scores(
                        *values,
                        positive["input_ids"],
                        processors,
                        alpha=0 if mode == "two_ordinary" else 1,
                    )
                    assert int(score.argmax()) == tokens[0]
                    path = remote / f"{identity}.safetensors"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    save_file(tensors, str(path))
                    vector = {"path": str(path.relative_to("/output")), "sha256": file_hash(path)}
                ended = bool(tokens and tokens[-1] in eos)
                assert ended or len(tokens) == 64
                row = {
                    "key": key,
                    "page_id": p["page_id"],
                    "id": r["id"],
                    "mode": mode,
                    "demo_ids": ids,
                    "reference_pairs": pairs,
                    "metadata": metadata,
                    "noise": noise,
                    "text": processor.decode(tokens, skip_special_tokens=True)
                    .strip()
                    .replace("\n", " "),
                    "output_token_ids": tokens,
                    "tokens": len(tokens),
                    "truncated": len(tokens) >= 64,
                    "ended_with_eos": ended,
                    "stop_reason": "eos" if ended else "length",
                    "trace": trace,
                    "checks": checks,
                    "vectors": vector,
                    "generation_seconds": seconds,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "app_id": payload["app_id"],
                    "call_id": modal.current_function_call_id(),
                }
                _save(
                    remote / f"{identity}.json",
                    {"identity": identity, "row": row, "row_sha256": sha_json(row)},
                )
                if full:
                    dirty = True
                else:
                    output_volume.commit()
            rows.append(row)
            boundary = len(rows) == len(payload["tasks"]) or payload["tasks"][len(rows)][0] != mode
            if full and (len(rows) - checkpoint_start >= recipe["spec"]["chunk_lines"] or boundary):
                chunk = {
                    "identity": sha_json([digest, payload["tasks"][checkpoint_start : len(rows)]]),
                    "recipe_sha256": digest,
                    "rows": rows[checkpoint_start:],
                }
                _save(
                    remote / ("chunk-" + sha_json([payload["tasks"][0], len(rows)])[:16] + ".json"),
                    chunk,
                )
                output_volume.commit()
                dirty = False
                checkpoint_start = len(rows)
                print(
                    json.dumps(
                        {"completed": len(rows), "mode": mode, "key": key, "committed": True}
                    ),
                    flush=True,
                )
            elif not full:
                print(json.dumps({"completed": len(rows), "mode": mode, "key": key}), flush=True)
    assert not dirty
    return {
        "recipe_sha256": digest,
        "rows": rows,
        "gpu": torch.cuda.get_device_name(),
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "model": type(model).__name__,
            "processor": type(processor).__name__,
        },
    }


def report(recipe, root):
    from rapidfuzz.distance import Levenshtein

    _, _, truth, _ = _source(recipe["config"])
    receipts = reference_data(recipe, truth)
    rows = []
    for path in sorted(root.glob("part-*.json")):
        data = json.loads(path.read_text())
        part = read_cache(path, data["identity"])
        assert part["recipe_sha256"] == sha_json(recipe) and "L4" in part["gpu"]
        rows.extend(part["rows"])
    assert (
        len(rows)
        == len({(r["mode"], r["key"]) for r in rows})
        == len(recipe["spec"]["arms"]) * len(recipe["cases"])
    )
    for mode in recipe["spec"]["arms"]:
        selected = sorted((r for r in rows if r["mode"] == mode), key=lambda r: r["key"])
        assert [r["key"] for r in selected] == recipe["cases"]
        receipts[mode] = _receipt(
            selected, recipe["queries"], recipe["config"], recipe["source_run"]
        )
        _save(
            root / f"{mode}.json",
            {
                "receipt": receipts[mode],
                "raw_line": raw_metrics(receipts[mode], truth),
                "recipe_sha256": sha_json(recipe),
                "reused": False,
            },
        )
    indices = {
        m: {line_key(p, r): r for p in x["predictions"] for r in p["regions"]}
        for m, x in receipts.items()
    }
    details = []
    for p in recipe["queries"]:
        for r in p["regions"]:
            key = line_key(p, r)
            target = page_text_view([truth[p["page_id"]][r["id"]]])
            values = {}
            for mode, index in indices.items():
                text = page_text_view([index[key]["text"]])
                ops = Counter(o.tag for o in Levenshtein.editops(target, text))
                values[mode] = {
                    "text": text,
                    "edits": Levenshtein.distance(target, text),
                    "chars": len(text),
                    "exact": text == target,
                    "operations": {k: ops[k] for k in ("replace", "delete", "insert")},
                }
            details.append(
                {"key": key, "page_id": p["page_id"], "reference": target, "predictions": values}
            )
    stats = {}
    for mode, receipt in receipts.items():
        vals = [d["predictions"][mode] for d in details]
        counts = {
            "metrics": raw_metrics(receipt, truth),
            "chars": sum(v["chars"] for v in vals),
            "exact": sum(v["exact"] for v in vals),
            "operations": {
                k: sum(v["operations"][k] for v in vals) for k in ("replace", "delete", "insert")
            },
        }
        if mode in recipe["spec"]["arms"]:
            own = [r for r in rows if r["mode"] == mode]
            counts.update(
                caps=sum(r["stop_reason"] == "length" for r in own),
                empty=sum(not r["text"] for r in own),
            )
        comparisons = {}
        for baseline in (
            ("zero", "paired-eight")
            if recipe["spec"].get("evaluation_scope") == "full"
            else ("paired-eight", "contrastive")
        ):
            ds = [
                d["predictions"][baseline]["edits"] - d["predictions"][mode]["edits"]
                for d in details
            ]
            comparisons[baseline] = {
                "saved_edits": sum(ds),
                "improved": sum(d > 0 for d in ds),
                "harmed": sum(d < 0 for d in ds),
                "tied": sum(d == 0 for d in ds),
                "per_page_saved_edits": {
                    p: sum(delta for d, delta in zip(details, ds, strict=True) if d["page_id"] == p)
                    for p in sorted({d["page_id"] for d in details})
                },
            }
        stats[mode] = counts | {"comparisons": comparisons}
    summary = {
        "recipe_sha256": sha_json(recipe),
        "statistics": stats,
        "rows": details,
        "lines": len(recipe["cases"]),
        "new_outputs": len(rows),
        "generation_seconds": sum(r["generation_seconds"] for r in rows),
        "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in rows),
    }
    _save(root / "summary.json", summary)
    print(json.dumps({m: v["metrics"] for m, v in stats.items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "report"))
    parser.add_argument("--spec", type=Path, default=SPEC)
    parser.add_argument("--readiness-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    root = Path(spec["output_directory"])
    if args.stage == "prepare":
        prepare_shots(spec, args.spec) if spec.get("shot_counts") else prepare(spec)
        return
    recipe = json.loads((root / "recipe.json").read_text())
    assert recipe["spec"] == spec
    if args.stage == "report":
        report(recipe, root)
        return
    for name, expected in recipe["code_sha256"].items():
        assert file_hash(Path(__file__).with_name(name)) == expected
    bank = json.loads((root / "bank.json").read_text())
    tasks = [(m, k) for m in spec["arms"] for k in recipe["cases"]]
    gate = [t for t in tasks if t[1] in recipe["native_tokens"]]
    rest = [t for t in tasks if t[1] not in recipe["native_tokens"]]
    with modal.enable_output(), app.run(environment_name="main"):
        for i, work in enumerate((gate, rest)):
            if i:
                ready = json.loads((root / "readiness.json").read_text())
                assert ready["passed"] and ready["recipe_sha256"] == sha_json(recipe)
                if spec.get("shot_counts"):
                    audited = json.loads((root / "readiness-audit.json").read_text())
                    assert audited["passed"] and audited["recipe_sha256"] == sha_json(recipe)
            path = root / f"part-{i}.json"
            identity = sha_json([sha_json(recipe), work])
            result = read_cache(path, identity)
            if result is None:
                function = (
                    run_reference_cases.with_options(timeout=10800)
                    if spec.get("evaluation_scope") == "full"
                    else run_reference_cases
                )
                call = function.spawn(
                    {"recipe": recipe, "bank": bank, "tasks": work, "app_id": app.app_id}
                )
                with (root / "calls.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "app_id": app.app_id,
                                "call_id": call.object_id,
                                "new_outputs": len(work),
                                "started_utc": datetime.now(UTC).isoformat(),
                            }
                        )
                        + "\n"
                    )
                result = call.get()
                _save(path, {"identity": identity, "row": result, "row_sha256": sha_json(result)})
            assert result["recipe_sha256"] == sha_json(recipe) and len(result["rows"]) == len(work)
            for row in result["rows"]:
                if row["vectors"]:
                    vector = row["vectors"]
                    dest = root / Path(vector["path"]).name
                    if not dest.exists():
                        dest.write_bytes(b"".join(output_volume.read_file(vector["path"])))
                    assert file_hash(dest) == vector["sha256"]
                if not i and spec.get("evaluation_scope") == "full":
                    noise = row["noise"]
                    dest = root / Path(noise["cache_file"]).name
                    if not dest.exists():
                        dest.write_bytes(b"".join(output_volume.read_file(noise["cache_file"])))
                    assert file_hash(dest) == noise["cache_sha256"]
            if not i:
                assert len(result["rows"]) == len(spec["arms"]) * len(
                    recipe["native_tokens"]
                ) and all(r["checks"]["alpha_zero"] for r in result["rows"])
                if spec.get("shot_counts"):
                    assert all(r["checks"].get("native") for r in result["rows"])
                assert all(
                    r["checks"].get("native") and r["checks"].get("original_contrast")
                    for r in result["rows"]
                    if r["mode"] == ARMS[0]
                )
                _save(
                    root / "readiness.json",
                    {
                        "recipe_sha256": sha_json(recipe),
                        "passed": True,
                        "outputs": len(result["rows"]),
                        "native_replays": len(result["rows"])
                        if spec.get("evaluation_scope") == "full"
                        else 3,
                        "original_contrast_replays": 0 if spec.get("shot_counts") else 3,
                        "alpha_zero_replays": len(result["rows"]),
                    },
                )
                if args.readiness_only:
                    return
    report(recipe, root)


if __name__ == "__main__":
    main()
