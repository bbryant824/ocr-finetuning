"""Bounded association-contrastive ICL on the existing READ2016 crop interfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

from active_ocr.icl import _receipt, check_input_capacity, line_key, read_cache, sample_context
from active_ocr.icl_controls import raw_metrics
from active_ocr.recognition_pipeline import _load_model, image, output_volume, sha_json
from active_ocr.region_pipeline import (
    MOUNTS,
    _check_records,
    _cuda,
    _inputs,
    _load_page,
    _save,
    app,
)

SPEC = Path("experiments/recipes/read2016-known-box-icl-contrastive-pilot.json")


def contrastive_scores(positive, negative, ids, processors, alpha=1, beta=0.1):
    """Combine raw logits, then native processors; eligibility uses processed positives."""
    import torch

    if not (0 < beta <= 1) or alpha < 0:
        raise ValueError("invalid contrastive strength or plausibility threshold")
    if not torch.isfinite(positive).all() or not torch.isfinite(negative).all():
        raise ValueError("contrastive branch logits must be finite")
    ordinary = processors(ids, positive.clone())
    if alpha == 0:
        return ordinary
    eligible = ordinary - ordinary.max(dim=-1, keepdim=True).values >= math.log(beta)
    if not eligible.any(dim=-1).all():
        raise ValueError("empty positive plausibility set")
    combined = processors(ids, positive + alpha * (positive - negative))
    return combined.masked_fill(~eligible, -torch.inf)


def _capture(model, input_ids, logits_processor, stopping_criteria, generation_config, **kwargs):
    """Use Transformers 5.6's callable custom_generate input preparation, without decoding."""
    kwargs.pop("synced_gpus", None)
    return {
        "ids": input_ids,
        "kwargs": kwargs,
        "processors": logits_processor,
        "stopping": stopping_criteria,
        "config": generation_config,
    }


def contrastive_generate(model, positive, negative, settings, alpha=1, beta=0.1):
    """One shared generated suffix, separate native position IDs and KV caches per branch."""
    import torch

    states = [
        model.generate(**batch, **settings, custom_generate=_capture)
        for batch in (positive, negative)
    ]
    lengths = [state["ids"].shape[1] for state in states]
    if any(state["ids"].shape[0] != 1 for state in states):
        raise ValueError("contrastive OCR supports one query at a time")
    trace = []
    while True:
        logits, cache_lengths = [], []
        for state in states:
            if not trace:
                output = model._prefill(state["ids"], state["config"], state["kwargs"])
            else:
                inputs = model.prepare_inputs_for_generation(
                    state["ids"], next_sequence_length=1, **state["kwargs"]
                )
                output = model(**inputs, return_dict=True)
            logits.append(output.logits[:, -1].to(copy=True, dtype=torch.float32))
            state["kwargs"] = model._update_model_kwargs_for_generation(output, state["kwargs"])
            cache_lengths.append(state["kwargs"]["past_key_values"].get_seq_length())
            del output
        if states[0]["kwargs"]["past_key_values"] is states[1]["kwargs"]["past_key_values"]:
            raise ValueError("contrastive caches must be independent")
        if cache_lengths != [n + len(trace) for n in lengths] or not torch.equal(
            states[0]["ids"][:, lengths[0] :], states[1]["ids"][:, lengths[1] :]
        ):
            raise ValueError("branch cache or shared-prefix mismatch")
        scores = contrastive_scores(
            *logits, states[0]["ids"], states[0]["processors"], alpha=alpha, beta=beta
        )
        token = scores.argmax(dim=-1, keepdim=True)
        trace.append(
            {
                "shared_prefix_tokens": len(trace),
                "positive_cache_tokens": cache_lengths[0],
                "negative_cache_tokens": cache_lengths[1],
                "token_id": int(token.item()),
            }
        )
        for state in states:
            state["ids"] = torch.cat((state["ids"], token), dim=-1)
        if states[0]["stopping"](states[0]["ids"], None).all():
            return states[0]["ids"], trace


def lens_trace(model, batch, prefix_ids):
    """Read decoder-block inputs after preceding DeepStack injection; final norm is separate."""
    import torch

    positions = sorted({0, min(4, len(prefix_ids) - 1), min(12, len(prefix_ids) - 1)})
    if not prefix_ids:
        raise ValueError("lens requires an observed output-token sequence")
    tokens = batch["input_ids"].shape[1]
    prefix = batch["input_ids"].new_tensor([prefix_ids[: max(positions)]])
    probe = dict(batch)
    probe["input_ids"] = torch.cat((probe["input_ids"], prefix), dim=-1)
    for key in ("attention_mask", "mm_token_type_ids"):
        extra = torch.ones_like(prefix) if key == "attention_mask" else torch.zeros_like(prefix)
        probe[key] = torch.cat((probe[key], extra), dim=-1)
    indices = probe["input_ids"].new_tensor([tokens - 1 + p for p in positions])
    prepared = model.generate(**probe, max_new_tokens=1, do_sample=False, custom_generate=_capture)
    language = model.model.language_model
    states, handles = {}, []
    for index, layer in enumerate(language.layers):

        def capture(module, args, index=index):
            states[index] = args[0][:, indices].detach().clone()

        handles.append(layer.register_forward_pre_hook(capture))
    handles.append(
        language.norm.register_forward_hook(
            lambda module, args, result: states.update(final=result[:, indices].detach().clone())
        )
    )
    try:
        prepared["kwargs"]["logits_to_keep"] = indices
        output = model._prefill(prepared["ids"], prepared["config"], prepared["kwargs"])
    finally:
        for handle in handles:
            handle.remove()
    final = model.lm_head(states["final"])
    max_error = float((final.float() - output.logits.float()).abs().max())
    if not torch.allclose(final.float(), output.logits.float(), atol=0.001, rtol=0):
        raise ValueError("final lens readout differs from native raw logits")
    rows = []
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    for layer, hidden in states.items():
        logits = model.lm_head(language.norm(hidden) if layer != "final" else hidden)[0].float()
        logp = logits.log_softmax(dim=-1)
        for j, position in enumerate(positions):
            observed = prefix_ids[position]
            rows.append(
                {
                    "layer": layer,
                    "answer_position": position,
                    "observed_token_id": observed,
                    "observed_logprob": float(logp[j, observed]),
                    "observed_rank": 1 + int((logits[j] > logits[j, observed]).sum()),
                    "top_token_id": int(logits[j].argmax()),
                    "eos_probability": float(logp[j, eos].exp().sum()),
                }
            )
    return {"final_max_abs_error": max_error, "positions": positions, "rows": rows}


@app.function(
    image=image.add_local_python_source("active_ocr"),
    gpu="L4",
    cpu=2,
    memory=32768,
    timeout=7200,
    volumes=MOUNTS,
)
def run_chunk(payload):
    import torch
    import transformers

    output_volume.reload()
    recipe, bank = payload["recipe"], payload["bank"]
    config, scope = recipe["config"], payload["scope"]
    digest = sha_json(recipe)
    run_name = "icl-contrastive-pilot-" + digest[:12]
    _check_records(bank, labelled=True)
    _check_records(payload["queries"], labelled=False)
    if any(":train:" not in r["page_id"] for r in bank) or any(
        ":validation:" not in r["page_id"] for r in payload["queries"]
    ):
        raise ValueError("pilot split leakage")
    if sha_json(bank) != recipe["bank_sha256"]:
        raise ValueError("bank identity mismatch")
    for name, expected in recipe["code_sha256"].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
            raise ValueError("cloud code differs from frozen pilot recipe")
    for name, expected in recipe["model_files_sha256"].items():
        with (Path("/input/model") / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("model asset differs from pinned local snapshot")
    for record in {r["page_id"]: r for r in bank + payload["queries"]}.values():
        if (
            hashlib.sha256(
                (Path("/input/images") / record["image_sha256"]).read_bytes()
            ).hexdigest()
            != record["image_sha256"]
        ):
            raise ValueError("input image checksum mismatch")
    processor, model = _load_model(config["models"]["qwen"])
    model = model.to("cuda").eval()
    pages = {r["page_id"]: _load_page(r) for r in bank + payload["queries"]}
    lines = {line_key(p, r): (p, r) for p in bank for r in p["regions"]}
    queries = {line_key(p, r): (p, r) for p in payload["queries"] for r in p["regions"]}
    settings = {
        k: config["inference"][k] for k in ("max_new_tokens", "do_sample", "repetition_penalty")
    }

    def inputs(key, mode, context_key=None):
        record, region = queries[key]
        ids = sample_context(list(lines), context_key or key, 8, 824, recipe["bank_sha256"])
        examples = [(pages[lines[i][0]["page_id"]], lines[i][1]) for i in ids]
        batch = _inputs(
            processor,
            pages[record["page_id"]],
            region,
            config,
            examples=examples,
            demonstration_mode=mode,
        )
        visual = check_input_capacity(batch, processor, model, 1 if mode == "text-only" else 9, 64)
        return _cuda(batch), ids, visual

    def generate(batch):
        return model.generate(**batch, **settings)[0, batch["input_ids"].shape[1] :].tolist()

    def decode(ids):
        eos = model.generation_config.eos_token_id
        ended = bool(ids and ids[-1] in (eos if isinstance(eos, list) else [eos]))
        return {
            "text": processor.decode(ids, skip_special_tokens=True).strip().replace("\n", " "),
            "output_token_ids": ids,
            "tokens": len(ids),
            "truncated": len(ids) >= 64,
            "ended_with_eos": ended,
            "stop_reason": "eos" if ended else "length" if len(ids) >= 64 else "other",
        }

    rows = []
    with torch.inference_mode():
        tasks = recipe["diagnostic_pairs"] if scope == "diagnose" else list(queries)
        for task in tasks:
            identity = sha_json(
                [digest, "diagnostic" if scope == "diagnose" else "contrastive", task]
            )
            path = Path("/output") / run_name / f"{identity}.json"
            row = read_cache(path, identity)
            if row is None:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                if scope == "diagnose":
                    anchor, replacement = task["anchor"], task["replacement"]
                    original, ids, _ = inputs(anchor, "text-only")
                    swapped, swap_ids, _ = inputs(replacement, "text-only", anchor)
                    if (
                        ids != swap_ids
                        or not torch.equal(original["input_ids"], swapped["input_ids"])
                        or not torch.equal(original["image_grid_thw"], swapped["image_grid_thw"])
                    ):
                        raise ValueError("query swap changed context tokens or visual geometry")
                    before, after = generate(original), generate(swapped)
                    if before != task["prefix_token_ids"]:
                        raise ValueError(
                            "native transcripts-only tokens differ from saved reference"
                        )
                    lenses = {}
                    for mode in ("paired", "mismatched", "text-only"):
                        batch, _, _ = inputs(anchor, mode)
                        lenses[mode] = lens_trace(model, batch, before)
                    lenses["query-swapped"] = lens_trace(model, swapped, before)
                    row = {
                        "anchor": anchor,
                        "replacement": replacement,
                        "demo_ids": ids,
                        "original": decode(before),
                        "swapped": decode(after),
                        "lenses": lenses,
                    }
                else:
                    positive, ids, visual = inputs(task, "paired")
                    negative, negative_ids, _ = inputs(task, "mismatched")
                    image_ids = ids[1:] + ids[:1]
                    if ids != negative_ids or any(
                        lines[a][1]["target"] == lines[b][1]["target"]
                        for a, b in zip(ids, image_ids, strict=True)
                    ):
                        raise ValueError("incorrect negative context construction")
                    checks = {}
                    if scope == "check":
                        native = generate(positive)
                        zero, _ = contrastive_generate(model, positive, negative, settings, alpha=0)
                        identical, _ = contrastive_generate(model, positive, positive, settings)
                        zero = zero[0, positive["input_ids"].shape[1] :].tolist()
                        identical = identical[0, positive["input_ids"].shape[1] :].tolist()
                        if (
                            native != zero
                            or native != identical
                            or native != recipe["native_check_tokens"][task]
                        ):
                            raise ValueError(
                                "native/alpha-zero/identical/reference token equivalence failed"
                            )
                        checks = {"native": native, "alpha_zero": zero, "identical": identical}
                    generated, trace = contrastive_generate(model, positive, negative, settings)
                    tokens = generated[0, positive["input_ids"].shape[1] :].tolist()
                    record, region = queries[task]
                    row = {
                        "page_id": record["page_id"],
                        "id": region["id"],
                        "shots": 8,
                        "demo_ids": ids,
                        "negative_demo_image_ids": image_ids,
                        "input_tokens": positive["input_ids"].shape[1],
                        "negative_input_tokens": negative["input_ids"].shape[1],
                        "visual_tokens": visual,
                        "images": 9,
                        "trace": trace,
                        "checks": checks,
                        **decode(tokens),
                    }
                torch.cuda.synchronize()
                row.update(
                    generation_seconds=time.perf_counter() - start,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    app_id=payload["app_id"],
                    call_id=modal.current_function_call_id(),
                )
                _save(path, {"identity": identity, "row": row, "row_sha256": sha_json(row)})
                output_volume.commit()
            if scope == "check" and not row["checks"]:
                raise ValueError("readiness cache has no equivalence evidence")
            rows.append(row)
            print(
                json.dumps({"scope": scope, "completed": len(rows), "task": identity[:12]}),
                flush=True,
            )
    output_volume.commit()
    return {
        "recipe_sha256": digest,
        "scope": scope,
        "rows": rows,
        "gpu": torch.cuda.get_device_name(),
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "model_class": type(model).__name__,
            "processor_class": type(processor).__name__,
        },
    }


def group_queries(records):
    """Collate flattened line chunks into the existing one-record-per-page schema."""
    grouped = {}
    for record in records:
        grouped.setdefault(record["page_id"], record | {"regions": []})["regions"].extend(
            record["regions"]
        )
    result = list(grouped.values())
    _check_records(result, labelled=False)
    return result


def _source(config):
    from active_ocr.pipeline import Pipeline

    source = Pipeline.for_simulation(
        Path(".local/verification/active-learning-source/runs")
    ).get_simulation(config["source_run_id"])
    raw = Path(source.dataset.source_manifest).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config["source_manifest_sha256"]:
        raise ValueError("source manifest differs from pinned evidence")
    pages = {p.id: p for p in source.dataset.pages}
    records = {r["id"]: r["regions"] for r in map(json.loads, raw.splitlines())}
    truth = {pid: {r["id"]: r["text"] for r in rows} for pid, rows in records.items()}
    geometry = {
        pid: [{"id": r["id"], "box": r["box"]} for r in rows] for pid, rows in records.items()
    }
    return source, pages, truth, geometry


def prepare(spec, root):
    from PIL import Image
    from transformers import AutoProcessor

    from active_ocr.region_pipeline import _record

    parent = Path(spec["parent_run"])
    original = json.loads((parent / "recipe.json").read_text())
    config = original["config"]
    assert (spec["shots"], spec["seed"], spec["alpha"], spec["beta"]) == (8, 824, 1, 0.1)
    assert config == json.loads(Path("experiments/recipes/read2016-known-box-icl.json").read_text())
    source, pages, truth, geometry = _source(config)
    bank = json.loads((parent / "bank.json").read_text())
    assert sha_json(bank) == original["bank_sha256"]
    assert len(bank) == 24 and sum(len(r["regions"]) for r in bank) == 589
    for record in bank:
        pid = record["page_id"]
        assert pages[pid].split.value == "train" and record == _record(
            pages[pid], geometry[pid], truth[pid]
        )
    queries = [_record(pages[pid], geometry[pid]) for pid in sorted(config["validation_page_ids"])]
    assert (
        len(queries) == 8 and sum(len(r["regions"]) for r in queries) == spec["pilot_lines"] == 182
    )
    assert all(pages[r["page_id"]].split.value == "validation" for r in queries)
    _check_records(bank, labelled=True)
    _check_records(queries, labelled=False)
    keys = [line_key(p, r) for p in bank for r in p["regions"]]
    targets = {line_key(p, r): r["target"] for p in bank for r in p["regions"]}
    contexts = {
        line_key(p, r): sample_context(keys, line_key(p, r), 8, 824, original["bank_sha256"])
        for p in queries
        for r in p["regions"]
    }
    assert all(
        targets[a] != targets[b]
        for ids in contexts.values()
        for a, b in zip(ids, ids[1:] + ids[:1], strict=True)
    )
    control_root = Path(spec["controls_run"])
    control_recipe = json.loads((control_root / "recipe.json").read_text())
    assert (
        control_recipe["config"] == config
        and control_recipe["bank_sha256"] == original["bank_sha256"]
    )
    references, reference_rows = {}, {}
    for name, path, item in (
        ("zero", Path(spec["paired_pilot"]), "0"),
        ("paired-eight", Path(spec["paired_pilot"]), "8"),
        ("transcripts-only-eight", control_root / "pilot/text-only/results.json", "8"),
        ("mismatched-eight", control_root / "pilot/mismatched/results.json", "8"),
    ):
        data = json.loads(path.read_text())[item]
        receipt = data["receipt"]
        assert receipt["source_run"] == source.id and receipt["adapter_sha256"] is None
        assert (receipt["model_repository"], receipt["model_revision"]) == (
            config["models"]["qwen"]["repository"],
            config["models"]["qwen"]["revision"],
        )
        assert [
            (p["page_id"], [r["id"] for r in p["regions"]]) for p in receipt["predictions"]
        ] == [(p["page_id"], [r["id"] for r in p["regions"]]) for p in queries]
        computed = raw_metrics(receipt, truth)
        assert all(
            computed[k] == data["raw_line"][k]
            for k in (
                "cer",
                "wer",
                "char_edits",
                "word_edits",
                "reference_chars",
                "reference_words",
            )
        )
        rows = {line_key(p, r): r for p in receipt["predictions"] for r in p["regions"]}
        assert all(
            row["demo_ids"] == ([] if name == "zero" else contexts[key])
            for key, row in rows.items()
        )
        reference_rows[name] = rows
        references[name] = {
            "path": str(path),
            "item": item,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "receipt_sha256": sha_json(receipt),
            "reused": True,
            "raw_line": computed,
            "caps": sum(
                r.get("stop_reason", "length" if r["truncated"] else "eos") == "length"
                for r in rows.values()
            ),
        }
    processor = AutoProcessor.from_pretrained(spec["model_assets"], local_files_only=True)
    local_pages = {
        p["page_id"]: Image.open(pages[p["page_id"]].image_uri).convert("RGB") for p in queries
    }
    grids, query_index = {}, {}
    for p in queries:
        for r in p["regions"]:
            key = line_key(p, r)
            grids[key] = _inputs(processor, local_pages[p["page_id"]], r, config)[
                "image_grid_thw"
            ].tolist()
            query_index[key] = (p, r)
    ordered = sorted(query_index, key=lambda k: sha_json([824, "query-swap", k]))
    used, pairs = set(), []
    for anchor in ordered:
        replacements = [
            k
            for k in ordered
            if k not in used
            and k != anchor
            and grids[k] == grids[anchor]
            and query_index[k][0]["page_id"] != query_index[anchor][0]["page_id"]
        ]
        if anchor in used or not replacements:
            continue
        replacement = replacements[0]
        pairs.append(
            {
                "anchor": anchor,
                "replacement": replacement,
                "grid": grids[anchor],
                "prefix_token_ids": reference_rows["transcripts-only-eight"][anchor][
                    "output_token_ids"
                ],
            }
        )
        used.update((anchor, replacement))
        if len(pairs) == spec["diagnostic_pairs"]:
            break
    assert 0 < len(pairs) <= 6
    check = original["check_queries"][: spec["readiness_queries"]]
    native_reference = json.loads((control_root / "check/paired/results.json").read_text())["8"][
        "receipt"
    ]
    native_tokens = {
        line_key(p, r): r["output_token_ids"]
        for p in native_reference["predictions"]
        for r in p["regions"]
    }
    files = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "chat_template.json",
        "tokenizer.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    model_hashes = {}
    for name in files:
        with (Path(spec["model_assets"]) / name).open("rb") as stream:
            model_hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    recipe = {
        "spec": spec,
        "spec_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(),
        "config": config,
        "source_run": source.id,
        "parent_recipe_sha256": sha_json(original),
        "bank_sha256": original["bank_sha256"],
        "code_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (*original["code_sha256"], "icl_controls.py", "contrastive_icl.py")
        },
        "model_files_sha256": model_hashes,
        "references": references,
        "queries": queries,
        "check_queries": check,
        "diagnostic_pairs": pairs,
        "native_check_tokens": {
            line_key(p, r): native_tokens[line_key(p, r)] for p in check for r in p["regions"]
        },
        "input_bundle": original["input_bundle"],
        "modal_image": original["modal_image"],
        "input_volume": original["input_volume"],
        "output_volume": original["output_volume"],
    }
    _save(root / "recipe.json", recipe)
    _save(root / "bank.json", bank)
    print(
        json.dumps(
            {
                "prepared": True,
                "recipe_sha256": sha_json(recipe),
                "pilot_lines": 182,
                "diagnostic_pairs": len(pairs),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "check", "pilot", "diagnose", "report"))
    args = parser.parse_args()
    spec = json.loads(SPEC.read_text())
    root = Path(spec["output_directory"])
    if args.stage == "prepare":
        prepare(spec, root)
        return
    recipe = json.loads((root / "recipe.json").read_text())
    assert (
        recipe["spec"] == spec
        and recipe["spec_sha256"] == hashlib.sha256(SPEC.read_bytes()).hexdigest()
    )
    assert all(
        hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest
        for name, digest in recipe["code_sha256"].items()
    )
    bank = json.loads((root / "bank.json").read_text())
    assert sha_json(bank) == recipe["bank_sha256"]
    digest = sha_json(recipe)
    if args.stage == "check" and (root / "readiness.json").exists():
        gate = json.loads((root / "readiness.json").read_text())
        assert gate["passed"] and gate["recipe_sha256"] == digest and gate["queries"] == 3
        print(json.dumps({"readiness_cache": True, "recipe_sha256": digest}), flush=True)
        return
    if args.stage == "report":
        report(recipe, root)
        return
    if args.stage != "check":
        gate = json.loads((root / "readiness.json").read_text())
        assert gate["passed"] and gate["recipe_sha256"] == digest and gate["queries"] == 3
    records = recipe["check_queries"] if args.stage == "check" else recipe["queries"]
    flat = [p | {"regions": [r]} for p in records for r in p["regions"]]
    if args.stage == "diagnose":
        wanted = {p[k] for p in recipe["diagnostic_pairs"] for k in ("anchor", "replacement")}
        flat = [p for p in flat if line_key(p, p["regions"][0]) in wanted]
    all_rows = []
    with modal.enable_output(), app.run(environment_name="main"):
        for start in range(0, len(flat), 64):
            part = flat[start : start + 64]
            path = root / args.stage / f"part-{start // 64}.json"
            identity = sha_json([digest, args.stage, part])
            result = read_cache(path, identity)
            if result is None:
                payload = {
                    "recipe": recipe,
                    "bank": bank,
                    "queries": group_queries(part),
                    "scope": args.stage,
                    "app_id": app.app_id,
                }
                call = run_chunk.spawn(payload)
                event = {
                    "scope": args.stage,
                    "app_id": app.app_id,
                    "call_id": call.object_id,
                    "start": start,
                    "queries": len(part),
                    "started_utc": datetime.now(UTC).isoformat(),
                }
                with (root / "calls.jsonl").open("a") as stream:
                    stream.write(json.dumps(event) + "\n")
                print(json.dumps(event), flush=True)
                result = call.get()
                _save(path, {"identity": identity, "row": result, "row_sha256": sha_json(result)})
            assert result["recipe_sha256"] == digest and "L4" in result["gpu"]
            all_rows.extend(result["rows"])
    if args.stage == "check":
        assert len(all_rows) == 3 and all(
            r["checks"]["native"] == r["checks"]["alpha_zero"] == r["checks"]["identical"]
            for r in all_rows
        )
        _save(
            root / "readiness.json",
            {"passed": True, "queries": 3, "recipe_sha256": digest, "app_id": app.app_id},
        )
    elif args.stage == "pilot":
        receipt = _receipt(all_rows, records, recipe["config"], recipe["source_run"])
        receipt.update(
            recipe_sha256=digest, study=spec["study"], primary_evaluator=spec["primary_evaluator"]
        )
        _, _, truth, _ = _source(recipe["config"])
        result = {"receipt": receipt, "raw_line": raw_metrics(receipt, truth)}
        _save(root / "pilot/results.json", result)
        print(json.dumps(result["raw_line"]), flush=True)
    else:
        assert len(all_rows) == len(recipe["diagnostic_pairs"])
        _save(root / "diagnose/results.json", {"recipe_sha256": digest, "rows": all_rows})
    print(json.dumps({"stage_completed": args.stage, "rows": len(all_rows)}), flush=True)


def report(recipe, root):
    from active_ocr.evaluation import edit_count, page_text_view

    _, _, truth, _ = _source(recipe["config"])
    pilot = json.loads((root / "pilot/results.json").read_text())
    assert pilot["raw_line"] == raw_metrics(pilot["receipt"], truth)
    expected = {line_key(p, r) for p in recipe["queries"] for r in p["regions"]}
    rows = {line_key(p, r): r for p in pilot["receipt"]["predictions"] for r in p["regions"]}
    assert set(rows) == expected and len(rows) == 182
    comparisons = {}
    for name, ref in recipe["references"].items():
        path = Path(ref["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == ref["file_sha256"]
        old = json.loads(path.read_text())[ref["item"]]["receipt"]
        assert sha_json(old) == ref["receipt_sha256"] and raw_metrics(old, truth) == ref["raw_line"]
        before = {line_key(p, r): r for p in old["predictions"] for r in p["regions"]}
        changes = []
        for key, row in rows.items():
            pid, rid = key.rsplit("/", 1)
            reference = page_text_view([truth[pid][rid]])
            old_edits = edit_count(reference, page_text_view([before[key]["text"]]))
            new_edits = edit_count(reference, page_text_view([row["text"]]))
            changes.append((old_edits, new_edits))
            assert row["demo_ids"] == sample_context(
                [
                    line_key(p, r)
                    for p in json.loads((root / "bank.json").read_text())
                    for r in p["regions"]
                ],
                key,
                8,
                824,
                recipe["bank_sha256"],
            )
            assert len(row["trace"]) == row["tokens"] == len(row["output_token_ids"])
        comparisons[name] = {
            "cer_delta": pilot["raw_line"]["cer"] - ref["raw_line"]["cer"],
            "improved": sum(n < o for o, n in changes),
            "worsened": sum(n > o for o, n in changes),
            "tied": sum(n == o for o, n in changes),
            "previously_correct_damaged": sum(o == 0 < n for o, n in changes),
        }
    diagnostics = json.loads((root / "diagnose/results.json").read_text())
    swap_metrics = []
    for row in diagnostics["rows"]:
        pa, ra = row["anchor"].rsplit("/", 1)
        pb, rb = row["replacement"].rsplit("/", 1)
        a, b = page_text_view([truth[pa][ra]]), page_text_view([truth[pb][rb]])
        before, after = (
            page_text_view([row["original"]["text"]]),
            page_text_view([row["swapped"]["text"]]),
        )
        swap_metrics.append(
            {
                "anchor": row["anchor"],
                "replacement": row["replacement"],
                "output_changed": before != after,
                "replacement_edits_before": edit_count(b, before),
                "replacement_edits_after": edit_count(b, after),
                "replacement_reference_chars": len(b),
                "anchor_edits_before": edit_count(a, before),
                "anchor_edits_after": edit_count(a, after),
                "identical_references": a == b,
                "max_lens_error": max(v["final_max_abs_error"] for v in row["lenses"].values()),
            }
        )
    summary = {
        "recipe_sha256": sha_json(recipe),
        "primary": pilot["raw_line"],
        "references": recipe["references"],
        "comparisons": comparisons,
        "caps": sum(r["stop_reason"] == "length" for r in rows.values()),
        "empty_outputs": sum(not r["text"] for r in rows.values()),
        "generation_seconds": sum(r["generation_seconds"] for r in rows.values()),
        "max_peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in rows.values()),
        "diagnostic_pairs": swap_metrics,
        "stage": "development pilot; not a full experiment",
    }
    _save(root / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
