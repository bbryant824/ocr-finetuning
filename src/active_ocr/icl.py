"""Random line demonstrations for the existing known-box runner; no fitting or retrieval."""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from pathlib import Path

import modal

from active_ocr.evaluation import page_text_metrics
from active_ocr.recognition_pipeline import (
    BUNDLE,
    IMAGE_ID,
    INPUT_VOLUME_ID,
    OUTPUT_VOLUME_ID,
    _load_model,
    image,
    output_volume,
    sha_json,
)
from active_ocr.region_pipeline import (
    MOUNTS,
    _check_records,
    _cuda,
    _inputs,
    _load_page,
    _metrics,
    _record,
    _reused_receipt,
    _save,
    app,
)


def line_key(record, region):
    return record["page_id"] + "/" + region["id"]


def sample_context(bank, query_key, shots, seed, bank_sha):
    if shots not in (0, 8, 24):
        raise ValueError("unsupported ICL shots")
    if shots == 0:
        return []
    if len(set(bank)) != len(bank) or len(bank) < 24:
        raise ValueError("invalid ICL bank")
    if any(":train:" not in key for key in bank):
        raise ValueError("ICL bank must contain TRAIN lines only")
    rng = random.Random(int(sha_json([seed, bank_sha, query_key]), 16))
    return rng.sample(sorted(bank), 24)[:shots]


def read_cache(path, identity):
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    if saved["identity"] != identity:
        raise ValueError("ICL cache identity mismatch")
    if saved["row_sha256"] != sha_json(saved["row"]):
        raise ValueError("ICL cache checksum mismatch")
    return saved["row"]


def require_readiness(path, recipe_sha):
    if not path.exists():
        raise ValueError("full ICL evaluation requires matching passed readiness")
    ready = json.loads(path.read_text())
    if (ready["recipe_sha256"], ready["passed"], ready["outputs"]) != (recipe_sha, True, 24):
        raise ValueError("full ICL evaluation requires matching passed readiness")


def reuse_reference(directory, control, stage):
    """A copied result retains its original executed recipe, not the copy's recipe."""
    saved = json.loads((directory / f"round-{stage}.json").read_text())
    if provenance := saved.get("reused_from"):
        original = Path(provenance["path"])
        raw = original.read_bytes()
        if hashlib.sha256(raw).hexdigest() != provenance["receipt_sha256"]:
            raise ValueError("reference provenance checksum mismatch")
        executed = json.loads(raw)
        if executed["receipt"] != saved["receipt"] or executed["metrics"] != saved["metrics"]:
            raise ValueError("reference provenance differs from copied result")
        directory = original.parent
    return _reused_receipt(directory, control, stage)


def _partial_evaluator(examples, predictions):
    """Score only the diagnostic line from each page, inside the oracle boundary."""
    refs = {e.page.id: {r.id: r.text for r in e.regions} for e in examples}
    return page_text_metrics(
        [tuple(refs[p.page_id][r.id] for r in p.regions) for p in predictions],
        [tuple(r.text for r in p.regions) for p in predictions],
        [p.status for p in predictions],
    )


def _receipt(rows, records, config, source_run):
    index = {(row["page_id"], row["id"]): row for row in rows}
    expected = [
        (record["page_id"], region["id"]) for record in records for region in record["regions"]
    ]
    if len(index) != len(rows) or set(index) != set(expected):
        raise ValueError("ICL prediction coverage mismatch")
    spec = config["models"]["qwen"]
    return {
        "stage": 0,
        "source_run": source_run,
        "model_repository": spec["repository"],
        "model_revision": spec["revision"],
        "adapter_sha256": None,
        "selected_ids": [],
        "gpu": "L4",
        "predictions": [
            {
                "page_id": record["page_id"],
                "regions": [
                    index[(record["page_id"], region["id"])] for region in record["regions"]
                ],
            }
            for record in records
        ],
    }


@app.function(
    image=image.add_local_python_source("active_ocr"),
    gpu="L4",
    cpu=2,
    memory=32768,
    timeout=7200,
    volumes=MOUNTS,
)
def run_queries(payload, run_name):
    """One bounded chunk; durable per-query cache retains work after interruption."""
    import torch

    config, bank, queries = payload["config"], payload["bank"], payload["queries"]
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}", run_name):
        raise ValueError("unsafe ICL run name")
    _check_records(bank, labelled=True)
    _check_records(queries, labelled=False)
    if any(":train:" not in r["page_id"] for r in bank) or any(
        ":validation:" not in r["page_id"] for r in queries
    ):
        raise ValueError("ICL split leakage")
    if sha_json(bank) != payload["bank_sha256"]:
        raise ValueError("ICL bank identity mismatch")
    for name, expected in payload["code_sha256"].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
            raise ValueError("executed ICL code differs from frozen recipe")
    for record in {r["page_id"]: r for r in bank + queries}.values():
        if (
            hashlib.sha256(
                (Path("/input/images") / record["image_sha256"]).read_bytes()
            ).hexdigest()
            != record["image_sha256"]
        ):
            raise ValueError("ICL input image checksum mismatch")
    processor, model = _load_model(config["models"]["qwen"])
    model = model.to("cuda").eval()
    pages = {r["page_id"]: _load_page(r) for r in bank + queries}
    lines = {line_key(r, line): (r, line) for r in bank for line in r["regions"]}
    decode = config["inference"]

    def inputs(record, region, shots):
        ids = sample_context(
            list(lines),
            line_key(record, region),
            shots,
            config["icl"]["seed"],
            payload["bank_sha256"],
        )
        examples = [(pages[lines[key][0]["page_id"]], lines[key][1]) for key in ids]
        batch = _inputs(processor, pages[record["page_id"]], region, config, examples=examples)
        if batch["image_grid_thw"].shape[0] != shots + 1:
            raise ValueError("ICL image count mismatch")
        tokens = batch["input_ids"].shape[1]
        limit = min(
            model.config.text_config.max_position_embeddings, processor.tokenizer.model_max_length
        )
        if tokens + decode["max_new_tokens"] > limit:
            raise ValueError("ICL context exceeds token limit")
        expected = (
            int(batch["image_grid_thw"].prod(dim=1).sum())
            // processor.image_processor.merge_size**2
        )
        visual = int((batch["input_ids"] == model.config.image_token_id).sum())
        if visual != expected:
            raise ValueError("ICL image tokens were lost or truncated")
        return batch, ids, visual

    tasks = [
        (shots, r, line) for shots in payload["shots"] for r in queries for line in r["regions"]
    ]
    if payload["scope"] == "check":
        # The largest actual 24-shot diagnostic prompt is the first generated query.
        lengths = {
            line_key(r, line): inputs(r, line, 24)[0]["input_ids"].shape[1]
            for r in queries
            for line in r["regions"]
        }
        tasks.sort(key=lambda task: lengths[line_key(task[1], task[2])], reverse=True)
        largest = next(task for task in tasks if task[0] == 24)
        tasks.remove(largest)
        tasks.insert(0, largest)
    rows = []
    for shots, record, region in tasks:
        batch, ids, visual = inputs(record, region, shots)
        identity = sha_json(
            [payload["recipe_sha256"], payload["scope"], shots, line_key(record, region), ids]
        )
        path = Path("/output") / run_name / payload["scope"] / f"shot-{shots}" / f"{identity}.json"
        row = read_cache(path, identity)
        if row is None:
            tokens = batch["input_ids"].shape[1]
            batch = _cuda(batch)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **batch,
                    max_new_tokens=decode["max_new_tokens"],
                    do_sample=False,
                    repetition_penalty=decode["repetition_penalty"],
                )
            torch.cuda.synchronize()
            output = generated[0, tokens:]
            row = {
                "page_id": record["page_id"],
                "id": region["id"],
                "shots": shots,
                "demo_ids": ids,
                "text": processor.decode(output, skip_special_tokens=True)
                .strip()
                .replace("\n", " "),
                "tokens": len(output),
                "truncated": len(output) >= decode["max_new_tokens"],
                "input_tokens": tokens,
                "visual_tokens": visual,
                "images": shots + 1,
                "generation_seconds": time.perf_counter() - start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "app_id": payload["app_id"],
                "call_id": modal.current_function_call_id(),
            }
            _save(path, {"identity": identity, "row": row, "row_sha256": sha_json(row)})
            print(
                json.dumps(
                    {
                        "shots": shots,
                        "query": line_key(record, region),
                        "input_tokens": tokens,
                        "seconds": round(row["generation_seconds"], 2),
                    }
                ),
                flush=True,
            )
        if (row["page_id"], row["id"], row["shots"], row["demo_ids"], row["images"]) != (
            record["page_id"],
            region["id"],
            shots,
            ids,
            shots + 1,
        ):
            raise ValueError("cached ICL row differs from query/context")
        rows.append(row)
        if len(rows) % 8 == 0:
            output_volume.commit()
    output_volume.commit()
    return {
        "recipe_sha256": payload["recipe_sha256"],
        "scope": payload["scope"],
        "gpu": torch.cuda.get_device_name(),
        "rows": rows,
    }


def run_icl(args, config):
    from active_ocr.integrations.simulation import LocalOracle
    from active_ocr.models import Split
    from active_ocr.pipeline import Pipeline

    icl = config["icl"]
    if (
        (
            args.strategy,
            args.seed,
            icl["seed"],
            icl["shots"],
            icl["bank_pages"],
            icl["bank_lines"],
            icl["readiness_pages"],
        )
        != ("random", 824, 824, [0, 8, 24], 24, 589, 8)
        or not args.baseline_from
        or not args.fit_from
    ):
        raise ValueError("ICL requires the frozen random pilot and both reference directories")
    source = Pipeline.for_simulation(args.source_runs).get_simulation(config["source_run_id"])
    raw = Path(source.dataset.source_manifest).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config["source_manifest_sha256"]:
        raise ValueError("ICL frozen source differs")
    pages = {p.id: p for p in source.dataset.pages}
    geometry = {
        r["id"]: [{"id": line["id"], "box": line["box"]} for line in r["regions"]]
        for r in map(json.loads, raw.splitlines())
    }
    oracle = LocalOracle(source.dataset)
    val8 = sorted(config["validation_page_ids"])
    val42 = sorted(p.id for p in pages.values() if p.split is Split.VALIDATION and p.id not in val8)
    if (
        len(val8) != 8
        or len(set(val8)) != 8
        or len(val42) != 42
        or any(pages[key].split is not Split.VALIDATION for key in val8)
    ):
        raise ValueError("ICL validation slice mismatch")
    control_config = {key: value for key, value in config.items() if key != "icl"}
    control_config["selection"] = {**control_config["selection"], "seed": args.seed}
    control = {
        "config": control_config,
        "strategy": "random",
        "source_run": source.id,
        "input_bundle": BUNDLE,
        "modal_image": IMAGE_ID,
        "validation_ids": val42,
    }
    references = {}
    for name, directory, stage in (
        ("original", args.baseline_from, 0),
        ("lora24", args.fit_from, 1),
    ):
        saved = reuse_reference(directory, control, stage)
        computed = _metrics(saved["receipt"], pages, geometry, oracle, val42, args.results.name)
        if computed != saved["metrics"]:
            raise ValueError("ICL reference metrics mismatch")
        references[name] = saved
    selection_path = args.fit_from / "selection-round-1.json"
    selection = json.loads(selection_path.read_text())
    selected = selection["selected_ids"]
    if (
        len(selected) != 24
        or len(set(selected)) != 24
        or any(pages[key].split is not Split.TRAIN for key in selected)
        or sha_json(selected) != selection["selected_sha256"]
        or selected != references["lora24"]["receipt"]["selected_ids"]
    ):
        raise ValueError("ICL bank differs from EXP-013 random selection")
    adapter = Path(
        ".local/models/exp-013/random/seed-824/round-1/adapter/adapter_model.safetensors"
    )
    if (
        hashlib.sha256(adapter.read_bytes()).hexdigest()
        != references["lora24"]["receipt"]["adapter_sha256"]
    ):
        raise ValueError("ICL reference adapter checksum differs")
    targets = {e.page.id: {r.id: r.text for r in e.regions} for e in oracle.reveal(tuple(selected))}
    bank = [_record(pages[key], geometry[key], targets[key]) for key in selected]
    _check_records(bank, labelled=True)
    if sum(len(r["regions"]) for r in bank) != 589:
        raise ValueError("ICL bank line count differs")
    full = [_record(pages[key], geometry[key]) for key in val42]
    check = [
        _record(
            pages[key],
            [
                sorted(geometry[key], key=lambda line: (line["box"]["width"], line["id"]))[
                    -1 if index % 2 == 0 else 0
                ]
            ],
        )
        for index, key in enumerate(val8)
    ]
    _check_records(full, labelled=False)
    _check_records(check, labelled=False)
    if sum(len(r["regions"]) for r in full) != 861:
        raise ValueError("ICL VAL42 line count differs")
    code = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("icl.py", "region_pipeline.py", "recognition_pipeline.py", "evaluation.py")
    }
    recipe = {
        "config": config,
        "source_run": source.id,
        "input_bundle": BUNDLE,
        "modal_image": IMAGE_ID,
        "input_volume": INPUT_VOLUME_ID,
        "output_volume": OUTPUT_VOLUME_ID,
        "code_sha256": code,
        "bank_sha256": sha_json(bank),
        "bank_selection": selection,
        "validation_ids": val42,
        "check_queries": check,
        "references": {name: value["reused_from"] for name, value in references.items()},
    }
    digest = sha_json(recipe)
    _save(args.results / "recipe.json", recipe)
    _save(args.results / "references.json", references)
    _save(args.results / "bank.json", bank)
    lines = [line_key(r, line) for r in bank for line in r["regions"]]
    contexts = {
        line_key(r, line): sample_context(lines, line_key(r, line), 24, 824, recipe["bank_sha256"])
        for r in full + check
        for line in r["regions"]
    }
    _save(args.results / "contexts.json", contexts)
    print(
        json.dumps(
            {
                "prepared": True,
                "bank_pages": 24,
                "bank_lines": 589,
                "validation_lines": 861,
                "recipe_sha256": digest,
                "references": {k: v["metrics"] for k, v in references.items()},
            }
        ),
        flush=True,
    )
    if args.icl_stage == "prepare":
        return
    if args.icl_stage == "evaluate":
        require_readiness(args.results / "readiness.json", digest)
    scope = "check" if args.icl_stage == "check" else "full"
    records = check if scope == "check" else full
    modes = [[0, 8, 24]] if scope == "check" else [[8], [24]]
    with modal.enable_output(), app.run(environment_name="main"):
        for shots in modes:
            all_rows = []
            flat = [r | {"regions": [line]} for r in records for line in r["regions"]]
            for start in range(0, len(flat), 64):
                path = (
                    args.results
                    / scope
                    / f"shot-{'-'.join(map(str, shots))}-part-{start // 64}.json"
                )
                identity = sha_json([digest, scope, shots, flat[start : start + 64]])
                saved = read_cache(path, identity)
                if saved is None:
                    payload = {
                        "config": config,
                        "bank": bank,
                        "bank_sha256": recipe["bank_sha256"],
                        "queries": flat[start : start + 64],
                        "scope": scope,
                        "shots": shots,
                        "recipe_sha256": digest,
                        "source_run": source.id,
                        "code_sha256": code,
                        "app_id": app.app_id,
                    }
                    # Merge lines from the same page before validating the chunk.
                    merged = {}
                    for record in payload["queries"]:
                        if record["page_id"] not in merged:
                            merged[record["page_id"]] = record | {"regions": []}
                        merged[record["page_id"]]["regions"].extend(record["regions"])
                    payload["queries"] = list(merged.values())
                    call = run_queries.spawn(payload, args.results.name)
                    call_path = args.results / "calls.jsonl"
                    with call_path.open("a") as stream:
                        stream.write(
                            json.dumps(
                                {
                                    "app_id": app.app_id,
                                    "call_id": call.object_id,
                                    "scope": scope,
                                    "shots": shots,
                                    "start": start,
                                }
                            )
                            + "\n"
                        )
                    saved = call.get()
                    if (
                        saved["recipe_sha256"] != digest
                        or saved["scope"] != scope
                        or "L4" not in saved["gpu"]
                    ):
                        raise ValueError("ICL remote receipt mismatch")
                    _save(path, {"identity": identity, "row": saved, "row_sha256": sha_json(saved)})
                all_rows.extend(saved["rows"])
            reports = {}
            for count in shots:
                rows = [row for row in all_rows if row["shots"] == count]
                receipt = _receipt(rows, records, config, source.id)
                query_geometry = {r["page_id"]: r["regions"] for r in records}
                metrics = _metrics(
                    receipt,
                    pages,
                    query_geometry,
                    oracle,
                    [r["page_id"] for r in records],
                    args.results.name,
                    evaluator=_partial_evaluator if scope == "check" else None,
                )
                reports[str(count)] = {"receipt": receipt, "metrics": metrics}
                _save(args.results / scope / f"shot-{count}.json", reports[str(count)])
                print(json.dumps({"scope": scope, "shots": count, "metrics": metrics}), flush=True)
            if scope == "check":
                _save(
                    args.results / "readiness.json",
                    {
                        "passed": True,
                        "recipe_sha256": digest,
                        "outputs": len(all_rows),
                        "reports": reports,
                        "app_id": app.app_id,
                    },
                )
