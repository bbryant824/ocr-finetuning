"""Content ablations for the existing ICL worker; no fitting or new orchestration."""

import argparse
import hashlib
import json
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import modal

from active_ocr.evaluation import page_text_metrics
from active_ocr.icl import (
    _partial_evaluator,
    _receipt,
    line_key,
    read_cache,
    run_queries,
    sample_context,
)
from active_ocr.integrations.simulation import LocalOracle
from active_ocr.models import PredictionStatus
from active_ocr.pipeline import Pipeline
from active_ocr.recognition_pipeline import sha_json
from active_ocr.region_pipeline import _check_records, _metrics, _record, _save, app

SPEC = Path("experiments/recipes/read2016-known-box-icl-controls.json")


def raw_metrics(receipt, truth):
    """Score every raw line, including caps, with boundaries fixed by source IDs."""
    references, hypotheses = [], []
    for page in receipt["predictions"]:
        for row in page["regions"]:
            references.append([truth[page["page_id"]][row["id"]]])
            hypotheses.append([row["text"]])
    result = page_text_metrics(references, hypotheses, [PredictionStatus.OK] * len(references))
    result["lines"] = result.pop("pages")
    return result


def require_gate(path, digest, outputs):
    if not path.exists():
        raise ValueError("matching completed ICL control gate required")
    saved = json.loads(path.read_text())
    if (saved["passed"], saved["recipe_sha256"], saved["outputs"]) != (True, digest, outputs):
        raise ValueError("matching completed ICL control gate required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "check", "pilot", "full", "report"))
    stage = parser.parse_args().stage
    spec = json.loads(SPEC.read_text())
    assert spec["demonstration_modes"] == ["text-only", "mismatched"]
    assert spec["shots"] == [8, 24] and spec["seed"] == 824 and spec["chunk_lines"] == 64
    root, parent = Path(spec["output_directory"]), Path(spec["parent_run"])
    original = json.loads((parent / "recipe.json").read_text())
    assert sha_json(original) == "72119aa627a9211ca6363ec061637a572f5062a2c395e4510477c512b460e34d"
    config = json.loads(Path(spec["base_recipe"]).read_text())
    assert config == original["config"] and config["inference"]["repetition_penalty"] == 1.1
    bank = json.loads((parent / "bank.json").read_text())
    assert sha_json(bank) == original["bank_sha256"]
    source = Pipeline.for_simulation(
        Path(".local/verification/active-learning-source/runs")
    ).get_simulation(original["source_run"])
    raw = Path(source.dataset.source_manifest).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == config["source_manifest_sha256"]
    oracle = LocalOracle(source.dataset)
    pages = {p.id: p for p in source.dataset.pages}
    source_rows = {p["id"]: p["regions"] for p in map(json.loads, raw.splitlines())}
    truth = {pid: {r["id"]: r["text"] for r in rows} for pid, rows in source_rows.items()}
    geometry = {
        pid: [{"id": r["id"], "box": r["box"]} for r in rows] for pid, rows in source_rows.items()
    }
    for page in bank:
        assert pages[page["page_id"]].split.value == "train"
        assert page == _record(
            pages[page["page_id"]], geometry[page["page_id"]], truth[page["page_id"]]
        )
    assert len(bank) == 24 and sum(len(p["regions"]) for p in bank) == 589
    scopes = {
        "check": original["check_queries"],
        "pilot": [
            _record(pages[pid], geometry[pid]) for pid in sorted(config["validation_page_ids"])
        ],
        "full": [_record(pages[pid], geometry[pid]) for pid in original["validation_ids"]],
    }
    assert [sum(len(p["regions"]) for p in scopes[s]) for s in scopes] == [8, 182, 861]
    assert not {p["page_id"] for p in scopes["pilot"]} & {p["page_id"] for p in scopes["full"]}
    for records in scopes.values():
        _check_records(records, labelled=False)
        assert all(pages[p["page_id"]].split.value == "validation" for p in records)
    _check_records(bank, labelled=True)
    code = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (*original["code_sha256"], "icl_controls.py")
    }
    recipe = original | {
        "experiment": "EXP-016",
        "controls": spec,
        "code_sha256": code,
        "spec_file_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(),
        "queries": scopes,
    }
    digest = sha_json(recipe)
    _save(root / "recipe.json", recipe)
    _save(root / "bank.json", bank)
    keys = [line_key(p, r) for p in bank for r in p["regions"]]
    bank_targets = {line_key(p, r): r["target"] for p in bank for r in p["regions"]}
    contexts = {
        line_key(p, r): sample_context(keys, line_key(p, r), 24, 824, original["bank_sha256"])
        for records in scopes.values()
        for p in records
        for r in p["regions"]
    }
    _save(root / "contexts.json", contexts)

    def summarize(receipt, records, scope, provenance=None):
        assert [p["page_id"] for p in receipt["predictions"]] == [p["page_id"] for p in records]
        assert (receipt["source_run"], receipt["model_repository"], receipt["model_revision"]) == (
            source.id,
            config["models"]["qwen"]["repository"],
            config["models"]["qwen"]["revision"],
        )
        for page, record in zip(receipt["predictions"], records, strict=True):
            assert [r["id"] for r in page["regions"]] == [r["id"] for r in record["regions"]]
        rows = [r for p in receipt["predictions"] for r in p["regions"]]
        computed = _metrics(
            receipt,
            pages,
            {p["page_id"]: p["regions"] for p in records},
            oracle,
            [p["page_id"] for p in records],
            "EXP-016",
            evaluator=_partial_evaluator if scope == "check" else None,
        )
        return {
            "receipt": receipt,
            "page_metrics": computed,
            "raw_line": raw_metrics(receipt, truth),
            "unexpected_cyrillic_lines": sum(
                any("CYRILLIC" in unicodedata.name(c, "") for c in r["text"])
                and not any(
                    "CYRILLIC" in unicodedata.name(c, "") for c in truth[p["page_id"]][r["id"]]
                )
                for p in receipt["predictions"]
                for r in p["regions"]
            ),
            "provenance": provenance,
            "generation_seconds": sum(r.get("generation_seconds", 0) for r in rows)
            if all("generation_seconds" in r for r in rows)
            else None,
        }

    def reference(path, scope, item=None):
        saved = json.loads(path.read_text())
        if item is not None:
            saved = saved[item]
        receipt, metrics = saved["receipt"], saved["metrics"]
        computed = summarize(
            receipt,
            scopes[scope],
            scope,
            {
                "path": str(path),
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "executed_recipe_sha256": receipt.get("recipe_sha256"),
                "reused": True,
            },
        )
        assert computed["page_metrics"] == metrics
        return computed

    refs = {"pilot": {}, "full": {}}
    pilot_path = Path(spec["pilot_reference"]) / "pilot.json"
    pilot_recipe = json.loads(pilot_path.with_name("recipe.json").read_text())
    assert (
        pilot_recipe["config"] == config and pilot_recipe["bank_sha256"] == original["bank_sha256"]
    )
    assert pilot_recipe["validation_ids"] == [p["page_id"] for p in scopes["pilot"]]
    for n in (0, 8, 24):
        refs["pilot"]["zero" if n == 0 else f"paired-{n}"] = reference(pilot_path, "pilot", str(n))
    for n in (8, 24):
        refs["full"][f"paired-{n}"] = reference(parent / "full" / f"shot-{n}.json", "full")
    historical = json.loads((parent / "references.json").read_text())
    for name, key in (("original", "zero"), ("lora24", "lora24")):
        origin = historical[name]["reused_from"]
        path = Path(origin["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == origin["receipt_sha256"]
        executed = json.loads(path.read_text())
        assert executed["receipt"] == historical[name]["receipt"]
        ref_recipe = json.loads(path.with_name("recipe.json").read_text())
        assert sha_json(ref_recipe) == origin["recipe_sha256"]
        for field in ("source_run", "input_bundle", "modal_image", "validation_ids"):
            assert ref_recipe[field] == original[field]
        old_config = {
            k: v for k, v in ref_recipe["config"].items() if k not in ("icl", "selection")
        }
        assert old_config == {k: v for k, v in config.items() if k not in ("icl", "selection")}
        refs["full"][key] = reference(path, "full")
        if key == "lora24":
            assert executed["receipt"]["selected_ids"] == [p["page_id"] for p in bank]
            weights = Path(
                ".local/models/exp-013/random/seed-824/round-1/adapter/adapter_model.safetensors"
            )
            assert (
                hashlib.sha256(weights.read_bytes()).hexdigest()
                == executed["receipt"]["adapter_sha256"]
            )
    for scope, reports in refs.items():
        _save(root / scope / "references.json", reports)
    print(
        json.dumps(
            {
                "prepared": True,
                "recipe_sha256": digest,
                "references": {
                    s: {k: v["raw_line"]["cer"] for k, v in rs.items()} for s, rs in refs.items()
                },
            }
        ),
        flush=True,
    )
    if stage == "prepare":
        return

    def audit(rows, mode, records, scope):
        reports = {}
        counts = (0, 8, 24) if mode == "paired" else spec["shots"]
        assert len(rows) == len(counts) * sum(len(p["regions"]) for p in records)
        for n in counts:
            selected = [r for r in rows if r["shots"] == n]
            for row in selected:
                key = row["page_id"] + "/" + row["id"]
                ids = sample_context(keys, key, n, 824, original["bank_sha256"])
                image_ids = (
                    []
                    if mode == "text-only"
                    else ids[1:] + ids[:1]
                    if mode == "mismatched"
                    else ids
                )
                assert row["demo_ids"] == ids and row["demo_image_ids"] == image_ids
                assert row["demonstration_mode"] == mode and row["images"] == (
                    1 if mode == "text-only" else n + 1
                )
                assert row["input_tokens"] > row["visual_tokens"] > 0
                assert len(row["output_token_ids"]) == row["tokens"] and 0 < row["tokens"] <= 64
                assert row["same_target_pairs"] == sum(
                    bank_targets[a] == bank_targets[b] for a, b in zip(ids, image_ids, strict=False)
                )
                assert row["truncated"] == (row["tokens"] == 64)
                assert row["stop_reason"] == (
                    "eos" if row["ended_with_eos"] else "length" if row["truncated"] else "other"
                )
                if mode == "mismatched":
                    assert all(a != b for a, b in zip(ids, image_ids, strict=True))
            receipt = _receipt(selected, records, config, source.id)
            receipt.update(
                recipe_sha256=digest,
                validation_ids=[p["page_id"] for p in records],
                demonstration_mode=mode,
                shots=n,
            )
            reports[str(n)] = summarize(receipt, records, scope)
        return reports

    if stage == "report":
        combined = {"recipe_sha256": digest, "references": refs, "new": {}}
        for scope in ("pilot", "full"):
            combined["new"][scope] = {}
            for mode in spec["demonstration_modes"]:
                saved = json.loads((root / scope / mode / "results.json").read_text())
                rows = [
                    r
                    for v in saved.values()
                    for p in v["receipt"]["predictions"]
                    for r in p["regions"]
                ]
                assert audit(rows, mode, scopes[scope], scope) == saved
                combined["new"][scope][mode] = saved
        _save(root / "summary.json", combined)
        print(
            json.dumps(
                {
                    s: {m: {n: v["raw_line"] for n, v in rs.items()} for m, rs in modes.items()}
                    for s, modes in combined["new"].items()
                }
            ),
            flush=True,
        )
        return

    if stage in ("pilot", "full"):
        require_gate(root / "readiness.json", digest, 56)
    if stage == "full":
        require_gate(root / "pilot-ready.json", digest, 728)
    records = scopes[stage]
    modes = (
        ["paired", *spec["demonstration_modes"]]
        if stage == "check"
        else spec["demonstration_modes"]
    )
    with modal.enable_output(), app.run(environment_name="main"):
        outputs = 0
        for mode in modes:
            shots = [0, 8, 24] if mode == "paired" else spec["shots"]
            flat = [p | {"regions": [r]} for p in records for r in p["regions"]]
            all_rows = []
            for start in range(0, len(flat), 64):
                part = flat[start : start + 64]
                identity = sha_json([digest, stage, mode, shots, part])
                path = root / stage / mode / f"part-{start // 64}.json"
                saved = read_cache(path, identity)
                if saved is None:
                    merged = {}
                    for p in part:
                        merged.setdefault(p["page_id"], p | {"regions": []})["regions"].extend(
                            p["regions"]
                        )
                    payload = {
                        "config": config,
                        "bank": bank,
                        "bank_sha256": original["bank_sha256"],
                        "queries": list(merged.values()),
                        "scope": stage,
                        "shots": shots,
                        "recipe_sha256": digest,
                        "source_run": source.id,
                        "code_sha256": code,
                        "app_id": app.app_id,
                        "demonstration_mode": mode,
                    }
                    call = run_queries.spawn(payload, f"icl-exp016-{mode}-seed824")
                    event = {
                        "app_id": app.app_id,
                        "call_id": call.object_id,
                        "scope": stage,
                        "mode": mode,
                        "start": start,
                        "lines": len(part),
                        "started_utc": datetime.now(UTC).isoformat(),
                    }
                    with (root / "calls.jsonl").open("a") as stream:
                        stream.write(json.dumps(event) + "\n")
                    print(json.dumps(event), flush=True)
                    saved = call.get()
                    _save(path, {"identity": identity, "row": saved, "row_sha256": sha_json(saved)})
                assert (
                    saved["recipe_sha256"] == digest
                    and saved["scope"] == stage
                    and "L4" in saved["gpu"]
                )
                all_rows.extend(saved["rows"])
            reports = audit(all_rows, mode, records, stage)
            _save(root / stage / mode / "results.json", reports)
            outputs += len(all_rows)
            if mode == "paired":
                for n in (0, 8, 24):
                    old = json.loads((parent / "check" / f"shot-{n}.json").read_text())
                    assert reports[str(n)]["page_metrics"] == old["metrics"]
                    before = {
                        (p["page_id"], r["id"]): r
                        for p in old["receipt"]["predictions"]
                        for r in p["regions"]
                    }
                    for p in reports[str(n)]["receipt"]["predictions"]:
                        for row in p["regions"]:
                            assert all(
                                row[k] == before[(p["page_id"], row["id"])][k]
                                for k in (
                                    "text",
                                    "tokens",
                                    "demo_ids",
                                    "input_tokens",
                                    "visual_tokens",
                                    "images",
                                )
                            )
            print(
                json.dumps(
                    {
                        "scope": stage,
                        "mode": mode,
                        "results": {
                            n: {
                                "raw_line": v["raw_line"],
                                "caps": v["page_metrics"]["truncated_regions"],
                            }
                            for n, v in reports.items()
                        },
                    }
                ),
                flush=True,
            )
        if stage in ("check", "pilot"):
            assert outputs == (56 if stage == "check" else 728)
            _save(
                root / ("readiness.json" if stage == "check" else "pilot-ready.json"),
                {"passed": True, "recipe_sha256": digest, "outputs": outputs, "app_id": app.app_id},
            )


if __name__ == "__main__":
    main()
