"""Bounded EXP-015 decoder ablation using the unchanged production ICL worker."""

import argparse
import copy
import hashlib
import json
import statistics
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import modal

from active_ocr.evaluation import edit_count, page_text_view
from active_ocr.icl import (
    _partial_evaluator,
    _receipt,
    line_key,
    read_cache,
    require_readiness,
    run_queries,
    sample_context,
)
from active_ocr.integrations.simulation import LocalOracle
from active_ocr.pipeline import Pipeline
from active_ocr.recognition_pipeline import sha_json
from active_ocr.region_pipeline import _check_records, _metrics, _record, _save, app

CONFIG = Path("experiments/recipes/exp015-decoding.json")


def decide(results):
    averages = {
        name: {
            "page_cer": statistics.mean(
                results[name][str(n)]["metrics"]["cer"] for n in (0, 8, 24)
            ),
            "raw_line_cer": statistics.mean(
                results[name][str(n)]["raw_line"]["cer"] for n in (0, 8, 24)
            ),
        }
        for name in ("1.1", "1.0")
    }
    chosen = 1.0 if all(averages["1.0"][k] < averages["1.1"][k] for k in averages["1.0"]) else 1.1
    return {"repetition_penalty": chosen, "averages": averages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "check", "pilot", "report"))
    args = parser.parse_args()
    spec = json.loads(CONFIG.read_text())
    assert spec["penalties"] == [1.1, 1.0] and spec["shots"] == [0, 8, 24]
    assert (spec["seed"], spec["pilot_pages"], spec["pilot_lines"], spec["chunk_lines"]) == (
        824,
        8,
        182,
        64,
    )
    root, parent = Path(spec["output_directory"]), Path(spec["parent_run"])
    original = json.loads((parent / "recipe.json").read_text())
    assert sha_json(original) == "72119aa627a9211ca6363ec061637a572f5062a2c395e4510477c512b460e34d"
    bank = json.loads((parent / "bank.json").read_text())
    config = original["config"]
    source = Pipeline.for_simulation(
        Path(".local/verification/active-learning-source/runs")
    ).get_simulation(original["source_run"])
    raw = Path(source.dataset.source_manifest).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == config["source_manifest_sha256"]
    oracle = LocalOracle(source.dataset)
    pages = {p.id: p for p in source.dataset.pages}
    truth = {p["id"]: p["regions"] for p in map(json.loads, raw.splitlines())}
    geometry = {
        pid: [{"id": r["id"], "box": r["box"]} for r in rows] for pid, rows in truth.items()
    }
    ids = sorted(config["validation_page_ids"])
    assert len(ids) == len(set(ids)) == 8 and not set(ids) & set(original["validation_ids"])
    assert all(pages[pid].split.value == "validation" for pid in ids)
    records = [_record(pages[pid], geometry[pid]) for pid in ids]
    check = original["check_queries"]
    assert len(bank) == 24 and sha_json(bank) == original["bank_sha256"]
    assert sum(len(p["regions"]) for p in bank) == 589
    for p in bank:
        assert pages[p["page_id"]].split.value == "train"
        assert [(r["id"], r["box"], r["target"]) for r in p["regions"]] == [
            (r["id"], r["box"], r["text"]) for r in truth[p["page_id"]]
        ]
    assert sum(len(p["regions"]) for p in records) == 182
    assert sum(len(p["regions"]) for p in check) == 8
    _check_records(bank, labelled=True)
    _check_records(records, labelled=False)
    _check_records(check, labelled=False)
    code = {}
    for name, expected in original["code_sha256"].items():
        code[name] = hashlib.sha256((Path("src/active_ocr") / name).read_bytes()).hexdigest()
        assert code[name] == expected, "production worker changed from EXP-015"
    plan = {
        "spec": spec,
        "parent_recipe_sha256": sha_json(original),
        "bank_sha256": original["bank_sha256"],
        "source_manifest_sha256": config["source_manifest_sha256"],
        "code_sha256": code,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_file_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "pilot_queries": records,
        "check_queries": check,
    }
    _save(root / "plan.json", plan)
    keys = [line_key(p, r) for p in bank for r in p["regions"]]
    contexts = {
        line_key(p, r): sample_context(keys, line_key(p, r), 24, 824, original["bank_sha256"])
        for p in records
        for r in p["regions"]
    }
    _save(root / "contexts.json", contexts)
    recipes = {}
    for penalty in spec["penalties"]:
        arm = copy.deepcopy(original)
        arm["config"]["inference"]["repetition_penalty"] = penalty
        arm["validation_ids"] = ids
        arm["decoder_ablation"] = {
            "plan_sha256": sha_json(plan),
            "penalty": penalty,
            "slice": "VAL8",
        }
        recipes[str(penalty)] = arm
        _save(root / f"penalty-{penalty}" / "recipe.json", arm)
    print(
        json.dumps(
            {"prepared": True, "pilot_pages": 8, "pilot_lines": 182, "plan_sha256": sha_json(plan)}
        ),
        flush=True,
    )
    if args.stage == "prepare":
        return

    def audit(rows, query_records, arm, scope):
        expected = {line_key(p, r) for p in query_records for r in p["regions"]}
        query_geometry = {p["page_id"]: p["regions"] for p in query_records}
        reports = {}
        for shots in spec["shots"]:
            selected = [r for r in rows if r["shots"] == shots]
            assert len(selected) == len(expected)
            assert {r["page_id"] + "/" + r["id"] for r in selected} == expected
            char_edits = word_edits = chars = words = cyrillic = 0
            for row in selected:
                key = row["page_id"] + "/" + row["id"]
                assert row["demo_ids"] == sample_context(
                    keys, key, shots, 824, original["bank_sha256"]
                )
                assert row["images"] == shots + 1 and row["input_tokens"] > row["visual_tokens"] > 0
                assert 0 < row["tokens"] <= 64 and row["truncated"] == (row["tokens"] == 64)
                target = next(r["text"] for r in truth[row["page_id"]] if r["id"] == row["id"])
                reference, hypothesis = page_text_view([target]), page_text_view([row["text"]])
                char_edits += edit_count(reference, hypothesis)
                word_edits += edit_count(reference.split(), hypothesis.split())
                chars += len(reference)
                words += len(reference.split())
                cyrillic += int(
                    any("CYRILLIC" in unicodedata.name(c, "") for c in hypothesis)
                    and not any("CYRILLIC" in unicodedata.name(c, "") for c in reference)
                )
            receipt = _receipt(selected, query_records, arm["config"], source.id)
            receipt["recipe_sha256"] = sha_json(arm)
            receipt["validation_ids"] = [p["page_id"] for p in query_records]
            metrics = _metrics(
                receipt,
                pages,
                query_geometry,
                oracle,
                receipt["validation_ids"],
                "EXP-015-decoding",
                evaluator=_partial_evaluator if scope == "check" else None,
            )
            reports[str(shots)] = {
                "receipt": receipt,
                "metrics": metrics,
                "raw_line": {
                    "cer": char_edits / chars,
                    "wer": word_edits / words,
                    "char_edits": char_edits,
                    "word_edits": word_edits,
                    "reference_chars": chars,
                    "reference_words": words,
                },
                "unexpected_cyrillic_lines": cyrillic,
                "max_input_tokens": max(r["input_tokens"] for r in selected),
                "generation_seconds": sum(r["generation_seconds"] for r in selected),
            }
        return reports

    if args.stage == "report":
        results = {}
        for penalty, arm in recipes.items():
            saved = json.loads((root / f"penalty-{penalty}" / "pilot.json").read_text())
            rows = [
                r
                for result in saved.values()
                for p in result["receipt"]["predictions"]
                for r in p["regions"]
            ]
            results[penalty] = audit(rows, records, arm, "pilot")
            assert results[penalty] == saved
        decision = decide(results) | {
            "plan_sha256": sha_json(plan),
            "rule": spec["decision_rule"],
            "scope": "VAL8 development tuning; no VAL42 rerun or main five-condition study",
            "all_outputs": 1092,
        }
        _save(root / "decision.json", decision)
        print(json.dumps(decision), flush=True)
        return

    scope = "check" if args.stage == "check" else "pilot"
    queries = check if scope == "check" else records
    if scope == "pilot":
        for penalty, arm in recipes.items():
            require_readiness(root / f"penalty-{penalty}" / "readiness.json", sha_json(arm))
    with modal.enable_output(), app.run(environment_name="main"):
        for penalty, arm in recipes.items():
            digest = sha_json(arm)
            run_name = f"icl-exp015-decoding-p{penalty.replace('.', '')}-seed824"
            flat = [_record(pages[p["page_id"]], [r]) for p in queries for r in p["regions"]]
            rows = []
            for start in range(0, len(flat), 64):
                part = flat[start : start + 64]
                identity = sha_json([digest, scope, part, spec["shots"]])
                path = root / f"penalty-{penalty}" / scope / f"part-{start // 64}.json"
                saved = read_cache(path, identity)
                if saved is None:
                    merged = {}
                    for p in part:
                        merged.setdefault(p["page_id"], p | {"regions": []})["regions"].extend(
                            p["regions"]
                        )
                    payload = {
                        "config": arm["config"],
                        "bank": bank,
                        "bank_sha256": original["bank_sha256"],
                        "queries": list(merged.values()),
                        "scope": scope,
                        "shots": spec["shots"],
                        "recipe_sha256": digest,
                        "source_run": source.id,
                        "code_sha256": code,
                        "app_id": app.app_id,
                    }
                    call = run_queries.spawn(payload, run_name)
                    event = {
                        "app_id": app.app_id,
                        "call_id": call.object_id,
                        "penalty": penalty,
                        "scope": scope,
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
                    and saved["scope"] == scope
                    and "L4" in saved["gpu"]
                )
                rows.extend(saved["rows"])
            reports = audit(rows, queries, arm, scope)
            _save(root / f"penalty-{penalty}" / f"{scope}.json", reports)
            if scope == "check":
                _save(
                    root / f"penalty-{penalty}" / "readiness.json",
                    {
                        "passed": True,
                        "outputs": len(rows),
                        "recipe_sha256": digest,
                        "app_id": app.app_id,
                    },
                )
            print(
                json.dumps(
                    {
                        "penalty": penalty,
                        "scope": scope,
                        "results": {
                            n: {k: v for k, v in r.items() if k != "receipt"}
                            for n, r in reports.items()
                        },
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
