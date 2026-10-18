"""EXP-018 saved-output audit and reference-prefix correctness evaluation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import modal

from active_ocr.contrastive_icl import _capture, _source, check_reference, contrastive_scores
from active_ocr.evaluation import page_text_view
from active_ocr.icl import check_input_capacity, line_key, read_cache, sample_context
from active_ocr.icl_controls import raw_metrics
from active_ocr.query_intervention import prefix_state
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

SPEC = Path("experiments/recipes/read2016-known-box-contrastive-diagnostic.json")
MODES = (
    "raw_paired",
    "raw_contrastive",
    "paired",
    "unmasked_contrastive",
    "paired_masked",
    "contrastive",
)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def token_readouts(positive, negative, ids, processors, gold, eos, terminal=False):
    """Record canonical-token decisions; EOS event mass/choice is separate."""
    import torch

    paired = processors(ids, positive.clone())
    eligible = paired - paired.max(-1, keepdim=True).values >= math.log(0.1)
    scores = dict(
        zip(
            MODES,
            (
                positive,
                2 * positive - negative,
                paired,
                processors(ids, 2 * positive - negative),
                paired.masked_fill(~eligible, -torch.inf),
                contrastive_scores(positive, negative, ids, processors),
            ),
            strict=True,
        )
    )
    result = {}
    for name, score in scores.items():
        vector = score[0].double()
        top = int(vector.argmax())
        normalizer = vector.logsumexp(-1)
        admitted = bool(torch.isfinite(vector[gold]))
        other = vector.clone()
        other[gold] = -torch.inf
        result[name] = {
            "top": top,
            "correct": top in eos if terminal else top == gold,
            "rank": 1 + int((vector > vector[gold]).sum()) if admitted else None,
            "nll": float(normalizer - vector[gold]) if admitted else None,
            "margin": float(vector[gold] - other.max()) if admitted else None,
            "excluded": not admitted,
            "eos_mass": float((vector[eos].logsumexp(-1) - normalizer).exp()),
            "stops": top in eos,
        }
    return result


def forced_states(model, batch, target, settings):
    """Yield next-token logits before the current target enters either cache."""
    import torch

    state = model.generate(**batch, **settings, custom_generate=_capture)
    for step, token in enumerate(target):
        if step == 0:
            out = model._prefill(state["ids"], state["config"], state["kwargs"])
        else:
            inputs = model.prepare_inputs_for_generation(
                state["ids"], next_sequence_length=1, **state["kwargs"]
            )
            out = model(**inputs, return_dict=True)
        state["kwargs"] = model._update_model_kwargs_for_generation(out, state["kwargs"])
        cache_tokens = state["kwargs"]["past_key_values"].get_seq_length()
        assert cache_tokens == state["ids"].shape[1]
        logits = out.logits[:, -1].to(copy=True, dtype=torch.float32)
        assert torch.isfinite(logits).all()
        del out
        yield {
            "logits": logits,
            "ids": state["ids"],
            "processors": state["processors"],
            "cache_tokens": cache_tokens,
        }
        state["ids"] = torch.cat((state["ids"], state["ids"].new_tensor([[token]])), -1)


def canonical_target(processor, text, eos):
    """Use native assistant boundaries, retaining only text and its first terminator."""
    messages = [{"role": "user", "content": [{"type": "text", "text": "Transcribe."}]}]
    before = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    after = processor.apply_chat_template(
        messages + [{"role": "assistant", "content": [{"type": "text", "text": text}]}],
        add_generation_prompt=False,
        tokenize=False,
    )

    def encode(s):
        return processor.tokenizer.encode(s, add_special_tokens=False)

    prefix, full = encode(before), encode(after)
    assert full[: len(prefix)] == prefix, "assistant boundary merges with reference"
    suffix = full[len(prefix) :]
    end = next(i for i, t in enumerate(suffix) if t in eos)
    target = suffix[: end + 1]
    assert processor.decode(target[:-1], skip_special_tokens=False) == text
    assert processor.decode(suffix[end + 1 :], skip_special_tokens=False).strip() == ""
    return target, prefix


def saved_output_audit(parent, truth):
    from rapidfuzz.distance import Levenshtein

    original = json.loads((parent / "recipe.json").read_text())
    saved = json.loads((parent / "full/results.json").read_text())
    queries, config = original["queries"], original["config"]
    bank = json.loads((parent / "bank.json").read_text())
    keys = [line_key(p, r) for p in bank for r in p["regions"]]
    contexts = {
        line_key(p, r): sample_context(keys, line_key(p, r), 8, 824, sha_json(bank))
        for p in queries
        for r in p["regions"]
    }
    receipts = {"contrastive": saved}
    files = {
        str(parent / "recipe.json"): file_hash(parent / "recipe.json"),
        str(parent / "full/results.json"): file_hash(parent / "full/results.json"),
    }
    for mode, name in (("paired", "paired-eight"), ("text-only", "transcripts-only-eight")):
        ref = original["references"][name]
        assert file_hash(ref["path"]) == ref["file_sha256"]
        data = json.loads(Path(ref["path"]).read_text())
        data = data[ref["item"]] if ref["item"] is not None else data
        assert sha_json(data["receipt"]) == ref["receipt_sha256"]
        assert raw_metrics(data["receipt"], truth) == ref["raw_line"]
        receipts[mode] = data
        files[ref["path"]] = ref["file_sha256"]
    indices = {
        m: check_reference(x["receipt"], queries, config, original["source_run"], contexts)
        for m, x in receipts.items()
    }
    metrics = {m: raw_metrics(x["receipt"], truth) for m, x in receipts.items()}
    assert metrics["contrastive"] == saved["raw_line"]
    assert metrics["contrastive"]["lines"] == 861
    rows = []
    for p in queries:
        for region in p["regions"]:
            key = line_key(p, region)
            target = page_text_view([truth[p["page_id"]][region["id"]]])
            values = {}
            for mode, index in indices.items():
                row = index[key]
                text = page_text_view([row["text"]])
                ops = Counter(op.tag for op in Levenshtein.editops(target, text))
                distance = Levenshtein.distance(target, text)
                assert sum(ops.values()) == distance
                words = text.split()
                grams = Counter(tuple(words[i : i + 3]) for i in range(len(words) - 2))
                values[mode] = {
                    "text": text,
                    "edits": distance,
                    "chars": len(text),
                    "operations": {k: ops[k] for k in ("replace", "delete", "insert")},
                    "repeated_trigrams": sum(max(0, n - 1) for n in grams.values()),
                    "limit_hit": row.get("tokens", len(row.get("output_token_ids", []))) >= 64,
                    "stop_reason": row.get("stop_reason"),
                    "exact": text == target,
                }
            rows.append(
                {"key": key, "page_id": p["page_id"], "reference": target, "predictions": values}
            )
    aggregates = {}
    for mode in indices:
        values = [r["predictions"][mode] for r in rows]
        assert sum(v["edits"] for v in values) == metrics[mode]["char_edits"]
        aggregates[mode] = {
            "metrics": metrics[mode],
            "chars": sum(v["chars"] for v in values),
            "exact": sum(v["exact"] for v in values),
            "repeated_trigrams": sum(v["repeated_trigrams"] for v in values),
            "limit_hits": sum(v["limit_hit"] for v in values),
            "operations": {
                op: sum(v["operations"][op] for v in values)
                for op in ("replace", "delete", "insert")
            },
        }
    comparisons, examples = {}, []
    for mode in ("paired", "text-only"):
        deltas = {
            r["key"]: r["predictions"][mode]["edits"] - r["predictions"]["contrastive"]["edits"]
            for r in rows
        }
        per_page = {
            p: sum(deltas[r["key"]] for r in rows if r["page_id"] == p)
            for p in sorted({r["page_id"] for r in rows})
        }
        comparisons[mode] = {
            "saved_edits": sum(deltas.values()),
            "improved": sum(d > 0 for d in deltas.values()),
            "harmed": sum(d < 0 for d in deltas.values()),
            "tied": sum(d == 0 for d in deltas.values()),
            "per_page_saved_edits": per_page,
            "saved_edits_outside_limit_hits": sum(
                deltas[r["key"]]
                for r in rows
                if not any(r["predictions"][m]["limit_hit"] for m in (mode, "contrastive"))
            ),
        }
        for label, sign in (("improved", 1), ("harmed", -1)):
            candidates = [r for r in rows if sign * deltas[r["key"]] > 0]
            chosen = sorted(candidates, key=lambda r: sha_json([824, mode, label, r["key"]]))[:2]
            examples.extend({"comparator": mode, "outcome": label, **r} for r in chosen)
    return {
        "source_files_sha256": files,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "examples": examples,
        "rows": rows,
        "alignment": "rapidfuzz optimal editops(reference, hypothesis); ties possible",
    }


def prepare(spec):
    from transformers import AutoProcessor

    root = Path(spec["output_directory"])
    parent = Path(spec["pilot_run"])
    original = json.loads((parent / "recipe.json").read_text())
    config = original["config"]
    source, pages, truth, _ = _source(config)
    assert file_hash(Path(spec["full_run"]) / "full/results.json") == spec["full_results_sha256"]
    audit = saved_output_audit(Path(spec["full_run"]), truth)
    _save(root / "step-1.json", audit)
    bank = json.loads((parent / "bank.json").read_text())
    queries = original["queries"]
    _check_records(bank, labelled=True)
    _check_records(queries, labelled=False)
    assert (len(bank), sum(len(p["regions"]) for p in bank)) == (24, 589)
    assert sha_json(bank) == original["bank_sha256"]
    assert len(queries) == 8 and sum(len(p["regions"]) for p in queries) == 182
    assert all(pages[p["page_id"]].split.value == "train" for p in bank)
    assert all(pages[p["page_id"]].split.value == "validation" for p in queries)
    assert not {p["image_sha256"] for p in bank} & {p["image_sha256"] for p in queries}
    for name, expected in original["model_files_sha256"].items():
        assert file_hash(Path(spec["model_assets"]) / name) == expected
    processor = AutoProcessor.from_pretrained(spec["model_assets"], local_files_only=True)
    eos = json.loads((Path(spec["model_assets"]) / "generation_config.json").read_text())[
        "eos_token_id"
    ]
    keys = [line_key(p, r) for p in bank for r in p["regions"]]
    cases = []
    for p in queries:
        for r in p["regions"]:
            key = line_key(p, r)
            target = page_text_view([truth[p["page_id"]][r["id"]]])
            tokens, native_prefix = canonical_target(processor, target, eos)
            demo_ids = sample_context(keys, key, 8, 824, sha_json(bank))
            cases.append(
                {
                    "key": key,
                    "page_id": p["page_id"],
                    "reference": target,
                    "target_ids": tokens,
                    "assistant_prefix_tail": native_prefix[-3:],
                    "demo_ids": demo_ids,
                }
            )
    cases.sort(key=lambda r: r["key"])
    names = set(original["code_sha256"]) | {"contrastive_diagnostic.py", "query_intervention.py"}
    recipe = {
        "spec": spec,
        "config": config,
        "source_run": source.id,
        "cases": cases,
        "queries": queries,
        "bank_sha256": sha_json(bank),
        "eos_ids": eos,
        "code_sha256": {n: file_hash(Path(__file__).with_name(n)) for n in sorted(names)},
        "model_files_sha256": original["model_files_sha256"],
        "parent_recipe_sha256": sha_json(original),
        "step_1_sha256": sha_json(audit),
        "reference_files_sha256": audit["source_files_sha256"]
        | {
            str(parent / "recipe.json"): file_hash(parent / "recipe.json"),
            str(parent / "bank.json"): file_hash(parent / "bank.json"),
        },
        **{
            k: original[k] for k in ("input_bundle", "input_volume", "output_volume", "modal_image")
        },
    }
    _save(root / "recipe.json", recipe)
    _save(root / "bank.json", bank)
    _save(
        root / "preflight.json",
        {
            "recipe_sha256": sha_json(recipe),
            "cases": len(cases),
            "target_positions": sum(len(c["target_ids"]) for c in cases),
            "beyond_budget_positions": sum(max(0, len(c["target_ids"]) - 64) for c in cases),
            "normalization": "NFC; native assistant continuation; omit newline after EOS",
            "readiness_ids": [c["key"] for c in cases[:3]],
        },
    )
    print(
        json.dumps(
            {
                "prepared": 182,
                "positions": sum(len(c["target_ids"]) for c in cases),
                "recipe_sha256": sha_json(recipe),
                "step_1": audit["comparisons"],
            }
        )
    )


@app.function(
    image=image.add_local_python_source("active_ocr"),
    gpu="L4",
    cpu=2,
    memory=32768,
    timeout=3600,
    volumes=MOUNTS,
)
def score_cases(payload):
    import torch
    import transformers
    from safetensors.torch import save_file

    recipe, bank = payload["recipe"], payload["bank"]
    config, digest = recipe["config"], sha_json(recipe)
    remote = Path("/output") / ("icl-correctness-" + digest[:12])
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
    processor, model = _load_model(config["models"]["qwen"])
    model = model.to("cuda").eval()
    pages = {p["page_id"]: _load_page(p) for p in records}
    lines = {line_key(p, r): (p, r) for p in records for r in p["regions"]}
    settings = {
        k: config["inference"][k] for k in ("max_new_tokens", "do_sample", "repetition_penalty")
    }
    readiness_ids = {c["key"] for c in recipe["cases"][:3]}
    rows = []
    with torch.inference_mode():
        for case in payload["cases"]:
            identity = sha_json([digest, case["key"]])
            row = read_cache(remote / f"{identity}.json", identity)
            if row is None:
                started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                p, r = lines[case["key"]]
                demos = [(pages[lines[k][0]["page_id"]], lines[k][1]) for k in case["demo_ids"]]
                batches = [
                    _inputs(
                        processor,
                        pages[p["page_id"]],
                        r,
                        config,
                        examples=demos,
                        demonstration_mode=m,
                    )
                    for m in ("paired", "mismatched")
                ]
                for batch in batches:
                    assert batch["input_ids"][0, -3:].tolist() == case["assistant_prefix_tail"]
                    check_input_capacity(
                        batch, processor, model, 9, max(64, len(case["target_ids"]))
                    )
                metadata = [
                    {
                        "input_ids_sha256": sha_json(b["input_ids"].tolist()),
                        "grid": b["image_grid_thw"].tolist(),
                        "prompt_tokens": b["input_ids"].shape[1],
                    }
                    for b in batches
                ]
                batches = [_cuda(b) for b in batches]
                target = case["target_ids"]
                audit_steps = (
                    {0, len(target) // 2, len(target) - 1}
                    if case["key"] in readiness_ids
                    else set()
                )
                tokens, tensors, checks = [], {}, []
                streams = [forced_states(model, b, target, settings) for b in batches]
                for step, (a, b) in enumerate(zip(*streams, strict=True)):
                    for state, meta in zip((a, b), metadata, strict=True):
                        assert state["ids"][0, meta["prompt_tokens"] :].tolist() == target[:step]
                        assert state["cache_tokens"] == meta["prompt_tokens"] + step
                    assert a["cache_tokens"] >= metadata[0]["prompt_tokens"]
                    readouts = token_readouts(
                        a["logits"],
                        b["logits"],
                        a["ids"],
                        a["processors"],
                        target[step],
                        recipe["eos_ids"],
                        step == len(target) - 1,
                    )
                    tokens.append(
                        {
                            "position": step,
                            "gold": target[step],
                            "terminal": step == len(target) - 1,
                            "within_budget": step < 64,
                            "readouts": readouts,
                        }
                    )
                    if step in audit_steps:
                        for label, state, batch in zip(
                            ("paired", "mismatched"), (a, b), batches, strict=True
                        ):
                            replay = prefix_state(model, batch, target[:step], settings)
                            assert torch.equal(state["logits"], replay["logits"])
                            assert replay["geometry"]["position_sha256"]
                            tensors[f"{step}.{label}.logits"] = state["logits"].cpu().contiguous()
                            tensors[f"{step}.{label}.ids"] = state["ids"].cpu().contiguous()
                        ordinary = a["processors"](a["ids"], a["logits"].clone())
                        assert torch.equal(
                            ordinary,
                            contrastive_scores(
                                a["logits"], b["logits"], a["ids"], a["processors"], alpha=0
                            ),
                        )
                        assert int(ordinary.argmax()) == int(
                            contrastive_scores(
                                a["logits"], a["logits"], a["ids"], a["processors"]
                            ).argmax()
                        )
                        checks.append(
                            {
                                "position": step,
                                "native_replay": True,
                                "alpha_zero": True,
                                "identical_branches_choice": True,
                            }
                        )
                assert len(tokens) == len(target)
                vector = None
                if tensors:
                    path = remote / f"{identity}.safetensors"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    save_file(tensors, str(path))
                    vector = {"path": str(path.relative_to("/output")), "sha256": file_hash(path)}
                torch.cuda.synchronize()
                row = {
                    "key": case["key"],
                    "page_id": case["page_id"],
                    "tokens": tokens,
                    "metadata": metadata,
                    "readiness_checks": checks,
                    "vectors": vector,
                    "seconds": time.perf_counter() - started,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "app_id": payload["app_id"],
                    "call_id": modal.current_function_call_id(),
                }
                _save(
                    remote / f"{identity}.json",
                    {"identity": identity, "row": row, "row_sha256": sha_json(row)},
                )
                output_volume.commit()
            rows.append(row)
            print(json.dumps({"completed": len(rows), "key": case["key"]}), flush=True)
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


def summarize(rows):
    def aggregate(lines):
        text = [t for r in lines for t in r["tokens"] if not t["terminal"]]
        ends = [r["tokens"][-1] for r in lines]
        result = {}
        for mode in MODES:
            valid = [t["readouts"][mode] for t in text]
            finite = [v["nll"] for v in valid if v["nll"] is not None]
            correct = sum(v["correct"] for v in valid)
            result[mode] = {
                "text_positions": len(valid),
                "text_correct": correct,
                "text_top1": correct / len(valid),
                "excluded": sum(v["excluded"] for v in valid),
                "text_mean_nll": sum(finite) / len(valid) if len(finite) == len(valid) else None,
                "finite_nll_sum": sum(finite),
                "finite_nll_count": len(finite),
                "premature_eos_choices": sum(v["stops"] for v in valid),
                "end_correct": sum(t["readouts"][mode]["correct"] for t in ends),
                "end_mean_eos_mass": sum(t["readouts"][mode]["eos_mass"] for t in ends) / len(ends),
            }
        changes = Counter(
            (t["readouts"]["paired"]["correct"], t["readouts"]["contrastive"]["correct"])
            for t in text
        )
        result["changes"] = {
            "wrong_to_correct": changes[False, True],
            "correct_to_wrong": changes[True, False],
        }
        return result

    return {
        "pooled": aggregate(rows),
        "per_page": {
            p: aggregate([r for r in rows if r["page_id"] == p])
            for p in sorted({r["page_id"] for r in rows})
        },
        "line_averaged_top1": {
            m: sum(
                sum(t["readouts"][m]["correct"] for t in r["tokens"][:-1]) / (len(r["tokens"]) - 1)
                for r in rows
            )
            / len(rows)
            for m in MODES
        },
        "beyond_budget_positions": sum(not t["within_budget"] for r in rows for t in r["tokens"]),
        "lines": len(rows),
        "positions": sum(len(r["tokens"]) for r in rows),
    }


def report(recipe, root):
    rows = []
    for path in sorted(root.glob("part-*.json")):
        saved = json.loads(path.read_text())
        part = read_cache(path, saved["identity"])
        assert part["recipe_sha256"] == sha_json(recipe) and "L4" in part["gpu"]
        rows.extend(part["rows"])
    assert [r["key"] for r in rows] == [c["key"] for c in recipe["cases"]]
    assert len(set(r["key"] for r in rows)) == 182
    for row, case in zip(rows, recipe["cases"], strict=True):
        assert [t["gold"] for t in row["tokens"]] == case["target_ids"]
        assert [t["position"] for t in row["tokens"]] == list(range(len(case["target_ids"])))
    summary = {
        "recipe_sha256": sha_json(recipe),
        **summarize(rows),
        "query_work_seconds": sum(r["seconds"] for r in rows),
        "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in rows),
        "note": "Reference-prefix token diagnostic; not free-running CER or visual attribution",
    }
    _save(root / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_page"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "report"))
    args = parser.parse_args()
    spec = json.loads(SPEC.read_text())
    root = Path(spec["output_directory"])
    if args.stage == "prepare":
        prepare(spec)
        return
    recipe = json.loads((root / "recipe.json").read_text())
    assert recipe["spec"] == spec
    for path, expected in recipe["reference_files_sha256"].items():
        assert file_hash(path) == expected
    if args.stage == "report":
        report(recipe, root)
        return
    for name, expected in recipe["code_sha256"].items():
        assert file_hash(Path(__file__).with_name(name)) == expected
    bank = json.loads((root / "bank.json").read_text())
    with modal.enable_output(), app.run(environment_name="main"):
        for i, cases in enumerate((recipe["cases"][:3], recipe["cases"][3:])):
            if i:
                gate = json.loads((root / "readiness.json").read_text())
                assert gate == {"recipe_sha256": sha_json(recipe), "queries": 3, "passed": True}
            path = root / f"part-{i}.json"
            identity = sha_json([sha_json(recipe), [c["key"] for c in cases]])
            result = read_cache(path, identity)
            if result is None:
                call = score_cases.spawn(
                    {"recipe": recipe, "bank": bank, "cases": cases, "app_id": app.app_id}
                )
                with (root / "calls.jsonl").open("a") as f:
                    f.write(
                        json.dumps(
                            {
                                "app_id": app.app_id,
                                "call_id": call.object_id,
                                "queries": len(cases),
                                "started_utc": datetime.now(UTC).isoformat(),
                            }
                        )
                        + "\n"
                    )
                result = call.get()
                _save(path, {"identity": identity, "row": result, "row_sha256": sha_json(result)})
            assert result["recipe_sha256"] == sha_json(recipe) and len(result["rows"]) == len(cases)
            for row in result["rows"]:
                if row["vectors"]:
                    v = row["vectors"]
                    dest = root / Path(v["path"]).name
                    if not dest.exists():
                        dest.write_bytes(b"".join(output_volume.read_file(v["path"])))
                    assert file_hash(dest) == v["sha256"]
            if not i:
                assert all(len(r["readiness_checks"]) == 3 for r in result["rows"])
                _save(
                    root / "readiness.json",
                    {"recipe_sha256": sha_json(recipe), "queries": 3, "passed": True},
                )
    report(recipe, root)


if __name__ == "__main__":
    main()
