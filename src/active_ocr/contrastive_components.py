"""EXP-019: fixed stopping/text interventions in freely generated contrastive OCR."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import modal

from active_ocr.contrastive_icl import (
    _capture,
    _source,
    check_reference,
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
    _save,
    app,
)

SPEC = Path("experiments/recipes/read2016-known-box-components.json")
POLICIES = {
    "paired": ("paired", "paired"),
    "contrastive": ("contrastive", "contrastive"),
    "contrast_stopping": ("contrastive", "paired"),
    "contrast_text": ("paired", "contrastive"),
}
ARMS = ("contrast_stopping", "contrast_text")


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def component_choice(paired, contrastive, eos, policy):
    """Swap hard decisions, not logits/probabilities; use one positive candidate set."""
    import torch

    choices = {"paired": int(paired.argmax()), "contrastive": int(contrastive.argmax())}
    text = {}
    for name, scores in (("paired", paired), ("contrastive", contrastive)):
        values = scores.masked_fill(~torch.isfinite(contrastive), -torch.inf).clone()
        values[:, eos] = -torch.inf
        text[name] = int(values.argmax()) if torch.isfinite(values).any() else None
    gate, words = POLICIES[policy]
    stopped = choices[gate] in eos
    chosen = choices[gate] if stopped else text[words]
    assert chosen is not None
    assert torch.isfinite(contrastive[0, chosen])
    return chosen, {
        "paired_choice": choices["paired"],
        "contrastive_choice": choices["contrastive"],
        "paired_text": text["paired"],
        "contrastive_text": text["contrastive"],
        "paired_stops": choices["paired"] in eos,
        "contrastive_stops": choices["contrastive"] in eos,
        "selected": chosen,
        "gate": gate,
        "text": words,
    }


def component_generate(model, positive, negative, settings, policy, audit=False):
    """One generated suffix in two independent native caches; original generator stays intact."""
    import torch

    states = [
        model.generate(**b, **settings, custom_generate=_capture) for b in (positive, negative)
    ]
    lengths = [s["ids"].shape[1] for s in states]
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    tokens, trace, tensors = [], [], {}

    def capture(step, logits, histories):
        for role, values, ids in zip(("paired", "mismatched"), logits, histories, strict=True):
            tensors[f"{step}.{role}.logits"] = values.cpu().contiguous()
            tensors[f"{step}.{role}.ids"] = ids.cpu().contiguous()

    while True:
        logits, caches = [], []
        for state in states:
            if not tokens:
                out = model._prefill(state["ids"], state["config"], state["kwargs"])
            else:
                inputs = model.prepare_inputs_for_generation(
                    state["ids"], next_sequence_length=1, **state["kwargs"]
                )
                out = model(**inputs, return_dict=True)
            logits.append(out.logits[:, -1].to(copy=True, dtype=torch.float32))
            state["kwargs"] = model._update_model_kwargs_for_generation(out, state["kwargs"])
            caches.append(state["kwargs"]["past_key_values"].get_seq_length())
            del out
        assert states[0]["kwargs"]["past_key_values"] is not states[1]["kwargs"]["past_key_values"]
        for i, s in enumerate(states):
            assert caches[i] == lengths[i] + len(tokens)
            assert s["ids"][0, lengths[i] :].tolist() == tokens
        histories = [s["ids"] for s in states]
        ordinary = states[0]["processors"](histories[0], logits[0].clone())
        contrast = contrastive_scores(*logits, histories[0], states[0]["processors"])
        chosen, decision = component_choice(ordinary, contrast, eos, policy)
        if audit and len(tokens) in (0, 4):
            capture(len(tokens), logits, histories)
        trace.append(
            decision
            | {
                "position": len(tokens),
                "positive_cache_tokens": caches[0],
                "negative_cache_tokens": caches[1],
            }
        )
        tokens.append(chosen)
        for state in states:
            state["ids"] = torch.cat((state["ids"], state["ids"].new_tensor([[chosen]])), -1)
        if states[0]["stopping"](states[0]["ids"], None).all():
            if audit:
                capture(len(tokens) - 1, logits, histories)
            return tokens, trace, tensors


def prepare(spec):
    parent = Path(spec["parent_run"])
    original = json.loads((parent / "recipe.json").read_text())
    config, queries = original["config"], original["queries"]
    source, pages, truth, _ = _source(config)
    bank = json.loads((parent / "bank.json").read_text())
    assert sha_json(bank) == original["bank_sha256"]
    _check_records(bank, labelled=True)
    _check_records(queries, labelled=False)
    assert len(bank) == 24 and sum(len(p["regions"]) for p in bank) == 589
    assert sum(len(p["regions"]) for p in queries) == 182 and len(queries) == 8
    assert all(pages[p["page_id"]].split.value == "train" for p in bank)
    assert all(pages[p["page_id"]].split.value == "validation" for p in queries)
    assert not {p["image_sha256"] for p in bank} & {p["image_sha256"] for p in queries}
    keys = [line_key(p, r) for p in bank for r in p["regions"]]
    contexts = {
        line_key(p, r): sample_context(keys, line_key(p, r), 8, 824, sha_json(bank))
        for p in queries
        for r in p["regions"]
    }
    targets = {line_key(p, r): r["target"] for p in bank for r in p["regions"]}
    assert all(
        targets[a] != targets[b]
        for ids in contexts.values()
        for a, b in zip(ids, ids[1:] + ids[:1], strict=True)
    )
    cases = sorted(contexts)
    refs, indices = {}, {}
    for mode, name in (("zero", "zero"), ("paired", "paired-eight")):
        ref = original["references"][name]
        assert file_sha(ref["path"]) == ref["file_sha256"]
        data = json.loads(Path(ref["path"]).read_text())[ref["item"]]
        assert sha_json(data["receipt"]) == ref["receipt_sha256"]
        assert raw_metrics(data["receipt"], truth) == ref["raw_line"]
        indices[mode] = check_reference(
            data["receipt"], queries, config, source.id, contexts if mode == "paired" else None
        )
        refs[mode] = ref
    path = parent / "pilot/results.json"
    saved = json.loads(path.read_text())
    assert file_sha(path) == spec["contrast_results_sha256"]
    indices["contrastive"] = check_reference(saved["receipt"], queries, config, source.id, contexts)
    assert raw_metrics(saved["receipt"], truth) == saved["raw_line"]
    refs["contrastive"] = {
        "path": str(path),
        "item": None,
        "file_sha256": file_sha(path),
        "receipt_sha256": sha_json(saved["receipt"]),
        "raw_line": saved["raw_line"],
        "reused": True,
    }
    for name, expected in original["model_files_sha256"].items():
        assert file_sha(Path(spec["model_assets"]) / name) == expected
    names = set(original["code_sha256"]) | {"contrastive_components.py"}
    recipe = {
        "spec": spec,
        "config": config,
        "source_run": source.id,
        "queries": queries,
        "cases": cases,
        "contexts": contexts,
        "bank_sha256": sha_json(bank),
        "references": refs,
        "endpoint_tokens": {
            k: {
                "paired": original["native_check_tokens"][k],
                "contrastive": indices["contrastive"][k]["output_token_ids"],
            }
            for k in sorted(original["native_check_tokens"])
        },
        "endpoint_source": "parent native paired tokens; saved contrast pilot tokens",
        "parent_recipe_sha256": sha_json(original),
        "parent_recipe_file_sha256": file_sha(parent / "recipe.json"),
        "code_sha256": {n: file_sha(Path(__file__).with_name(n)) for n in sorted(names)},
        "model_files_sha256": original["model_files_sha256"],
        **{
            k: original[k] for k in ("input_bundle", "input_volume", "output_volume", "modal_image")
        },
    }
    root = Path(spec["output_directory"])
    _save(root / "recipe.json", recipe)
    _save(root / "bank.json", bank)
    _save(
        root / "preflight.json",
        {
            "recipe_sha256": sha_json(recipe),
            "lines": 182,
            "arms": list(ARMS),
            "new_outputs": 364,
            "endpoint_replays": 6,
            "references": {m: r["raw_line"] for m, r in refs.items()},
        },
    )
    print(json.dumps({"prepared": 364, "recipe_sha256": sha_json(recipe)}), flush=True)


@app.function(
    image=image.add_local_python_source("active_ocr"),
    gpu="L4",
    cpu=2,
    memory=32768,
    timeout=3600,
    volumes=MOUNTS,
)
def run_cases(payload):
    import torch
    import transformers
    from safetensors.torch import save_file

    recipe, bank = payload["recipe"], payload["bank"]
    digest, config = sha_json(recipe), recipe["config"]
    remote = Path("/output") / ("exp019-components-" + digest[:12])
    output_volume.reload()
    assert sha_json(bank) == recipe["bank_sha256"]
    _check_records(bank, labelled=True)
    _check_records(recipe["queries"], labelled=False)
    for name, expected in recipe["code_sha256"].items():
        assert file_sha(Path(__file__).with_name(name)) == expected
    for name, expected in recipe["model_files_sha256"].items():
        assert file_sha(Path("/input/model") / name) == expected
    records = bank + recipe["queries"]
    for p in records:
        assert file_sha(Path("/input/images") / p["image_sha256"]) == p["image_sha256"]
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
    with torch.inference_mode():
        for mode, key in payload["tasks"]:
            identity = sha_json([digest, mode, key])
            row = read_cache(remote / f"{identity}.json", identity)
            if row is None:
                start = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                p, r = lines[key]
                ids = recipe["contexts"][key]
                examples = [(pages[lines[k][0]["page_id"]], lines[k][1]) for k in ids]
                batches = [
                    _inputs(
                        processor,
                        pages[p["page_id"]],
                        r,
                        config,
                        examples=examples,
                        demonstration_mode=m,
                    )
                    for m in ("paired", "mismatched")
                ]
                metadata = []
                for batch in batches:
                    check_input_capacity(batch, processor, model, 9, 64)
                    metadata.append(
                        {
                            "prompt_tokens": batch["input_ids"].shape[1],
                            "ids_sha256": sha_json(batch["input_ids"].tolist()),
                            "grid": batch["image_grid_thw"].tolist(),
                        }
                    )
                batches = [_cuda(b) for b in batches]
                checks = {}
                readiness = key in recipe["endpoint_tokens"]
                if readiness and mode == ARMS[0]:
                    for endpoint in ("paired", "contrastive"):
                        got, _, _ = component_generate(model, *batches, settings, endpoint)
                        assert got == recipe["endpoint_tokens"][key][endpoint], (
                            f"endpoint replay {endpoint}: {key}"
                        )
                        checks[endpoint] = True
                tokens, trace, tensors = component_generate(
                    model, *batches, settings, mode, audit=readiness
                )
                vector = None
                if tensors:
                    path = remote / f"{identity}.safetensors"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    save_file(tensors, str(path))
                    vector = {"path": str(path.relative_to("/output")), "sha256": file_sha(path)}
                ended = bool(tokens and tokens[-1] in eos)
                assert ended or len(tokens) == 64
                torch.cuda.synchronize()
                row = {
                    "key": key,
                    "page_id": p["page_id"],
                    "id": r["id"],
                    "policy": mode,
                    "demo_ids": ids,
                    "negative_demo_image_ids": ids[1:] + ids[:1],
                    "metadata": metadata,
                    "text": processor.decode(tokens, skip_special_tokens=True)
                    .strip()
                    .replace("\n", " "),
                    "output_token_ids": tokens,
                    "tokens": len(tokens),
                    "truncated": len(tokens) >= 64,
                    "ended_with_eos": ended,
                    "stop_reason": "eos" if ended else "length",
                    "trace": trace,
                    "endpoint_checks": checks,
                    "vectors": vector,
                    "seconds": time.perf_counter() - start,
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
            print(json.dumps({"completed": len(rows), "policy": mode, "key": key}), flush=True)
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
    receipts, metrics, indices = {}, {}, {}
    for mode, ref in recipe["references"].items():
        assert file_sha(ref["path"]) == ref["file_sha256"]
        x = json.loads(Path(ref["path"]).read_text())
        x = x[ref["item"]] if ref["item"] is not None else x
        assert sha_json(x["receipt"]) == ref["receipt_sha256"]
        receipts[mode] = x["receipt"]
        metrics[mode] = raw_metrics(x["receipt"], truth)
        assert metrics[mode] == ref["raw_line"]
        indices[mode] = check_reference(
            x["receipt"],
            recipe["queries"],
            recipe["config"],
            recipe["source_run"],
            recipe["contexts"] if mode != "zero" else None,
        )
    rows = []
    for path in sorted(root.glob("part-*.json")):
        data = json.loads(path.read_text())
        part = read_cache(path, data["identity"])
        assert part["recipe_sha256"] == sha_json(recipe) and "L4" in part["gpu"]
        rows.extend(part["rows"])
    assert len(rows) == len({(r["policy"], r["key"]) for r in rows}) == 364
    details = []
    for mode in ARMS:
        selected = sorted((r for r in rows if r["policy"] == mode), key=lambda r: r["key"])
        assert [r["key"] for r in selected] == recipe["cases"]
        receipt = _receipt(selected, recipe["queries"], recipe["config"], recipe["source_run"])
        receipts[mode] = receipt
        metrics[mode] = raw_metrics(receipt, truth)
        indices[mode] = check_reference(
            receipt, recipe["queries"], recipe["config"], recipe["source_run"], recipe["contexts"]
        )
        _save(
            root / f"{mode}.json",
            {
                "receipt": receipt,
                "raw_line": metrics[mode],
                "recipe_sha256": sha_json(recipe),
                "reused": False,
            },
        )
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
    statistics = {}
    for mode in metrics:
        values = [d["predictions"][mode] for d in details]
        counts = {
            "metrics": metrics[mode],
            "chars": sum(v["chars"] for v in values),
            "exact": sum(v["exact"] for v in values),
            "operations": {
                k: sum(v["operations"][k] for v in values) for k in ("replace", "delete", "insert")
            },
        }
        if mode in ARMS:
            own = [r for r in rows if r["policy"] == mode]
            counts.update(
                caps=sum(r["stop_reason"] == "length" for r in own),
                empty=sum(not r["text"] for r in own),
                gate_disagreements=sum(
                    t["paired_stops"] != t["contrastive_stops"] for r in own for t in r["trace"]
                ),
                text_disagreements=sum(
                    t["paired_text"] != t["contrastive_text"] for r in own for t in r["trace"]
                ),
            )
        comparisons = {}
        for baseline in ("paired", "contrastive"):
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
        statistics[mode] = counts | {"comparisons": comparisons}
    summary = {
        "recipe_sha256": sha_json(recipe),
        "statistics": statistics,
        "lines": 182,
        "new_outputs": 364,
        "query_work_seconds": sum(r["seconds"] for r in rows),
        "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in rows),
        "rows": details,
    }
    _save(root / "summary.json", summary)
    print(
        json.dumps(
            {
                m: {
                    "cer": v["metrics"]["cer"],
                    "wer": v["metrics"]["wer"],
                    "char_edits": v["metrics"]["char_edits"],
                    "exact": v["exact"],
                }
                for m, v in statistics.items()
            },
            indent=2,
        ),
        flush=True,
    )


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
    if args.stage == "report":
        report(recipe, root)
        return
    for name, expected in recipe["code_sha256"].items():
        assert file_sha(Path(__file__).with_name(name)) == expected
    bank = json.loads((root / "bank.json").read_text())
    tasks = [(m, k) for m in ARMS for k in recipe["cases"]]
    gate = [t for t in tasks if t[1] in recipe["endpoint_tokens"]]
    rest = [t for t in tasks if t[1] not in recipe["endpoint_tokens"]]
    with modal.enable_output(), app.run(environment_name="main"):
        for i, work in enumerate((gate, rest)):
            if i:
                assert json.loads((root / "readiness.json").read_text())[
                    "recipe_sha256"
                ] == sha_json(recipe)
            path = root / f"part-{i}.json"
            identity = sha_json([sha_json(recipe), work])
            result = read_cache(path, identity)
            if result is None:
                call = run_cases.spawn(
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
            for r in result["rows"]:
                if r["vectors"]:
                    v = r["vectors"]
                    dest = root / Path(v["path"]).name
                    if not dest.exists():
                        dest.write_bytes(b"".join(output_volume.read_file(v["path"])))
                    assert file_sha(dest) == v["sha256"]
            if not i:
                checked = [r for r in result["rows"] if r["policy"] == ARMS[0]]
                assert len(checked) == 3 and all(
                    r["endpoint_checks"] == {"paired": True, "contrastive": True} for r in checked
                )
                _save(
                    root / "readiness.json",
                    {
                        "recipe_sha256": sha_json(recipe),
                        "passed": True,
                        "new_outputs": 6,
                        "endpoint_replays": 6,
                    },
                )
    report(recipe, root)


if __name__ == "__main__":
    main()
