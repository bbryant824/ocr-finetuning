"""Observed-prefix query-image intervention; EXP-018 follow-up, without fitting or decoding."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

import modal

from active_ocr.contrastive_icl import _capture, _source, check_reference, contrastive_scores
from active_ocr.icl import check_input_capacity, line_key, read_cache, sample_context
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

SPEC = Path("experiments/recipes/read2016-known-box-query-intervention.json")
MODES = ("paired", "mismatched", "text-only")


def first_difference(contrastive, text_only, eos):
    for i, (a, b) in enumerate(zip(contrastive, text_only, strict=False)):
        if a != b:
            return {
                "prefix_token_ids": contrastive[:i],
                "contrastive_token_id": a,
                "text_only_token_id": b,
                "answer_position": i,
                "eos_difference": a in eos or b in eos,
            }
    raise ValueError("observed token sequences do not differ at a shared decision")


def select_cases(candidates, per_group):
    selected = []
    for group in ("improved", "harmed"):
        counts = Counter()
        for stratum in ("later", "first"):
            eligible = sorted(
                [c for c in candidates if c["group"] == group and c["stratum"] == stratum],
                key=lambda c: sha_json([824, "query-intervention-case", c["key"]]),
            )
            # Reserve an available later EOS case before filling the frozen hash order.
            reserved = next((c for c in eligible if c["eos_difference"]), None)
            ordered = ([reserved] if reserved else []) + [c for c in eligible if c != reserved]
            chosen = []
            for case in ordered:
                if counts[case["page_id"]] < 2:
                    chosen.append(case)
                    counts[case["page_id"]] += 1
                if len(chosen) == per_group // 2:
                    break
            if len(chosen) != per_group // 2:
                raise ValueError("insufficient balanced matched-grid cases")
            selected.extend(chosen)
    return selected


def prefix_state(model, batch, prefix, settings):
    """Replay observed tokens through the native cache; retain only the final readout."""
    import torch

    state = model.generate(**batch, **settings, custom_generate=_capture)
    language = getattr(model.model, "language_model", model.model)
    positions, hidden = [], {}

    def capture_positions(module, args, kwargs):
        ids = kwargs.get("position_ids")
        positions.append(None if ids is None else sha_json(ids.tolist()))

    handles = [
        language.register_forward_pre_hook(capture_positions, with_kwargs=True),
        language.norm.register_forward_hook(
            lambda module, args, value: hidden.update(last=value[:, -1:].detach().clone())
        ),
    ]
    try:
        for step in range(len(prefix) + 1):
            if step == 0:
                out = model._prefill(state["ids"], state["config"], state["kwargs"])
            else:
                inputs = model.prepare_inputs_for_generation(
                    state["ids"], next_sequence_length=1, **state["kwargs"]
                )
                out = model(**inputs, return_dict=True)
            state["kwargs"] = model._update_model_kwargs_for_generation(out, state["kwargs"])
            if step < len(prefix):
                token = state["ids"].new_tensor([[prefix[step]]])
                state["ids"] = torch.cat((state["ids"], token), dim=-1)
        logits = out.logits[:, -1].to(copy=True, dtype=torch.float32)
        readout = model.lm_head(hidden["last"])[:, -1].float()
        error = float((readout - logits).abs().max())
        if not torch.allclose(readout, logits, atol=0.001, rtol=0):
            raise ValueError("final readout differs from native logits")
        cache = state["kwargs"]["past_key_values"].get_seq_length()
        if cache != state["ids"].shape[1] or not torch.isfinite(logits).all():
            raise ValueError("native prefix cache or finite-logit check failed")
        return {
            "logits": logits,
            "ids": state["ids"],
            "processors": state["processors"],
            "geometry": {
                "cache_tokens": cache,
                "position_sha256": positions,
                "final_max_abs_error": error,
                "forward_calls": len(prefix) + 1,
            },
        }
    finally:
        for handle in handles:
            handle.remove()


def decision(logits, ids, processors, contrastive_token, text_token):
    import torch

    raw = logits | {"contrastive": 2 * logits["paired"] - logits["mismatched"]}
    processed = {m: processors[m](ids[m], logits[m].clone()) for m in MODES}
    processed["contrastive"] = contrastive_scores(
        logits["paired"], logits["mismatched"], ids["paired"], processors["paired"]
    )
    result = {}
    for mode, scores in raw.items():
        candidates = []
        for token in (contrastive_token, text_token):
            value = processed[mode][0, token]
            candidates.append(
                {
                    "token_id": token,
                    "raw_logit": float(scores[0, token]),
                    "raw_logprob": float(scores.double().log_softmax(-1)[0, token]),
                    "rank": 1 + int((scores[0] > scores[0, token]).sum()),
                    "admissible": bool(torch.isfinite(value)),
                    "processed_score": float(value) if torch.isfinite(value) else None,
                    "processed_logprob": float(processed[mode].double().log_softmax(-1)[0, token])
                    if torch.isfinite(value)
                    else None,
                }
            )
        result[mode] = {
            "margin": candidates[0]["raw_logit"] - candidates[1]["raw_logit"],
            "rank_advantage": candidates[1]["rank"] - candidates[0]["rank"],
            "top_token_id": int(processed[mode].argmax()),
            "candidates": candidates,
        }
    return result


def same_readout(a, b):
    """Allow CPU/GPU log-softmax rounding; ranks, choices and structure remain exact."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_readout(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same_readout(x, y) for x, y in zip(a, b, strict=True))
    return math.isclose(a, b, abs_tol=2e-5, rel_tol=0) if isinstance(a, float) else a == b


def query_effects(original, donor):
    effects = {m: original[m]["margin"] - donor[m]["margin"] for m in original}
    rank = {m: original[m]["rank_advantage"] - donor[m]["rank_advantage"] for m in original}
    return {
        "D": effects,
        "rank_D": rank,
        "G": effects["contrastive"] - effects["text-only"],
        "rank_G": rank["contrastive"] - rank["text-only"],
        "paired_minus_text_D": effects["paired"] - effects["text-only"],
        "paired_minus_text_rank_D": rank["paired"] - rank["text-only"],
        "both_admissible": all(
            c["admissible"] for view in (original, donor) for c in view["contrastive"]["candidates"]
        ),
    }


def screen(rows):
    strata = {}
    for stratum in ("first", "later"):
        groups = {
            g: [r for r in rows if r["group"] == g and r["stratum"] == stratum]
            for g in ("improved", "harmed")
        }
        result = {}
        for name, subset in (
            ("all", groups),
            (
                "non_eos_both_admissible",
                {
                    g: [r for r in subset if not r["eos_difference"] and r["both_admissible"]]
                    for g, subset in groups.items()
                },
            ),
        ):
            if min(map(len, subset.values())) < 2:
                result[name] = {"passed": False, "reason": "fewer than two cases per group"}
                continue
            medians = {
                g: {k: median(r[k] for r in values) for k in ("G", "rank_G")}
                for g, values in subset.items()
            }
            passed = all(
                medians["improved"][k] > max(0, medians["harmed"][k]) for k in ("G", "rank_G")
            )
            if name == "all":
                passed &= all(
                    sum(r[k] > 0 for r in subset["improved"])
                    >= math.ceil(0.75 * len(subset["improved"]))
                    for k in ("G", "rank_G")
                )
            result[name] = {
                "passed": bool(passed),
                "medians": medians,
                "counts": {g: len(v) for g, v in subset.items()},
            }
        strata[stratum] = result
    return {
        "passed": all(v["passed"] for s in strata.values() for v in s.values()),
        "strata": strata,
    }


def prepare(spec, scope):
    import torch
    from PIL import Image
    from rapidfuzz.distance import Levenshtein
    from transformers import AutoProcessor

    from active_ocr.evaluation import page_text_view

    torch.set_num_threads(2)
    root = Path(spec["output_directory"]) / scope
    if scope == "confirmation":
        gate = json.loads((root.parent / "pilot/summary.json").read_text())
        if not gate["advance"]:
            raise ValueError("pilot screening criteria did not authorize confirmation")
    parent = Path(spec["pilot_run"] if scope == "pilot" else spec["full_run"])
    original = json.loads((parent / "recipe.json").read_text())
    config = original["config"]
    assert original["spec"]["alpha"] == 1 and original["spec"]["beta"] == 0.1
    source, pages, truth, _ = _source(config)
    bank = json.loads((parent / "bank.json").read_text())
    assert sha_json(bank) == original["bank_sha256"]
    _check_records(bank, labelled=True)
    records = original["queries"]
    _check_records(records, labelled=False)
    assert not {p["image_sha256"] for p in bank} & {p["image_sha256"] for p in records}
    keys = [line_key(p, r) for p in bank for r in p["regions"]]
    contexts = {
        line_key(p, r): sample_context(keys, line_key(p, r), 8, 824, sha_json(bank))
        for p in records
        for r in p["regions"]
    }
    phase = "pilot" if scope == "pilot" else "full"
    contrast_path = parent / phase / "results.json"
    contrast = json.loads(contrast_path.read_text())
    cd = check_reference(contrast["receipt"], records, config, source.id, contexts)
    assert raw_metrics(contrast["receipt"], truth) == contrast["raw_line"]
    ref = original["references"]["transcripts-only-eight"]
    text_path = Path(ref["path"])
    assert hashlib.sha256(text_path.read_bytes()).hexdigest() == ref["file_sha256"]
    saved = json.loads(text_path.read_text())[ref["item"]]
    assert sha_json(saved["receipt"]) == ref["receipt_sha256"]
    tt = check_reference(saved["receipt"], records, config, source.id, contexts)
    assert raw_metrics(saved["receipt"], truth) == ref["raw_line"]
    files = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (parent / "recipe.json", contrast_path, text_path, parent / "bank.json")
    }
    for name, expected in original["model_files_sha256"].items():
        with (Path(spec["model_assets"]) / name).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected
    processor = AutoProcessor.from_pretrained(spec["model_assets"], local_files_only=True)
    eos = json.loads((Path(spec["model_assets"]) / "generation_config.json").read_text())[
        "eos_token_id"
    ]
    eos = set(eos if isinstance(eos, list) else [eos])
    index, grids = {}, {}
    for record in records:
        with Image.open(pages[record["page_id"]].image_uri) as opened:
            page = opened.convert("RGB")
        for region in record["regions"]:
            key = line_key(record, region)
            index[key] = (record, region)
            grids[key] = _inputs(processor, page, region, config)["image_grid_thw"].tolist()
    candidates = []
    eligible = Counter()
    for key, (record, region) in index.items():
        target = page_text_view([truth[record["page_id"]][region["id"]]])
        outputs = [page_text_view([rows[key]["text"]]) for rows in (cd, tt)]
        edits = [Levenshtein.distance(target, output) for output in outputs]
        if edits[0] == edits[1]:
            continue
        group = "improved" if edits[0] < edits[1] else "harmed"
        difference = first_difference(cd[key]["output_token_ids"], tt[key]["output_token_ids"], eos)
        stratum = "first" if difference["answer_position"] == 0 else "later"
        eligible[group, stratum] += 1
        donors = [
            k
            for k in index
            if grids[k] == grids[key] and index[k][0]["page_id"] != record["page_id"]
        ]
        if not donors:
            continue
        donor = min(donors, key=lambda k: sha_json([824, "query-intervention-donor", key, k]))
        assert cd[key]["demo_ids"] == tt[key]["demo_ids"] == contexts[key]
        operations = [
            dict(Counter(op.tag for op in Levenshtein.editops(target, output)))
            for output in outputs
        ]
        candidates.append(
            {
                "key": key,
                "page_id": record["page_id"],
                "donor": donor,
                "grid": grids[key],
                "group": group,
                "stratum": stratum,
                **difference,
                "demo_ids": contexts[key],
                "char_edits": {"contrastive": edits[0], "text-only": edits[1]},
                "edit_operations": dict(zip(("contrastive", "text-only"), operations, strict=True)),
                "output_chars": dict(
                    zip(("contrastive", "text-only"), map(len, outputs), strict=True)
                ),
            }
        )
    per_group = 8 if scope == "pilot" else 32
    cases = select_cases(candidates, per_group)
    wanted = {c[k] for c in cases for k in ("key", "donor")}
    queries = [
        p | {"regions": [r for r in p["regions"] if line_key(p, r) in wanted]}
        for p in records
        if any(line_key(p, r) in wanted for r in p["regions"])
    ]
    names = set(original["code_sha256"]) | {"query_intervention.py"}
    recipe = {
        "spec": spec,
        "scope": scope,
        "config": config,
        "source_run": source.id,
        "bank_sha256": sha_json(bank),
        "queries": queries,
        "cases": cases,
        "reference_files_sha256": files,
        "parent_recipe_sha256": sha_json(original),
        "code_sha256": {
            n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
            for n in sorted(names)
        },
        "model_files_sha256": original["model_files_sha256"],
        "input_bundle": original["input_bundle"],
        "input_volume": original["input_volume"],
        "output_volume": original["output_volume"],
        "modal_image": original["modal_image"],
        "eligibility": {
            f"{g}/{s}": {
                "before_grid": eligible[g, s],
                "matched_grid": sum(c["group"] == g and c["stratum"] == s for c in candidates),
            }
            for g in ("improved", "harmed")
            for s in ("first", "later")
        },
        "selection_rule": (
            "seed824 frozen hash order; balanced outcome/position; "
            "max2/group/page; reserve later EOS"
        ),
    }
    path = root / "recipe.json"
    if path.exists() and json.loads(path.read_text()) != recipe:
        raise ValueError("existing frozen intervention recipe differs")
    _save(path, recipe)
    _save(root / "bank.json", bank)
    print(
        json.dumps(
            {
                "prepared": scope,
                "cases": len(cases),
                "prefix_evaluations": len(cases) * 6,
                "recipe_sha256": sha_json(recipe),
                "eligibility": recipe["eligibility"],
            }
        ),
        flush=True,
    )


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
    config = recipe["config"]
    digest = sha_json(recipe)
    remote = Path("/output") / ("icl-query-intervention-" + digest[:12])
    output_volume.reload()
    _check_records(bank, labelled=True)
    _check_records(recipe["queries"], labelled=False)
    assert sha_json(bank) == recipe["bank_sha256"]
    assert all(":train:" in p["page_id"] for p in bank)
    assert all(":validation:" in p["page_id"] for p in recipe["queries"])
    for name, expected in recipe["code_sha256"].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == expected
    for name, expected in recipe["model_files_sha256"].items():
        with (Path("/input/model") / name).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected
    records = bank + recipe["queries"]
    for p in records:
        assert (
            hashlib.sha256((Path("/input/images") / p["image_sha256"]).read_bytes()).hexdigest()
            == p["image_sha256"]
        )
    processor, model = _load_model(config["models"]["qwen"])
    model = model.to("cuda").eval()
    pages = {p["page_id"]: _load_page(p) for p in records}
    lines = {line_key(p, r): (p, r) for p in records for r in p["regions"]}
    settings = {
        k: config["inference"][k] for k in ("max_new_tokens", "do_sample", "repetition_penalty")
    }
    rows = []
    with torch.inference_mode():
        for case in payload["cases"]:
            identity = sha_json([digest, case["key"]])
            row = read_cache(remote / f"{identity}.json", identity)
            if row is None:
                start = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                views, tensors, metadata = {}, {}, {}
                for view, key in (("original", case["key"]), ("donor", case["donor"])):
                    p, r = lines[key]
                    examples = [
                        (pages[lines[k][0]["page_id"]], lines[k][1]) for k in case["demo_ids"]
                    ]
                    logits, histories, processors = {}, {}, {}
                    metadata[view] = {}
                    for mode in MODES:
                        batch = _inputs(
                            processor,
                            pages[p["page_id"]],
                            r,
                            config,
                            examples=examples,
                            demonstration_mode=mode,
                        )
                        check_input_capacity(
                            batch, processor, model, 1 if mode == "text-only" else 9, 64
                        )
                        pixels = batch["pixel_values"][-int(batch["image_grid_thw"][-1].prod()) :]
                        pixel_hash = hashlib.sha256(pixels.numpy().tobytes()).hexdigest()
                        grid = batch["image_grid_thw"].tolist()
                        nonpixel = sha_json(
                            {k: v.tolist() for k, v in batch.items() if k != "pixel_values"}
                        )
                        assert grid[-1:] == case["grid"]
                        batch = _cuda(batch)
                        state = prefix_state(model, batch, case["prefix_token_ids"], settings)
                        logits[mode], histories[mode], processors[mode] = (
                            state["logits"],
                            state["ids"],
                            state["processors"],
                        )
                        metadata[view][mode] = state["geometry"] | {
                            "grid": grid,
                            "query_pixel_sha256": pixel_hash,
                            "nonpixel_sha256": nonpixel,
                            "input_ids_sha256": sha_json(state["ids"].tolist()),
                            "processors": [
                                {"class": type(p).__name__, "penalty": getattr(p, "penalty", None)}
                                for p in state["processors"]
                            ],
                        }
                        tensors[f"{view}.{mode}.logits"] = state["logits"].cpu().contiguous()
                        tensors[f"{view}.{mode}.ids"] = state["ids"].cpu().contiguous()
                    views[view] = decision(
                        logits,
                        histories,
                        processors,
                        case["contrastive_token_id"],
                        case["text_only_token_id"],
                    )
                    del logits, histories, processors, state, batch
                for mode in MODES:
                    a, b = metadata["original"][mode], metadata["donor"][mode]
                    for k in (
                        "grid",
                        "nonpixel_sha256",
                        "input_ids_sha256",
                        "cache_tokens",
                        "position_sha256",
                        "processors",
                    ):
                        assert a[k] == b[k], f"query swap changed {k}"
                    assert a["query_pixel_sha256"] != b["query_pixel_sha256"]
                    assert a["position_sha256"] and all(a["position_sha256"])
                expected = {
                    "contrastive": case["contrastive_token_id"],
                    "text-only": case["text_only_token_id"],
                }
                checks = {
                    m: views["original"][m]["top_token_id"] == token
                    for m, token in expected.items()
                }
                vector_path = remote / f"{identity}.safetensors"
                vector_path.parent.mkdir(parents=True, exist_ok=True)
                save_file(tensors, str(vector_path))
                torch.cuda.synchronize()
                row = {
                    "key": case["key"],
                    "views": views,
                    "metadata": metadata,
                    "effects": query_effects(views["original"], views["donor"]),
                    "original_choice_checks": checks,
                    "vector_path": str(vector_path.relative_to("/output")),
                    "vectors_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
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
            if not all(row["original_choice_checks"].values()):
                raise ValueError(f"original next-token replay failed: {case['key']}")
            rows.append(row)
            print(json.dumps({"completed": len(rows), "case": case["key"]}), flush=True)
    return {
        "recipe_sha256": digest,
        "rows": rows,
        "gpu": torch.cuda.get_device_name(),
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "model_class": type(model).__name__,
            "processor_class": type(processor).__name__,
        },
    }


def report(recipe, root):
    from safetensors.torch import load_file
    from transformers.generation.logits_process import (
        LogitsProcessorList,
        RepetitionPenaltyLogitsProcessor,
    )

    rows, seen = [], set()
    for path in sorted(root.glob("part-*.json")):
        saved = json.loads(path.read_text())
        data = read_cache(path, saved["identity"])
        assert data["recipe_sha256"] == sha_json(recipe)
        for row in data["rows"]:
            assert row["key"] not in seen and all(row["original_choice_checks"].values())
            seen.add(row["key"])
            path = root / Path(row["vector_path"]).name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["vectors_sha256"]
            tensors = load_file(str(path))
            case = next(c for c in recipe["cases"] if c["key"] == row["key"])
            for view in ("original", "donor"):
                processors = {}
                for mode in MODES:
                    assert row["metadata"][view][mode]["processors"] == [
                        {"class": "RepetitionPenaltyLogitsProcessor", "penalty": 1.1}
                    ]
                    processors[mode] = LogitsProcessorList([RepetitionPenaltyLogitsProcessor(1.1)])
                checked = decision(
                    {m: tensors[f"{view}.{m}.logits"] for m in MODES},
                    {m: tensors[f"{view}.{m}.ids"] for m in MODES},
                    processors,
                    case["contrastive_token_id"],
                    case["text_only_token_id"],
                )
                assert same_readout(checked, row["views"][view])
            assert query_effects(row["views"]["original"], row["views"]["donor"]) == row["effects"]
            rows.append(case | row["effects"])
    assert seen == {c["key"] for c in recipe["cases"]}
    screening = screen(rows)
    page_checks = {
        page: screen([r for r in rows if r["page_id"] != page])["passed"]
        for page in sorted({r["page_id"] for r in rows})
    }
    summary = {
        "recipe_sha256": sha_json(recipe),
        "execution_module_sha256": recipe["code_sha256"]["query_intervention.py"],
        "analysis_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": len(rows),
        "prefix_evaluations": len(rows) * 6,
        "screen": screening,
        "leave_one_page_out": page_checks,
        "advance": screening["passed"] and all(page_checks.values()),
        "rows": rows,
        "note": (
            "Exploratory outcome-selected diagnostic; not a population benchmark or grounding proof"
        ),
    }
    _save(root / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "report"))
    parser.add_argument("--scope", choices=("pilot", "confirmation"), default="pilot")
    args = parser.parse_args()
    spec = json.loads(SPEC.read_text())
    if args.stage == "prepare":
        prepare(spec, args.scope)
        return
    root = Path(spec["output_directory"]) / args.scope
    recipe = json.loads((root / "recipe.json").read_text())
    assert recipe["spec"] == spec
    for path, expected in recipe["reference_files_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    if args.stage == "report":
        # Offline reanalysis records its source separately; frozen tensor receipts stay immutable.
        report(recipe, root)
        return
    for name, expected in recipe["code_sha256"].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == expected
    bank = json.loads((root / "bank.json").read_text())
    assert sha_json(bank) == recipe["bank_sha256"]
    # The first production case is the GPU readiness check; then continue the same fixed sample.
    with modal.enable_output(), app.run(environment_name="main"):
        for index, cases in enumerate((recipe["cases"][:1], recipe["cases"][1:])):
            path = root / f"part-{index}.json"
            identity = sha_json([sha_json(recipe), [c["key"] for c in cases]])
            result = read_cache(path, identity)
            if result is None:
                call = run_cases.spawn(
                    {"recipe": recipe, "bank": bank, "cases": cases, "app_id": app.app_id}
                )
                event = {
                    "app_id": app.app_id,
                    "call_id": call.object_id,
                    "cases": len(cases),
                    "scope": args.scope,
                    "started_utc": datetime.now(UTC).isoformat(),
                }
                with (root / "calls.jsonl").open("a") as stream:
                    stream.write(json.dumps(event) + "\n")
                print(json.dumps(event), flush=True)
                result = call.get()
                _save(path, {"identity": identity, "row": result, "row_sha256": sha_json(result)})
            assert result["recipe_sha256"] == sha_json(recipe) and "L4" in result["gpu"]
            for row in result["rows"]:
                destination = root / Path(row["vector_path"]).name
                if not destination.exists():
                    destination.write_bytes(b"".join(output_volume.read_file(row["vector_path"])))
                assert hashlib.sha256(destination.read_bytes()).hexdigest() == row["vectors_sha256"]
    report(recipe, root)


if __name__ == "__main__":
    main()
