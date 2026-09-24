"""Qwen READ2016 active-learning selections on the frozen page-text protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import modal

from active_ocr.recognition_pipeline import (
    BUNDLE,
    IMAGE_ID,
    INPUT_VOLUME_ID,
    OUTPUT_VOLUME_ID,
    _inputs,
    _load_model,
    app,
    image,
    input_volume,
    output_volume,
    run_stage,
    sha_json,
)

ANCHOR = Path(".local/verification/exp-013/attempt-1/qwen-page-text")


@app.function(
    image=image,
    gpu="L40S",
    cpu=2,
    memory=49152,
    timeout=7200,
    volumes={
        "/input": input_volume.with_mount_options(sub_path="/bundles/" + BUNDLE, read_only=True),
        "/output": output_volume,
    },
)
def score_pool(records, config, checkpoint, mode):
    """Return label-free scores or frozen visual features for TRAIN images."""
    import torch
    from peft import PeftModel

    if mode not in {"uncertainty", "visual"}:
        raise ValueError("unknown scoring mode")
    if not records or len({r["page_id"] for r in records}) != len(records):
        raise ValueError("empty or duplicate scoring pages")
    for record in records:
        if set(record) != {"page_id", "image_sha256"} or ":train:" not in record["page_id"]:
            raise ValueError("pool scorer accepts TRAIN image identity only")
        image_path = Path("/input/images") / record["image_sha256"]
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != record["image_sha256"]:
            raise ValueError("pool image checksum mismatch")
    spec = config["models"]["qwen"]
    processor, model = _load_model(spec)
    if mode == "uncertainty":
        receipt_path = (
            Path("/output") / checkpoint["run_name"] / (f"round-{checkpoint['stage']}/receipt.json")
        )
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt["source_run"],
            receipt["selected_ids"],
            receipt["adapter_sha256"],
            receipt["model_revision"],
        ) != (
            config["source_run_id"],
            checkpoint["selected_ids"],
            checkpoint["adapter_sha256"],
            spec["revision"],
        ):
            raise ValueError("score checkpoint differs from selected training pages")
        adapter = receipt_path.parent / "adapter"
        if (
            hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
            != checkpoint["adapter_sha256"]
        ):
            raise ValueError("score adapter checksum mismatch")
        model = PeftModel.from_pretrained(model, adapter)
    model = model.to("cuda").eval()
    output = []
    for record in records:
        batch = _inputs(processor, record, config, spec)
        batch = {
            k: v.to(device="cuda", dtype=torch.bfloat16) if v.is_floating_point() else v.to("cuda")
            for k, v in batch.items()
        }
        with torch.inference_mode():
            if mode == "visual":
                feature = (
                    model.get_image_features(batch["pixel_values"], batch["image_grid_thw"])
                    .pooler_output[0]
                    .float()
                    .mean(dim=0)
                )
                norm = feature.norm()
                if not torch.isfinite(norm) or norm <= 0:
                    raise ValueError("invalid visual feature")
                output.append(
                    {
                        "page_id": record["page_id"],
                        "embedding": (feature / norm).tolist(),
                    }
                )
                continue
            generation = model.generate(
                **batch,
                max_new_tokens=config["inference"]["max_new_tokens"],
                do_sample=False,
                repetition_penalty=config["inference"]["repetition_penalty"],
                return_dict_in_generate=True,
                output_logits=True,
            )
            emitted = generation.sequences[0, batch["input_ids"].shape[1] :]
            if len(emitted) == 0 or len(emitted) != len(generation.logits):
                raise ValueError("generated token/logit alignment failed")
            entropy_sum = 0.0
            log_probability_sum = 0.0
            for token, raw_logits in zip(emitted, generation.logits, strict=True):
                logp = torch.log_softmax(raw_logits[0].float(), dim=-1)
                probabilities = logp.exp()
                entropy_sum += float(-(probabilities * logp).sum())
                log_probability_sum += float(logp[token])
            vocab_size = generation.logits[0].shape[-1]
            entropy = entropy_sum / (len(emitted) * math.log(vocab_size))
            confidence = math.exp(log_probability_sum / len(emitted))
            if not math.isfinite(entropy) or not math.isfinite(confidence):
                raise ValueError("nonfinite acquisition score")
            if not 0 <= entropy <= 1 or not 0 <= confidence <= 1:
                raise ValueError("acquisition score outside [0, 1]")
            output.append(
                {
                    "page_id": record["page_id"],
                    "entropy": entropy,
                    "confidence": confidence,
                    "tokens": len(emitted),
                    "truncated": len(emitted) >= config["inference"]["max_new_tokens"],
                    "generated_ids_sha256": hashlib.sha256(
                        json.dumps(emitted.tolist()).encode()
                    ).hexdigest(),
                }
            )
        del batch
    return output


def kcenter_greedy(embeddings, labelled, candidates, count):
    """Select farthest TRAIN pages, updating coverage after every choice."""
    import numpy as np

    candidate_ids = sorted(set(candidates))
    labelled_ids = sorted(set(labelled))
    if not labelled_ids or count <= 0 or count > len(candidate_ids):
        raise ValueError("k-center needs labelled seeds and a feasible positive budget")
    if set(candidate_ids) & set(labelled_ids) or set(candidate_ids + labelled_ids) != set(
        embeddings
    ):
        raise ValueError("k-center embedding membership mismatch")
    keys = labelled_ids + candidate_ids
    vectors = np.asarray([embeddings[k] for k in keys], dtype=np.float32)
    if (
        vectors.ndim != 2
        or not np.isfinite(vectors).all()
        or np.any(np.linalg.norm(vectors, axis=1) == 0)
    ):
        raise ValueError("invalid k-center embeddings")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    seeds = vectors[: len(labelled_ids)]
    pool = vectors[len(labelled_ids) :]
    distances = np.min(np.linalg.norm(pool[:, None, :] - seeds[None, :, :], axis=-1), axis=1)
    chosen = []
    available = np.ones(len(candidate_ids), dtype=bool)
    for _ in range(count):
        index = int(np.argmax(np.where(available, distances, -1)))
        chosen.append(candidate_ids[index])
        available[index] = False
        distances = np.minimum(distances, np.linalg.norm(pool - pool[index], axis=1))
    return tuple(chosen)


def _checked_json(path, value):
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != value:
            raise ValueError(f"existing artifact differs: {path}")
        return saved
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    return value


def _metrics(receipt, pages, oracle, validation_ids, experiment_id, config):
    from active_ocr.integrations.simulation import PageTextEvaluatorV1
    from active_ocr.models import (
        Box,
        Prediction,
        PredictionPurpose,
        PredictionStatus,
        Region,
    )

    if [p["page_id"] for p in receipt["predictions"]] != validation_ids:
        raise ValueError("prediction coverage/order differs from validation images")
    predictions = []
    for raw in receipt["predictions"]:
        page = pages[raw["page_id"]]
        status = (
            PredictionStatus.TRUNCATED
            if raw["tokens"] >= config["inference"]["max_new_tokens"]
            else (PredictionStatus.OK if raw["text"] else PredictionStatus.INVALID_OUTPUT)
        )
        predictions.append(
            Prediction(
                page_id=raw["page_id"],
                experiment_id=experiment_id,
                round_number=receipt["stage"],
                model_id=(
                    "adapter:sha256:" + receipt["adapter_sha256"]
                    if receipt["stage"]
                    else "hf:" + receipt["model_repository"] + "@" + receipt["model_revision"]
                ),
                status=status,
                purpose=(
                    PredictionPurpose.VALIDATION
                    if receipt["stage"]
                    else PredictionPurpose.BASELINE_VALIDATION
                ),
                regions=(
                    (
                        Region(
                            id="page-text",
                            box=Box(x=0, y=0, width=page.width, height=page.height),
                            text=raw["text"],
                        ),
                    )
                    if status is PredictionStatus.OK
                    else ()
                ),
            )
        )
    return oracle.evaluate_validation(
        tuple(predictions), PageTextEvaluatorV1(), tuple(validation_ids)
    )


def main():
    from active_ocr.active_learning import select_pages
    from active_ocr.evaluation import page_text_view
    from active_ocr.integrations.simulation import LocalOracle
    from active_ocr.models import Prediction, Split, Strategy
    from active_ocr.pipeline import Pipeline

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_runs", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/recipes/read2016-page-text.json")
    )
    parser.add_argument("--anchor", type=Path, default=ANCHOR)
    parser.add_argument(
        "--strategy",
        choices=("entropy", "least_confidence", "kcenter_greedy"),
        required=True,
    )
    parser.add_argument("--through-stage", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--score-smoke", action="store_true")
    parser.add_argument("--holdout", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}", args.results.name):
        raise ValueError("results basename must be a safe Modal run name")
    if args.holdout and args.through_stage != 3:
        raise ValueError("holdout requires three completed rounds")
    config_path = args.config
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    visual_config = {
        **config,
        "selection": {k: v for k, v in config["selection"].items() if k != "seed"},
    }
    source = Pipeline.for_simulation(args.source_runs).get_simulation(config["source_run_id"])
    if source.dataset.manifest_sha256 != config["source_manifest_sha256"]:
        raise ValueError("source manifest differs from frozen recipe")
    oracle = LocalOracle(source.dataset)
    pages = {p.id: p for p in source.dataset.pages}
    train_ids = {p.id for p in pages.values() if p.split is Split.TRAIN}
    val8 = list(config["validation_page_ids"])
    val42 = sorted(p.id for p in pages.values() if p.split is Split.VALIDATION and p.id not in val8)
    if len(train_ids) != 350 or len(val8) != 8 or len(val42) != 42:
        raise ValueError("unexpected frozen READ membership")
    first64 = list(select_pages(train_ids, Strategy.RANDOM, 64, config["selection"]["seed"]))
    anchor_recipe = json.loads((args.anchor / "recipe.json").read_text())
    anchor = json.loads((args.anchor / "round-1.json").read_text())["receipt"]
    if (
        anchor_recipe["config"],
        anchor_recipe["script_sha256"],
        anchor["selected_ids"],
        anchor["recipe_sha256"],
        anchor["model_revision"],
    ) != (
        config,
        hashlib.sha256(Path("src/active_ocr/recognition_pipeline.py").read_bytes()).hexdigest(),
        first64,
        sha_json(anchor_recipe),
        config["models"]["qwen"]["revision"],
    ):
        raise ValueError("first-round Qwen anchor is not the frozen random control")
    if not anchor["adapter_sha256"]:
        raise ValueError("first-round Qwen adapter is missing")
    recipe = {
        "strategy": args.strategy,
        "source_run": source.id,
        "source_manifest_sha256": source.dataset.manifest_sha256,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "visual_config_sha256": sha_json(visual_config),
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "anchor_receipt_sha256": sha_json(anchor),
        "anchor_run_name": args.anchor.name,
        "anchor_adapter_sha256": anchor["adapter_sha256"],
        "image_id": IMAGE_ID,
        "input_volume": INPUT_VOLUME_ID,
        "output_volume": OUTPUT_VOLUME_ID,
        "bundle": BUNDLE,
        "validation_ids": val8,
        "holdout_ids": val42,
    }
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "anchor_pages": len(first64),
                    "anchor_sha256": sha_json(first64),
                    "source_pages": len(pages),
                }
            )
        )
        return
    args.results.mkdir(parents=True, exist_ok=True)
    _checked_json(args.results / "recipe.json", recipe)
    recipe_sha = sha_json(recipe)

    def collect_scores(checkpoint, mode, records, tag):
        identity = {
            "source_run": source.id,
            "mode": mode,
            "checkpoint": checkpoint,
            "records": records,
            "config_sha256": (
                recipe["visual_config_sha256"] if mode == "visual" else recipe["config_sha256"]
            ),
            "scorer_sha256": recipe["scorer_sha256"],
        }
        key = sha_json(identity)
        cache_root = args.results.parent / "_score-cache"
        cache_root.mkdir(exist_ok=True)
        cached = cache_root / f"{key}.json"
        if cached.exists():
            result = json.loads(cached.read_text())
            if result["identity"] != identity:
                raise ValueError("score cache identity mismatch")
            rows = result["rows"]
        else:
            rows = []
            with modal.enable_output(), app.run(environment_name="main"):
                for start in range(0, len(records), 16):
                    chunk = records[start : start + 16]
                    part = cache_root / f"{key}-part-{start // 16}.json"
                    if part.exists():
                        saved = json.loads(part.read_text())
                        if saved["identity"] != identity or saved["start"] != start:
                            raise ValueError("score chunk identity mismatch")
                        scored = saved["rows"]
                    else:
                        scored = score_pool.remote(chunk, config, checkpoint, mode)
                        _checked_json(part, {"identity": identity, "start": start, "rows": scored})
                    if [row["page_id"] for row in scored] != [r["page_id"] for r in chunk]:
                        raise ValueError("score chunk coverage mismatch")
                    rows.extend(scored)
            _checked_json(cached, {"identity": identity, "rows": rows})
        if [row["page_id"] for row in rows] != [r["page_id"] for r in records]:
            raise ValueError("score coverage differs from frozen TRAIN pool")
        _checked_json(
            args.results / f"scores-{tag}.json",
            {"cache_key": key, "identity": identity},
        )
        return rows

    if args.score_smoke:
        candidates = sorted(train_ids - set(first64))[:2]
        checkpoint = {
            "run_name": args.anchor.name,
            "stage": 1,
            "selected_ids": first64,
            "adapter_sha256": anchor["adapter_sha256"],
        }
        mode = "visual" if args.strategy == "kcenter_greedy" else "uncertainty"
        rows = collect_scores(
            checkpoint if mode == "uncertainty" else None,
            mode,
            [{"page_id": i, "image_sha256": pages[i].image_sha256} for i in candidates],
            "smoke",
        )
        print(
            json.dumps(
                {
                    "mode": mode,
                    "pages": len(rows),
                    "first": {k: v for k, v in rows[0].items() if k != "embedding"},
                }
            )
        )
        return

    selected = first64.copy()
    for stage in (2, 3):
        if stage > args.through_stage:
            break
        pool_ids = sorted(train_ids - set(selected))
        mode = "visual" if args.strategy == "kcenter_greedy" else "uncertainty"
        if mode == "visual":
            score_ids = sorted(train_ids)
            checkpoint = None
            tag = "visual-base"
        else:
            score_ids = pool_ids
            prior = (
                anchor
                if stage == 2
                else json.loads((args.results / f"round-{stage - 1}.json").read_text())["receipt"]
            )
            checkpoint = {
                "run_name": args.anchor.name if stage == 2 else args.results.name,
                "stage": stage - 1,
                "selected_ids": selected,
                "adapter_sha256": prior["adapter_sha256"],
            }
            tag = f"round-{stage - 1}"
        records = [{"page_id": i, "image_sha256": pages[i].image_sha256} for i in score_ids]
        rows = collect_scores(checkpoint, mode, records, tag)
        if mode == "visual":
            new_ids = kcenter_greedy(
                {r["page_id"]: r["embedding"] for r in rows}, selected, pool_ids, 64
            )
        else:
            predictions = [
                Prediction(
                    page_id=r["page_id"],
                    experiment_id=args.results.name,
                    round_number=stage - 1,
                    model_id="adapter:sha256:" + checkpoint["adapter_sha256"],
                    confidence=r["confidence"],
                    entropy=r["entropy"],
                )
                for r in rows
            ]
            new_ids = select_pages(
                pool_ids, Strategy(args.strategy), 64, 0, predictions=predictions
            )
        choice = {
            "strategy": args.strategy,
            "stage": stage,
            "pool_sha256": sha_json(pool_ids),
            "score_rows_sha256": sha_json(rows),
            "selected_ids": list(new_ids),
            "selected_sha256": sha_json(new_ids),
        }
        _checked_json(args.results / f"selection-round-{stage}.json", choice)
        selected.extend(new_ids)
        if (
            len(selected) != 64 * stage
            or len(set(selected)) != len(selected)
            or not set(selected) <= train_ids
        ):
            raise ValueError("selected IDs violate frozen TRAIN budget")
        target_rows = [
            {
                "page_id": e.page.id,
                "image_sha256": e.page.image_sha256,
                "target": page_text_view([r.text for r in e.regions]),
            }
            for e in oracle.reveal(tuple(selected))
        ]
        payload = {
            "source_run": source.id,
            "recipe_sha256": recipe_sha,
            "model": "qwen",
            "config": config,
            "selected_ids": selected,
            "train": target_rows,
            "validation": [{"page_id": i, "image_sha256": pages[i].image_sha256} for i in val8],
        }
        path = args.results / f"round-{stage}.json"
        if path.exists():
            result = json.loads(path.read_text())
            receipt = result["receipt"]
        else:
            with modal.enable_output(), app.run(environment_name="main"):
                receipt = run_stage.remote(payload, stage, args.results.name)
            result = {"receipt": receipt}
        if (
            receipt["source_run"],
            receipt["recipe_sha256"],
            receipt["selected_ids"],
            receipt["validation_ids"],
            receipt["model_revision"],
        ) != (
            source.id,
            recipe_sha,
            selected,
            val8,
            config["models"]["qwen"]["revision"],
        ):
            raise ValueError("fit receipt differs from selected pages or recipe")
        metrics = _metrics(receipt, pages, oracle, val8, args.results.name, config)
        result["metrics"] = metrics
        _checked_json(path, result)
        print(
            json.dumps(
                {
                    "stage": stage,
                    "labelled_pages": len(selected),
                    "cer": metrics["cer"],
                    "wer": metrics["wer"],
                    "failed_pages": metrics["failed_pages"],
                }
            )
        )
    if args.holdout:
        for stage in (2, 3):
            ids = first64.copy()
            for n in (2, 3):
                if n > stage:
                    break
                ids.extend(
                    json.loads((args.results / f"selection-round-{n}.json").read_text())[
                        "selected_ids"
                    ]
                )
            payload = {
                "source_run": source.id,
                "recipe_sha256": recipe_sha,
                "model": "qwen",
                "config": config,
                "selected_ids": ids,
                "train": [],
                "validation": [
                    {"page_id": i, "image_sha256": pages[i].image_sha256} for i in val42
                ],
            }
            path = args.results / f"round-{stage}-holdout.json"
            if path.exists():
                result = json.loads(path.read_text())
                receipt = result["receipt"]
            else:
                with modal.enable_output(), app.run(environment_name="main"):
                    receipt = run_stage.remote(payload, stage, args.results.name, True)
                result = {"receipt": receipt}
            if (
                receipt["selected_ids"],
                receipt["recipe_sha256"],
                receipt["validation_ids"],
            ) != (ids, recipe_sha, val42):
                raise ValueError("holdout receipt differs from selected pages")
            result["metrics"] = _metrics(receipt, pages, oracle, val42, args.results.name, config)
            _checked_json(path, result)
            print(
                json.dumps(
                    {
                        "stage": stage,
                        "holdout": True,
                        "cer": result["metrics"]["cer"],
                        "wer": result["metrics"]["wer"],
                    }
                )
            )


if __name__ == "__main__":
    main()
