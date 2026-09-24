"""Isolated READ2016 known-box line recognition and page-level active learning on L4."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
from pathlib import Path
from uuid import uuid4

import modal

from active_ocr.recognition_pipeline import (
    BUNDLE,
    IMAGE_ID,
    INPUT_VOLUME_ID,
    OUTPUT_VOLUME_ID,
    _load_model,
    image,
    input_volume,
    output_volume,
    sha_json,
)

app = modal.App("fyp-read2016-known-box-regions")
MOUNTS = {
    "/input": input_volume.with_mount_options(sub_path="/bundles/" + BUNDLE, read_only=True),
    "/output": output_volume,
}


def _crop(page, box, padding):
    """PAGE coordinates are in original image pixels; keep them and reading order unchanged."""
    x, y, width, height = (box[k] for k in ("x", "y", "width", "height"))
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > page.width
        or y + height > page.height
    ):
        raise ValueError("source box outside page")
    return page.crop(
        (
            max(0, math.floor(x - padding)),
            max(0, math.floor(y - padding)),
            min(page.width, math.ceil(x + width + padding)),
            min(page.height, math.ceil(y + height + padding)),
        )
    )


def _inputs(processor, page, region, config, *, train=False):

    crop = _crop(page, region["box"], config["crop_padding_px"])
    crop.thumbnail((config["training"]["image_longest_edge"],) * 2)
    user = {
        "role": "user",
        "content": [
            {"type": "image", "image": crop},
            {"type": "text", "text": config["prompt"]},
        ],
    }
    prompt = processor.apply_chat_template(
        [user], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    if not train:
        return prompt
    assistant = {"role": "assistant", "content": [{"type": "text", "text": region["target"]}]}
    full = processor.apply_chat_template(
        [user, assistant],
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    prefix = prompt["input_ids"].shape[1]
    prompt_ids = prompt["input_ids"][0].tolist()
    full_ids = full["input_ids"][0].tolist()
    common = next(
        (i for i, (a, b) in enumerate(zip(prompt_ids, full_ids, strict=False)) if a != b), prefix
    )
    if (
        common < prefix - 1
        or not 0 < len(full_ids) - prefix <= config["inference"]["max_new_tokens"]
    ):
        raise ValueError("invalid supervised target span")
    if "pixel_values" not in full or not full["pixel_values"].numel():
        raise ValueError("region has no pixels")
    full["labels"] = full["input_ids"].clone()
    full["labels"][:, :common] = -100
    return full


def _cuda(batch):
    import torch

    return {
        key: value.to(device="cuda", dtype=torch.bfloat16)
        if value.is_floating_point()
        else value.to("cuda")
        for key, value in batch.items()
    }


def _load_page(record):
    from PIL import Image

    path = Path("/input/images") / record["image_sha256"]
    with Image.open(path) as opened:
        page = opened.convert("RGB")
    if page.size != (record["width"], record["height"]):
        raise ValueError("PAGE dimensions differ from image")
    return page


def _check_records(records, *, labelled):
    if len({r["page_id"] for r in records}) != len(records):
        raise ValueError("duplicate page IDs")
    for record in records:
        if not re.fullmatch(r"[0-9a-f]{64}", record["image_sha256"]):
            raise ValueError("invalid image content address")
        if not record["regions"] or len({r["id"] for r in record["regions"]}) != len(
            record["regions"]
        ):
            raise ValueError("missing or duplicate ordered regions")
        if any(("target" in region) != labelled for region in record["regions"]):
            raise ValueError("label leakage or missing selected target")
        for region in record["regions"]:
            box = region["box"]
            if (
                box["x"] < 0
                or box["y"] < 0
                or box["width"] <= 0
                or box["height"] <= 0
                or box["x"] + box["width"] > record["width"]
                or box["y"] + box["height"] > record["height"]
            ):
                raise ValueError("invalid source box")


@app.function(image=image, gpu="L4", cpu=2, memory=32768, timeout=7200, volumes=MOUNTS)
def run_stage(payload: dict, stage: int, run_name: str) -> dict:
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import Trainer, TrainingArguments

    config = payload["config"]
    fit = config["training"]
    spec = config["models"]["qwen"]
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}", run_name):
        raise ValueError("unsafe run name")
    selected = payload["selected_ids"]
    if (
        len(selected) != config["selection"]["pages_per_round"] * stage
        or len(set(selected)) != len(selected)
        or any(":train:" not in i for i in selected)
    ):
        raise ValueError("wrong selected TRAIN pages")
    train = payload["train"]
    validation = payload["validation"]
    if [r["page_id"] for r in train] != selected:
        raise ValueError("training labels differ from selected pages")
    if not validation or any(":validation:" not in r["page_id"] for r in validation):
        raise ValueError("wrong validation membership")
    _check_records(train, labelled=True)
    _check_records(validation, labelled=False)
    root = Path("/output") / run_name / f"round-{stage}"
    if root.exists():
        receipt = json.loads((root / "receipt.json").read_text())
        if (receipt["recipe_sha256"], receipt["selected_ids"], receipt["validation_ids"]) != (
            payload["recipe_sha256"],
            selected,
            [r["page_id"] for r in validation],
        ):
            raise ValueError("remote receipt identity mismatch")
        return receipt
    work = root.with_name(root.name + "-attempt-" + uuid4().hex)
    work.mkdir(parents=True, exist_ok=False)
    for record in (*train, *validation):
        path = Path("/input/images") / record["image_sha256"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["image_sha256"]:
            raise ValueError("input image checksum mismatch")
    processor, model = _load_model(spec)
    adapter_sha = None
    steps = 0
    train_metrics = None
    if stage:
        model.config.use_cache = False
        model = get_peft_model(
            model,
            LoraConfig(
                r=fit["lora_rank"],
                lora_alpha=fit["lora_alpha"],
                lora_dropout=fit["lora_dropout"],
                target_modules=fit["lora_targets"],
                task_type=TaskType.CAUSAL_LM,
            ),
        )
        samples = [(record, region) for record in train for region in record["regions"]]

        class Selected:
            def __len__(self):
                return len(samples)

            def __getitem__(self, index):
                return samples[index]

        def collate(batch):
            if len(batch) != 1:
                raise ValueError("one crop per training batch")
            record, region = batch[0]
            return _inputs(processor, _load_page(record), region, config, train=True)

        args = TrainingArguments(
            output_dir=str(work / "trainer"),
            num_train_epochs=fit["epochs"],
            per_device_train_batch_size=1,
            gradient_accumulation_steps=fit["gradient_accumulation_steps"],
            learning_rate=fit["learning_rate"],
            lr_scheduler_type=fit["scheduler"],
            warmup_ratio=fit["warmup_ratio"],
            optim=fit["optimizer"],
            bf16=True,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            save_strategy="no",
            report_to="none",
            logging_steps=50,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            seed=config["selection"]["seed"],
        )
        trainer = Trainer(model=model, args=args, train_dataset=Selected(), data_collator=collate)
        outcome = trainer.train()
        steps = trainer.state.global_step
        if steps <= 0:
            raise ValueError("no training steps")
        adapter = work / "adapter"
        model.save_pretrained(adapter, safe_serialization=True)
        adapter_sha = hashlib.sha256(
            (adapter / "adapter_model.safetensors").read_bytes()
        ).hexdigest()
        train_metrics = {
            k: float(v) for k, v in outcome.metrics.items() if isinstance(v, (int, float))
        }
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        _, model = _load_model(spec)
        model = PeftModel.from_pretrained(model, adapter)
    model = model.to("cuda").eval()
    predictions = []
    for record in validation:
        page = _load_page(record)
        rows = []
        for region in record["regions"]:
            batch = _cuda(_inputs(processor, page, region, config))
            with torch.inference_mode():
                ids = model.generate(
                    **batch,
                    max_new_tokens=config["inference"]["max_new_tokens"],
                    do_sample=False,
                    repetition_penalty=config["inference"]["repetition_penalty"],
                )[0, batch["input_ids"].shape[1] :]
            rows.append(
                {
                    "id": region["id"],
                    "text": processor.decode(ids, skip_special_tokens=True)
                    .strip()
                    .replace("\n", " "),
                    "tokens": len(ids),
                    "truncated": len(ids) >= config["inference"]["max_new_tokens"],
                }
            )
        predictions.append({"page_id": record["page_id"], "regions": rows})
    receipt = {
        "source_run": payload["source_run"],
        "recipe_sha256": payload["recipe_sha256"],
        "model_repository": spec["repository"],
        "model_revision": spec["revision"],
        "gpu": "L4",
        "stage": stage,
        "selected_ids": selected,
        "validation_ids": [r["page_id"] for r in validation],
        "training_regions": sum(len(r["regions"]) for r in train),
        "adapter_sha256": adapter_sha,
        "optimizer_steps": steps,
        "train_metrics": train_metrics,
        "predictions": predictions,
    }
    (work / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False))
    work.rename(root)
    output_volume.commit()
    return receipt


@app.function(image=image, gpu="L4", cpu=2, memory=32768, timeout=7200, volumes=MOUNTS)
def score_pool(records, config, checkpoint, mode):
    """Score three fixed line positions per TRAIN page without loading reference text."""
    import torch
    from peft import PeftModel

    if mode not in ("uncertainty", "visual"):
        raise ValueError("unknown scoring mode")
    _check_records(records, labelled=False)
    spec = config["models"]["qwen"]
    processor, model = _load_model(spec)
    if checkpoint:
        path = Path("/output") / checkpoint["run_name"] / f"round-{checkpoint['stage']}"
        receipt = json.loads((path / "receipt.json").read_text())
        if (receipt["adapter_sha256"], receipt["selected_ids"], receipt["recipe_sha256"]) != (
            checkpoint["adapter_sha256"],
            checkpoint["selected_ids"],
            checkpoint["recipe_sha256"],
        ):
            raise ValueError("score checkpoint identity mismatch")
        if (
            hashlib.sha256((path / "adapter/adapter_model.safetensors").read_bytes()).hexdigest()
            != checkpoint["adapter_sha256"]
        ):
            raise ValueError("score adapter checksum mismatch")
        model = PeftModel.from_pretrained(model, path / "adapter")
    model = model.to("cuda").eval()
    output = []
    for record in records:
        page = _load_page(record)
        regions = record["regions"]
        positions = sorted({len(regions) // 4, len(regions) // 2, 3 * len(regions) // 4})
        entropies, confidences, features = [], [], []
        for index in positions:
            batch = _cuda(_inputs(processor, page, regions[index], config))
            with torch.inference_mode():
                if mode == "visual":
                    feature = (
                        model.get_image_features(batch["pixel_values"], batch["image_grid_thw"])
                        .pooler_output[0]
                        .float()
                        .mean(dim=0)
                    )
                    features.append(feature)
                    continue
                generated = model.generate(
                    **batch,
                    max_new_tokens=config["inference"]["max_new_tokens"],
                    do_sample=False,
                    repetition_penalty=config["inference"]["repetition_penalty"],
                    return_dict_in_generate=True,
                    output_logits=True,
                )
                ids = generated.sequences[0, batch["input_ids"].shape[1] :]
                if not len(ids) or len(ids) != len(generated.logits):
                    raise ValueError("scoring token/logit alignment failed")
                entropy_sum = logp_sum = 0.0
                for token, logits in zip(ids, generated.logits, strict=True):
                    logp = torch.log_softmax(logits[0].float(), dim=-1)
                    entropy_sum += float(-(logp.exp() * logp).sum())
                    logp_sum += float(logp[token])
                entropies.append(entropy_sum / (len(ids) * math.log(generated.logits[0].shape[-1])))
                confidences.append(math.exp(logp_sum / len(ids)))
        if mode == "visual":
            vector = torch.stack(features).mean(dim=0)
            vector /= vector.norm()
            row = {"page_id": record["page_id"], "embedding": vector.tolist()}
        else:
            row = {
                "page_id": record["page_id"],
                "entropy": sum(entropies) / len(entropies),
                "confidence": sum(confidences) / len(confidences),
            }
        output.append(row)
    return output


def _record(page, regions, targets=None):
    rows = []
    for region in regions:
        row = {"id": region["id"], "box": region["box"]}
        if targets is not None:
            row["target"] = targets[region["id"]]
        rows.append(row)
    return {
        "page_id": page.id,
        "image_sha256": page.image_sha256,
        "width": page.width,
        "height": page.height,
        "regions": rows,
    }


def _save(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"saved artifact differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    return value


def _metrics(receipt, pages, geometry, oracle, validation_ids, experiment_id):
    from active_ocr.integrations.simulation import PageTextEvaluatorV1
    from active_ocr.models import Box, Prediction, PredictionPurpose, PredictionStatus, Region

    if [r["page_id"] for r in receipt["predictions"]] != validation_ids:
        raise ValueError("validation page order mismatch")
    predictions = []
    for item in receipt["predictions"]:
        page = pages[item["page_id"]]
        if [r["id"] for r in item["regions"]] != [r["id"] for r in geometry[page.id]]:
            raise ValueError("predicted region order differs from source PAGE order")
        regions = tuple(
            Region(
                id=raw["id"],
                box=Box.model_validate(reference["box"]),
                text=("" if raw["truncated"] else raw["text"]),
            )
            for raw, reference in zip(item["regions"], geometry[page.id], strict=True)
        )
        predictions.append(
            Prediction(
                page_id=page.id,
                experiment_id=experiment_id,
                round_number=receipt["stage"],
                model_id=(
                    "adapter:sha256:" + receipt["adapter_sha256"]
                    if receipt["stage"]
                    else "hf:" + receipt["model_repository"] + "@" + receipt["model_revision"]
                ),
                regions=regions,
                status=PredictionStatus.OK,
                purpose=(
                    PredictionPurpose.VALIDATION
                    if receipt["stage"]
                    else PredictionPurpose.BASELINE_VALIDATION
                ),
            )
        )
    result = oracle.evaluate_validation(
        tuple(predictions), PageTextEvaluatorV1(), tuple(validation_ids)
    )
    result["truncated_regions"] = sum(
        r["truncated"] for p in receipt["predictions"] for r in p["regions"]
    )
    result["empty_regions"] = sum(
        not r["text"] for p in receipt["predictions"] for r in p["regions"]
    )
    result["regions"] = sum(len(p["regions"]) for p in receipt["predictions"])
    return result


def main():
    from active_ocr.acquisition_pipeline import kcenter_greedy
    from active_ocr.active_learning import select_pages
    from active_ocr.integrations.simulation import LocalOracle
    from active_ocr.models import Prediction, Split, Strategy
    from active_ocr.pipeline import Pipeline

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_runs", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/recipes/read2016-known-box-regions.json")
    )
    parser.add_argument(
        "--strategy",
        choices=("random", "entropy", "least_confidence", "kcenter_greedy"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=824)
    parser.add_argument("--through-stage", type=int, choices=range(4), default=3)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if (
        args.seed not in (824, 825, 826)
        or config["selection"]["rounds"] != 3
        or config["selection"]["pages_per_round"] != 64
    ):
        raise ValueError("unsupported frozen comparison")
    config["selection"]["seed"] = args.seed
    source = Pipeline.for_simulation(args.source_runs).get_simulation(config["source_run_id"])
    if source.dataset.manifest_sha256 != config["source_manifest_sha256"]:
        raise ValueError("frozen source manifest mismatch")
    pages = {p.id: p for p in source.dataset.pages}
    manifest_bytes = Path(source.dataset.source_manifest).read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != source.dataset.manifest_sha256:
        raise ValueError("source PAGE manifest checksum mismatch")
    geometry = {}
    for line in manifest_bytes.splitlines():
        row = json.loads(line)
        page = pages[row["id"]]
        if (row["image_sha256"], row["width"], row["height"], row["split"]) != (
            page.image_sha256,
            page.width,
            page.height,
            page.split.value,
        ):
            raise ValueError("geometry page differs from frozen source")
        geometry[page.id] = [{"id": r["id"], "box": r["box"]} for r in row["regions"]]
    if len(geometry) != len(pages):
        raise ValueError("geometry coverage mismatch")
    train_ids = {p.id for p in pages.values() if p.split is Split.TRAIN}
    val8 = set(config["validation_page_ids"])
    val42 = sorted(p.id for p in pages.values() if p.split is Split.VALIDATION and p.id not in val8)
    if (len(train_ids), len(val42)) != (350, 42):
        raise ValueError("unexpected official split")
    oracle = LocalOracle(source.dataset)
    run_name = args.results.name
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}", run_name):
        raise ValueError("unsafe result directory name")
    recipe = {
        "config": config,
        "strategy": args.strategy,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_run": source.id,
        "input_bundle": BUNDLE,
        "modal_image": IMAGE_ID,
        "input_volume": INPUT_VOLUME_ID,
        "output_volume": OUTPUT_VOLUME_ID,
        "validation_ids": val42,
        "gpu": "L4",
    }
    recipe_sha = sha_json(recipe)
    selected = []
    anchor = args.results.parent / f"region-exp012-random-seed{args.seed}"
    if not args.prepare_only:
        _save(args.results / "recipe.json", recipe)
        if args.strategy != "random":
            anchor_recipe = json.loads((anchor / "recipe.json").read_text())
            if (
                anchor_recipe["config"] != config
                or anchor_recipe["runner_sha256"] != recipe["runner_sha256"]
            ):
                raise ValueError("random anchor differs from current code/recipe")
    for stage in range(args.through_stage + 1):
        if stage:
            pool = sorted(train_ids - set(selected))
            if args.strategy == "random" or stage == 1:
                chosen = select_pages(pool, Strategy.RANDOM, 64, args.seed + stage - 1)
            else:
                mode = "visual" if args.strategy == "kcenter_greedy" else "uncertainty"
                previous = anchor if stage == 2 else args.results
                prior = json.loads((previous / f"round-{stage - 1}.json").read_text())["receipt"]
                checkpoint = (
                    None
                    if mode == "visual"
                    else {
                        "run_name": previous.name,
                        "stage": stage - 1,
                        "selected_ids": selected,
                        "adapter_sha256": prior["adapter_sha256"],
                        "recipe_sha256": prior["recipe_sha256"],
                    }
                )
                ids = sorted(train_ids) if mode == "visual" else pool
                records = [_record(pages[i], geometry[i]) for i in ids]
                score_config = (
                    config
                    if mode == "uncertainty"
                    else {
                        **config,
                        "selection": {
                            key: value
                            for key, value in config["selection"].items()
                            if key != "seed"
                        },
                    }
                )
                identity = {
                    "mode": mode,
                    "checkpoint": checkpoint,
                    "records_sha256": sha_json(records),
                    "config_sha256": sha_json(score_config),
                    "runner_sha256": recipe["runner_sha256"],
                }
                key = sha_json(identity)
                cache_root = args.results.parent / "_score-cache"
                cache = cache_root / f"{key}.json"
                if cache.exists():
                    saved = json.loads(cache.read_text())
                    if saved["identity"] != identity:
                        raise ValueError("score cache identity mismatch")
                    rows = saved["rows"]
                elif args.prepare_only:
                    rows = []
                else:
                    rows = []
                    with modal.enable_output(), app.run(environment_name="main"):
                        for start in range(0, len(records), 16):
                            part = cache_root / f"{key}-part-{start // 16}.json"
                            part_identity = {**identity, "start": start}
                            if part.exists():
                                saved = json.loads(part.read_text())
                                if saved["identity"] != part_identity:
                                    raise ValueError("score part identity mismatch")
                                chunk = saved["rows"]
                            else:
                                chunk = score_pool.remote(
                                    records[start : start + 16], config, checkpoint, mode
                                )
                                _save(part, {"identity": part_identity, "rows": chunk})
                            rows.extend(chunk)
                    _save(cache, {"identity": identity, "rows": rows})
                if args.prepare_only:
                    print(
                        json.dumps(
                            {
                                "stage": stage,
                                "strategy": args.strategy,
                                "pool_pages": len(pool),
                                "requires_gpu_scoring": True,
                            }
                        )
                    )
                    break
                if [r["page_id"] for r in rows] != ids:
                    raise ValueError("score coverage mismatch")
                if mode == "visual":
                    chosen = kcenter_greedy(
                        {r["page_id"]: r["embedding"] for r in rows}, selected, pool, 64
                    )
                else:
                    estimates = [
                        Prediction(
                            page_id=r["page_id"],
                            experiment_id=run_name,
                            round_number=stage - 1,
                            model_id="adapter:sha256:" + checkpoint["adapter_sha256"],
                            entropy=r["entropy"],
                            confidence=r["confidence"],
                        )
                        for r in rows
                    ]
                    chosen = select_pages(
                        pool, Strategy(args.strategy), 64, 0, predictions=estimates
                    )
            selected.extend(chosen)
            if (
                len(selected) != 64 * stage
                or len(set(selected)) != len(selected)
                or not set(selected) <= train_ids
            ):
                raise ValueError("selected pages violate TRAIN budget")
            if not args.prepare_only:
                _save(
                    args.results / f"selection-round-{stage}.json",
                    {
                        "selected_ids": selected,
                        "new_ids": list(chosen),
                        "selected_sha256": sha_json(selected),
                    },
                )
        if args.prepare_only:
            print(
                json.dumps(
                    {
                        "stage": stage,
                        "selected_pages": len(selected),
                        "selected_sha256": sha_json(selected),
                    }
                )
            )
            continue
        if args.strategy != "random" and stage < 2:
            saved = json.loads((anchor / f"round-{stage}.json").read_text())
            if (
                saved["receipt"]["selected_ids"] != selected
                or saved["receipt"]["validation_ids"] != val42
            ):
                raise ValueError("random anchor coverage mismatch")
            print(
                json.dumps(
                    {
                        "strategy": args.strategy,
                        "seed": args.seed,
                        "stage": stage,
                        "pages": len(selected),
                        "cer": saved["metrics"]["cer"],
                        "wer": saved["metrics"]["wer"],
                        "reused_from": str(anchor),
                    }
                )
            )
            continue
        revealed = [] if not stage else oracle.reveal(tuple(selected))
        targets = {e.page.id: {r.id: r.text for r in e.regions} for e in revealed}
        payload = {
            "source_run": source.id,
            "recipe_sha256": recipe_sha,
            "config": config,
            "selected_ids": selected,
            "train": [_record(pages[i], geometry[i], targets[i]) for i in selected],
            "validation": [_record(pages[i], geometry[i]) for i in val42],
        }
        path = args.results / f"round-{stage}.json"
        if path.exists():
            receipt = json.loads(path.read_text())["receipt"]
        else:
            with modal.enable_output(), app.run(environment_name="main"):
                receipt = run_stage.remote(payload, stage, run_name)
        if (
            receipt["source_run"],
            receipt["recipe_sha256"],
            receipt["selected_ids"],
            receipt["validation_ids"],
            receipt["gpu"],
        ) != (source.id, recipe_sha, selected, val42, "L4"):
            raise ValueError("fit receipt identity mismatch")
        metrics = _metrics(receipt, pages, geometry, oracle, val42, run_name)
        _save(path, {"receipt": receipt, "metrics": metrics})
        print(
            json.dumps(
                {
                    "strategy": args.strategy,
                    "seed": args.seed,
                    "stage": stage,
                    "pages": len(selected),
                    "cer": metrics["cer"],
                    "wer": metrics["wer"],
                    "truncated_regions": metrics["truncated_regions"],
                }
            )
        )


if __name__ == "__main__":
    main()
