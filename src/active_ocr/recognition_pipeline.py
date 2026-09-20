"""READ2016 page-text active-learning control for LightOnOCR or Qwen on Modal."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
from pathlib import Path
from uuid import uuid4

import modal

BUNDLE = "94ed4f7f79d5d1110b7454c12a3b86d643cdb9dc0dece83d63734c2e4d87dcb6"
IMAGE_ID = os.getenv("FYP_MODAL_IMAGE_ID", "im-jgZ7uRYWA9OT6Tl7sPh8iN")
INPUT_VOLUME_ID = os.getenv("FYP_MODAL_INPUT_VOLUME_ID", "vo-nEKaAnh02D9OQkgGVF9QPc")
OUTPUT_VOLUME_ID = os.getenv("FYP_MODAL_OUTPUT_VOLUME_ID", "vo-J6KGzVVUq9jFE6mJFHO1PJ")
image = modal.Image.from_id(IMAGE_ID).env({"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"})
input_volume = modal.Volume.from_id(INPUT_VOLUME_ID)
output_volume = modal.Volume.from_id(OUTPUT_VOLUME_ID)
app = modal.App("fyp-read2016-page-text")


def sha_json(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_model(spec):
    import torch
    from transformers import (
        AutoProcessor,
        LightOnOcrForConditionalGeneration,
        LightOnOcrProcessor,
        Qwen3VLForConditionalGeneration,
    )

    if spec["source"] == "pinned-input-bundle":
        source = "/input/model"
        return (
            AutoProcessor.from_pretrained(source, local_files_only=True),
            Qwen3VLForConditionalGeneration.from_pretrained(
                source, torch_dtype=torch.bfloat16, local_files_only=True
            ),
        )
    if spec["source"] == "huggingface":
        source = spec["repository"]
        revision = spec["revision"]
        return (
            LightOnOcrProcessor.from_pretrained(source, revision=revision),
            LightOnOcrForConditionalGeneration.from_pretrained(
                source, revision=revision, torch_dtype=torch.bfloat16
            ),
        )
    raise ValueError("unsupported pinned model source")


def _inputs(processor, record, config, spec, *, train=False):
    from PIL import Image

    with Image.open("/input/images/" + record["image_sha256"]) as opened:
        page = opened.convert("RGB")
    edge = config["training"]["image_longest_edge"]
    page.thumbnail((edge, edge))
    content = [{"type": "image", "image": page}]
    if spec["prompt"] is not None:
        content.append({"type": "text", "text": spec["prompt"]})
    user = {"role": "user", "content": content}
    prompt = processor.apply_chat_template(
        [user], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    if not train:
        return prompt
    assistant = {"role": "assistant", "content": [{"type": "text", "text": record["target"]}]}
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
        (i for i, (a, b) in enumerate(zip(prompt_ids, full_ids, strict=False)) if a != b),
        prefix,
    )
    if common < prefix - 1:
        raise ValueError("train target changed more than final prompt token")
    if "pixel_values" not in full or not full["pixel_values"].numel():
        raise ValueError("training page has no image pixels")
    if not 0 < len(full_ids) - prefix <= config["inference"]["max_new_tokens"]:
        raise ValueError("target exceeds output budget")
    full["labels"] = full["input_ids"].clone()
    full["labels"][:, :common] = -100
    return full


def _predict(model, processor, records, config, spec):
    import torch

    decode = config["inference"]
    model = model.to("cuda").eval()
    predictions = []
    for record in records:
        batch = _inputs(processor, record, config, spec)
        batch = {
            k: v.to(device="cuda", dtype=torch.bfloat16) if v.is_floating_point() else v.to("cuda")
            for k, v in batch.items()
        }
        with torch.inference_mode():
            ids = model.generate(
                **batch,
                max_new_tokens=decode["max_new_tokens"],
                do_sample=decode["do_sample"],
                repetition_penalty=decode["repetition_penalty"],
            )[0, batch["input_ids"].shape[1] :]
        predictions.append(
            {
                "page_id": record["page_id"],
                "tokens": len(ids),
                "text": processor.decode(ids, skip_special_tokens=True),
            }
        )
    return predictions


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
def run_stage(payload: dict, stage: int, run_name: str, holdout: bool = False) -> dict:
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import Trainer, TrainingArguments

    config = payload["config"]
    spec = config["models"][payload["model"]]
    selection = config["selection"]
    fit = config["training"]
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}", run_name):
        raise ValueError("invalid run name")
    if stage not in range(selection["rounds"] + 1):
        raise ValueError("wrong stage")
    selected_ids = payload["selected_ids"]
    if (
        len(selected_ids) != selection["pages_per_round"] * stage
        or len(set(selected_ids)) != len(selected_ids)
        or any(":train:" not in i for i in selected_ids)
    ):
        raise ValueError("wrong selected TRAIN pages")
    if holdout:
        if payload["train"]:
            raise ValueError("holdout must not upload training targets")
    elif [r["page_id"] for r in payload["train"]] != selected_ids:
        raise ValueError("fit targets differ from selected pages")
    validation = payload["validation"]
    expected = 42 if holdout else 8
    if (
        len(validation) != expected
        or len({r["page_id"] for r in validation}) != expected
        or any("target" in r or ":validation:" not in r["page_id"] for r in validation)
    ):
        raise ValueError("validation labels must stay local")
    name = f"round-{stage}" + ("-holdout" if holdout else "")
    root = Path("/output") / run_name / name
    receipt_path = root / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt["source_run"],
            receipt["selected_ids"],
            receipt["recipe_sha256"],
            receipt["validation_ids"],
            receipt["model_repository"],
        ) != (
            payload["source_run"],
            selected_ids,
            payload["recipe_sha256"],
            [r["page_id"] for r in validation],
            spec["repository"],
        ):
            raise ValueError("existing remote receipt differs from selected stage")
        return receipt
    work = root.with_name(root.name + "-attempt-" + uuid4().hex)
    work.mkdir(parents=True, exist_ok=False)
    for record in (*payload["train"], *validation):
        if not re.fullmatch(r"[0-9a-f]{64}", record["image_sha256"]):
            raise ValueError("invalid image content address")
        path = Path("/input/images") / record["image_sha256"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["image_sha256"]:
            raise ValueError("input image checksum mismatch")
    processor, model = _load_model(spec)
    train_metrics = None
    trainable = 0
    steps = 0
    adapter_sha = None
    if holdout and stage:
        previous = Path("/output") / run_name / f"round-{stage}"
        prior = json.loads((previous / "receipt.json").read_text())
        if (prior["source_run"], prior["selected_ids"], prior["recipe_sha256"]) != (
            payload["source_run"],
            selected_ids,
            payload["recipe_sha256"],
        ):
            raise ValueError("holdout checkpoint does not match selected stage")
        adapter = previous / "adapter"
        adapter_sha = hashlib.sha256(
            (adapter / "adapter_model.safetensors").read_bytes()
        ).hexdigest()
        if adapter_sha != prior["adapter_sha256"]:
            raise ValueError("holdout adapter checksum mismatch")
        model = PeftModel.from_pretrained(model, adapter)
    elif stage:
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
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable <= 0:
            raise ValueError("LoRA has no trainable parameters")

        class Selected:
            def __len__(self):
                return len(payload["train"])

            def __getitem__(self, index):
                return payload["train"][index]

        def collate(batch):
            if len(batch) != 1:
                raise ValueError("one page per training batch")
            return _inputs(processor, batch[0], config, spec, train=True)

        args = TrainingArguments(
            output_dir=str(work / "trainer"),
            num_train_epochs=fit["epochs"],
            per_device_train_batch_size=fit["batch_size"],
            gradient_accumulation_steps=fit["gradient_accumulation_steps"],
            learning_rate=fit["learning_rate"],
            lr_scheduler_type=fit["scheduler"],
            warmup_ratio=fit["warmup_ratio"],
            optim=fit["optimizer"],
            bf16=fit["precision"] == "bf16",
            gradient_checkpointing=True,
            remove_unused_columns=False,
            save_strategy="no",
            report_to="none",
            logging_steps=16,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            seed=selection["seed"],
        )
        trainer = Trainer(model=model, args=args, train_dataset=Selected(), data_collator=collate)
        outcome = trainer.train()
        steps = trainer.state.global_step
        if steps != 48 * stage:
            raise ValueError("unexpected optimizer step count")
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
    predictions = _predict(model, processor, validation, config, spec)
    receipt = {
        "source_run": payload["source_run"],
        "recipe_sha256": payload["recipe_sha256"],
        "model_repository": spec["repository"],
        "model_revision": spec["revision"],
        "stage": stage,
        "holdout": holdout,
        "selected_ids": selected_ids,
        "validation_ids": [r["page_id"] for r in validation],
        "adapter_sha256": adapter_sha,
        "trainable_parameters": trainable,
        "optimizer_steps": steps,
        "train_metrics": train_metrics,
        "predictions": predictions,
    }
    (work / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False))
    work.rename(root)
    output_volume.commit()
    return receipt


def main() -> None:
    from active_ocr.active_learning import select_pages
    from active_ocr.evaluation import page_text_view
    from active_ocr.integrations.simulation import LocalOracle, PageTextEvaluatorV1
    from active_ocr.models import (
        Box,
        Prediction,
        PredictionPurpose,
        PredictionStatus,
        Region,
        Split,
        Strategy,
    )
    from active_ocr.pipeline import Pipeline

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_runs", type=Path, help="frozen source-run SQLite store")
    parser.add_argument("results", type=Path, help="new or resumable result directory")
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/recipes/read2016-page-text.json")
    )
    parser.add_argument("--model", required=True, choices=("lighton", "qwen"))
    parser.add_argument("--prepare-only", action="store_true", help="verify selection without GPU")
    parser.add_argument("--through-stage", type=int, choices=range(4), default=3)
    parser.add_argument(
        "--holdout", action="store_true", help="evaluate remaining 42 VAL pages after all fits"
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}", args.results.name):
        raise ValueError("results directory basename must be a safe run name")
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    selection = config["selection"]
    if (
        selection["strategy"],
        selection["fit"],
        selection["rounds"],
        selection["pages_per_round"],
    ) != ("random", "reset_from_base_on_cumulative_selected_pages", 3, 64):
        raise ValueError("unsupported frozen round protocol")
    if config["inference"]["evaluator"] != PageTextEvaluatorV1.identifier:
        raise ValueError("wrong text evaluator")
    if args.holdout and args.through_stage != selection["rounds"]:
        raise ValueError("holdout requires all three acquired rounds")
    source_run = Pipeline.for_simulation(args.source_runs).get_simulation(config["source_run_id"])
    if source_run.dataset.manifest_sha256 != config["source_manifest_sha256"]:
        raise ValueError("frozen source manifest differs from recipe")
    oracle = LocalOracle(source_run.dataset)
    pages = {p.id: p for p in source_run.dataset.pages}
    train_ids = {p.id for p in pages.values() if p.split is Split.TRAIN}
    validation_ids = list(config["validation_page_ids"])
    if list(source_run.config.validation_page_ids or []) != validation_ids:
        raise ValueError("recipe validation pages differ from frozen source")
    holdout_ids = sorted(
        p.id for p in pages.values() if p.split is Split.VALIDATION and p.id not in validation_ids
    )
    if len(holdout_ids) != 42:
        raise ValueError("expected untouched 42-page engineering validation slice")
    spec = config["models"][args.model]
    recipe = {
        "model": args.model,
        "config": config,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "bundle": BUNDLE,
        "image_id": IMAGE_ID,
        "input_volume": INPUT_VOLUME_ID,
        "output_volume": OUTPUT_VOLUME_ID,
        "holdout_ids": holdout_ids,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    recipe_sha = sha_json(recipe)
    if not args.prepare_only:
        args.results.mkdir(parents=True, exist_ok=True)
        recipe_path = args.results / "recipe.json"
        if recipe_path.exists():
            if json.loads(recipe_path.read_text()) != recipe:
                raise ValueError("result directory is bound to a different recipe")
        else:
            with recipe_path.open("x") as stream:
                json.dump(recipe, stream, ensure_ascii=False, indent=2)
    selected = []

    def execute(stage, ids, *, holdout=False):
        targets = (
            []
            if holdout
            else [
                {
                    "page_id": e.page.id,
                    "image_sha256": e.page.image_sha256,
                    "target": page_text_view([r.text for r in e.regions]),
                }
                for e in oracle.reveal(tuple(selected))
            ]
        )
        payload = {
            "source_run": source_run.id,
            "recipe_sha256": recipe_sha,
            "model": args.model,
            "config": config,
            "selected_ids": list(selected),
            "train": targets,
            "validation": [{"page_id": i, "image_sha256": pages[i].image_sha256} for i in ids],
        }
        suffix = "-holdout" if holdout else ""
        path = args.results / f"round-{stage}{suffix}.json"
        if path.exists():
            result = json.loads(path.read_text())
            receipt = result["receipt"]
        else:
            with modal.enable_output(), app.run(environment_name="main"):
                receipt = run_stage.remote(payload, stage, args.results.name, holdout)
            result = {"receipt": receipt}
        if (
            receipt["source_run"],
            receipt["recipe_sha256"],
            receipt["stage"],
            receipt["holdout"],
            receipt["selected_ids"],
            receipt["validation_ids"],
            receipt["model_repository"],
            receipt["model_revision"],
        ) != (
            source_run.id,
            recipe_sha,
            stage,
            holdout,
            selected,
            ids,
            spec["repository"],
            spec["revision"],
        ):
            raise ValueError("receipt differs from frozen selection/model/validation")
        predictions = []
        for raw in receipt["predictions"]:
            page = pages[raw["page_id"]]
            status = (
                PredictionStatus.TRUNCATED
                if raw["tokens"] >= config["inference"]["max_new_tokens"]
                else (PredictionStatus.OK if raw["text"] else PredictionStatus.INVALID_OUTPUT)
            )
            regions = (
                (
                    Region(
                        id="page-text",
                        box=Box(x=0, y=0, width=page.width, height=page.height),
                        text=raw["text"],
                    ),
                )
                if status is PredictionStatus.OK
                else ()
            )
            predictions.append(
                Prediction(
                    page_id=raw["page_id"],
                    experiment_id=args.results.name,
                    round_number=stage,
                    model_id=(
                        "adapter:sha256:" + receipt["adapter_sha256"]
                        if stage
                        else "hf:" + spec["repository"] + "@" + spec["revision"]
                    ),
                    regions=regions,
                    status=status,
                    purpose=(
                        PredictionPurpose.BASELINE_VALIDATION
                        if stage == 0
                        else PredictionPurpose.VALIDATION
                    ),
                )
            )
        metrics = oracle.evaluate_validation(tuple(predictions), PageTextEvaluatorV1(), tuple(ids))
        if "metrics" in result and result["metrics"] != metrics:
            raise ValueError("saved stage metrics differ from current evaluator")
        if not path.exists():
            result["metrics"] = metrics
            with path.open("x") as stream:
                json.dump(result, stream, ensure_ascii=False)
        print(
            json.dumps(
                {
                    "model": args.model,
                    "stage": stage,
                    "holdout": holdout,
                    "labelled_pages": len(selected),
                    "cer": metrics["cer"],
                    "wer": metrics["wer"],
                    "failed_pages": metrics["failed_pages"],
                }
            )
        )

    for stage in range(args.through_stage + 1):
        if stage:
            selected.extend(
                select_pages(
                    train_ids - set(selected),
                    Strategy.RANDOM,
                    selection["pages_per_round"],
                    selection["seed"] + stage - 1,
                )
            )
        if args.prepare_only:
            print(
                json.dumps(
                    {
                        "model": args.model,
                        "stage": stage,
                        "labelled_pages": len(selected),
                        "selected_sha256": sha_json(selected),
                        "validation_count": len(validation_ids),
                    }
                )
            )
            continue
        execute(stage, validation_ids)
    if args.holdout and not args.prepare_only:
        selected = []
        for stage in range(selection["rounds"] + 1):
            if stage:
                selected.extend(
                    select_pages(
                        train_ids - set(selected),
                        Strategy.RANDOM,
                        selection["pages_per_round"],
                        selection["seed"] + stage - 1,
                    )
                )
            execute(stage, holdout_ids, holdout=True)


if __name__ == "__main__":
    main()
