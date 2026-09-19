"""Three-round READ2016 page-text OCR study using a frozen source run and Modal.

The source run supplies validated image metadata and the local label oracle. Only
selected TRAIN targets cross to the GPU; validation truth stays in the evaluator.
"""

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

REPO = "lightonai/LightOnOCR-2-1B-base"
REVISION = "c1bc8eb6625be7b34178ff53481b242f7b8ffcb8"
SEED = 824
PAGES_PER_ROUND = 64
ROUNDS = 3
MAX_NEW_TOKENS = 1536
REPETITION_PENALTY = 1.1
BUNDLE = "94ed4f7f79d5d1110b7454c12a3b86d643cdb9dc0dece83d63734c2e4d87dcb6"
IMAGE_ID = os.getenv("FYP_MODAL_IMAGE_ID", "im-jgZ7uRYWA9OT6Tl7sPh8iN")
INPUT_VOLUME_ID = os.getenv("FYP_MODAL_INPUT_VOLUME_ID", "vo-nEKaAnh02D9OQkgGVF9QPc")
OUTPUT_VOLUME_ID = os.getenv("FYP_MODAL_OUTPUT_VOLUME_ID", "vo-J6KGzVVUq9jFE6mJFHO1PJ")
image = modal.Image.from_id(IMAGE_ID).env({"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"})
input_volume = modal.Volume.from_id(INPUT_VOLUME_ID)
output_volume = modal.Volume.from_id(OUTPUT_VOLUME_ID)
app = modal.App("fyp-lighton-read2016-three-rounds")


def sha_json(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@app.function(
    image=image,
    gpu="L4",
    cpu=2,
    memory=49152,
    timeout=3600,
    volumes={
        "/input": input_volume.with_mount_options(sub_path="/bundles/" + BUNDLE, read_only=True),
        "/output": output_volume,
    },
)
def run_stage(payload: dict, stage: int, run_name: str) -> dict:
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from PIL import Image
    from transformers import (
        LightOnOcrForConditionalGeneration,
        LightOnOcrProcessor,
        Trainer,
        TrainingArguments,
    )

    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}", run_name):
        raise ValueError("invalid run name")
    if stage not in range(ROUNDS + 1) or len(payload["train"]) != PAGES_PER_ROUND * stage:
        raise ValueError("wrong stage or selected budget")
    if len({r["page_id"] for r in payload["train"]}) != len(payload["train"]):
        raise ValueError("duplicate selected page")
    if len(payload["validation"]) != 8 or any("target" in r for r in payload["validation"]):
        raise ValueError("validation labels must stay local")
    if any(":train:" not in r["page_id"] for r in payload["train"]):
        raise ValueError("non-TRAIN fit target")
    if any(":validation:" not in r["page_id"] for r in payload["validation"]):
        raise ValueError("non-VAL evaluation image")
    root = Path("/output") / run_name / f"round-{stage}"
    receipt_path = root / "receipt.json"
    selected_ids = [r["page_id"] for r in payload["train"]]
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt["source_run"], receipt["selected_ids"], receipt["recipe_sha256"]) != (
            payload["source_run"],
            selected_ids,
            payload["recipe_sha256"],
        ):
            raise ValueError("existing remote receipt differs from selected stage")
        return receipt
    work = root.with_name(root.name + "-attempt-" + uuid4().hex)
    work.mkdir(parents=True, exist_ok=False)
    for record in (*payload["train"], *payload["validation"]):
        if not re.fullmatch(r"[0-9a-f]{64}", record["image_sha256"]):
            raise ValueError("invalid image content address")
        path = Path("/input/images") / record["image_sha256"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["image_sha256"]:
            raise ValueError("input image checksum mismatch")
    processor = LightOnOcrProcessor.from_pretrained(REPO, revision=REVISION)
    model = LightOnOcrForConditionalGeneration.from_pretrained(
        REPO, revision=REVISION, torch_dtype=torch.bfloat16
    )
    train_metrics = None
    trainable = 0
    steps = 0
    adapter_sha = None

    def encoded(record):
        with Image.open("/input/images/" + record["image_sha256"]) as opened:
            page = opened.convert("RGB")
        page.thumbnail((1540, 1540))
        user = {"role": "user", "content": [{"type": "image", "image": page}]}
        assistant = {"role": "assistant", "content": [{"type": "text", "text": record["target"]}]}
        prompt = processor.apply_chat_template(
            [user], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
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
        if not 0 < full["input_ids"].shape[1] - prefix <= MAX_NEW_TOKENS:
            raise ValueError("target exceeds output budget")
        full["labels"] = full["input_ids"].clone()
        full["labels"][:, :common] = -100
        return full

    if stage:
        model.config.use_cache = False
        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                target_modules="all-linear",
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
            return encoded(batch[0])

        args = TrainingArguments(
            output_dir=str(work / "trainer"),
            num_train_epochs=3.0,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            optim="adamw_torch",
            bf16=True,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            save_strategy="no",
            report_to="none",
            logging_steps=16,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            seed=SEED,
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
        model = LightOnOcrForConditionalGeneration.from_pretrained(
            REPO, revision=REVISION, torch_dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(model, adapter)
    model = model.to("cuda").eval()
    predictions = []
    for record in payload["validation"]:
        with Image.open("/input/images/" + record["image_sha256"]) as opened:
            page = opened.convert("RGB")
        page.thumbnail((1540, 1540))
        user = {"role": "user", "content": [{"type": "image", "image": page}]}
        batch = processor.apply_chat_template(
            [user], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        batch = {
            k: v.to(device="cuda", dtype=torch.bfloat16) if v.is_floating_point() else v.to("cuda")
            for k, v in batch.items()
        }
        with torch.inference_mode():
            ids = model.generate(
                **batch,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=REPETITION_PENALTY,
            )[0, batch["input_ids"].shape[1] :]
        predictions.append(
            {
                "page_id": record["page_id"],
                "tokens": len(ids),
                "text": processor.decode(ids, skip_special_tokens=True),
            }
        )
    receipt = {
        "source_run": payload["source_run"],
        "recipe_sha256": payload["recipe_sha256"],
        "model_repository": REPO,
        "model_revision": REVISION,
        "stage": stage,
        "selected_ids": selected_ids,
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
    parser.add_argument(
        "source_runs", type=Path, help="directory containing the frozen source-run SQLite store"
    )
    parser.add_argument("source_run_id")
    parser.add_argument("results", type=Path, help="new or resumable local result directory")
    parser.add_argument(
        "--prepare-only", action="store_true", help="verify selections without a GPU call"
    )
    parser.add_argument("--through-stage", type=int, choices=range(ROUNDS + 1), default=ROUNDS)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}", args.results.name):
        raise ValueError("results directory basename must be a safe run name")
    pipeline = Pipeline.for_simulation(args.source_runs)
    source_run = pipeline.get_simulation(args.source_run_id)
    oracle = LocalOracle(source_run.dataset)
    pages = {p.id: p for p in source_run.dataset.pages}
    train_ids = {p.id for p in pages.values() if p.split is Split.TRAIN}
    validation_ids = source_run.config.validation_page_ids
    if validation_ids is None or len(validation_ids) != 8:
        raise ValueError("this frozen engineering recipe requires eight fixed VAL pages")
    recipe = {
        "repo": REPO,
        "revision": REVISION,
        "seed": SEED,
        "batch": PAGES_PER_ROUND,
        "rounds": ROUNDS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "repetition_penalty": REPETITION_PENALTY,
        "bundle": BUNDLE,
        "image_id": IMAGE_ID,
        "input_volume": INPUT_VOLUME_ID,
        "output_volume": OUTPUT_VOLUME_ID,
        "source_manifest_sha256": source_run.dataset.manifest_sha256,
        "source_run": source_run.id,
        "validation_ids": list(validation_ids),
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
    for stage in range(args.through_stage + 1):
        if stage:
            selected.extend(
                select_pages(
                    train_ids - set(selected), Strategy.RANDOM, PAGES_PER_ROUND, SEED + stage - 1
                )
            )
        examples = oracle.reveal(tuple(selected))
        payload = {
            "source_run": source_run.id,
            "seed": SEED,
            "recipe_sha256": recipe_sha,
            "train": [
                {
                    "page_id": e.page.id,
                    "image_sha256": e.page.image_sha256,
                    "target": page_text_view([r.text for r in e.regions]),
                }
                for e in examples
            ],
            "validation": [
                {"page_id": i, "image_sha256": pages[i].image_sha256} for i in validation_ids
            ],
        }
        if args.prepare_only:
            print(
                json.dumps(
                    {
                        "stage": stage,
                        "labelled_pages": len(selected),
                        "selected_sha256": sha_json(selected),
                        "validation_count": len(validation_ids),
                    }
                )
            )
            continue
        path = args.results / f"round-{stage}.json"
        if path.exists():
            result = json.loads(path.read_text())
            receipt = result["receipt"]
        else:
            receipt = None
            with modal.enable_output(), app.run(environment_name="main"):
                receipt = run_stage.remote(payload, stage, args.results.name)
            if receipt is None:
                raise RuntimeError("Modal stage exited without a receipt")
            result = {"receipt": receipt}
        if (
            receipt["source_run"],
            receipt["recipe_sha256"],
            receipt["stage"],
            receipt["selected_ids"],
            receipt["model_repository"],
            receipt["model_revision"],
        ) != (source_run.id, recipe_sha, stage, selected, REPO, REVISION):
            raise ValueError("stage receipt differs from frozen selection")
        predictions = []
        for raw in receipt["predictions"]:
            page = pages[raw["page_id"]]
            status = (
                PredictionStatus.TRUNCATED
                if raw["tokens"] >= MAX_NEW_TOKENS
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
                        else "hf:" + REPO + "@" + REVISION
                    ),
                    regions=regions,
                    status=status,
                    purpose=PredictionPurpose.BASELINE_VALIDATION
                    if stage == 0
                    else PredictionPurpose.VALIDATION,
                )
            )
        metrics = oracle.evaluate_validation(
            tuple(predictions), PageTextEvaluatorV1(), validation_ids
        )
        if "metrics" in result and result["metrics"] != metrics:
            raise ValueError("saved stage metrics differ from current evaluator")
        if not path.exists():
            result["metrics"] = metrics
            with path.open("x") as stream:
                json.dump(result, stream, ensure_ascii=False)
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


if __name__ == "__main__":
    main()
