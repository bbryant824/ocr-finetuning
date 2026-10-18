"""One-line Qwen train/save/reload check under Slurm; not a research experiment."""

import argparse
import ast
import gc
import hashlib
import json
import math
import os
import socket
import time
from importlib.metadata import version
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from PIL import Image
    from transformers import (
        AutoProcessor,
        Qwen3VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    assert torch.cuda.is_available(), "CUDA unavailable"
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
    payload = json.loads((args.bundle / "input.json").read_text())
    train, query = payload["train"], payload["query"]
    assert ":train:" in train["page_id"] and ":validation:" in query["page_id"]
    assert "target" in train["region"] and "target" not in query["region"]
    config = payload["config"]
    helper = args.bundle / "region_pipeline.py"
    names = {"_crop", "_inputs", "_cuda"}
    tree = ast.parse(helper.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == names
    # Reuse the existing pure helpers without importing their Modal-bound module.
    namespace = {"math": math}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(helper), "exec"), namespace)
    inputs, cuda = namespace["_inputs"], namespace["_cuda"]
    for item in (train, query):
        assert sha(args.bundle / item["file"]) == item["image_sha256"]
    train_image = Image.open(args.bundle / train["file"]).convert("RGB")
    query_image = Image.open(args.bundle / query["file"]).convert("RGB")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)

    def load():
        return Qwen3VLForConditionalGeneration.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            local_files_only=True,
        ).to("cuda")

    def predict(model):
        model.eval()
        batch = cuda(inputs(processor, query_image, query["region"], config))
        with torch.inference_mode():
            tokens = model.generate(
                **batch,
                max_new_tokens=config["inference"]["max_new_tokens"],
                do_sample=False,
                repetition_penalty=config["inference"]["repetition_penalty"],
            )[0, batch["input_ids"].shape[1] :].tolist()
        assert tokens
        return {"tokens": tokens, "text": processor.decode(tokens, skip_special_tokens=True)}

    model = load()
    baseline = predict(model)
    set_seed(824)  # Before constructing random LoRA parameters.
    fit = config["training"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=fit["lora_rank"],
            lora_alpha=fit["lora_alpha"],
            lora_dropout=0,
            target_modules=fit["lora_targets"],
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    batch = inputs(processor, train_image, train["region"], config, train=True)
    supervised = int((batch["labels"] != -100).sum())
    assert 0 < supervised <= config["inference"]["max_new_tokens"]
    trainer = Trainer(
        model=model,
        train_dataset=[batch],
        data_collator=lambda samples: samples[0],
        args=TrainingArguments(
            output_dir=str(args.output / "trainer"),
            max_steps=1,
            per_device_train_batch_size=1,
            learning_rate=fit["learning_rate"],
            bf16=True,
            seed=824,
            data_seed=824,
            report_to=[],
            remove_unused_columns=False,
            save_strategy="no",
            logging_steps=1,
            dataloader_pin_memory=False,
            lr_scheduler_type="constant",
            optim="adamw_torch",
        ),
    )
    result = trainer.train()
    assert math.isfinite(result.training_loss) and result.global_step == 1
    assert any(
        "lora_B" in name and torch.count_nonzero(parameter).item()
        for name, parameter in model.named_parameters()
    ), "LoRA did not update"
    trained = predict(model)
    adapter = args.output / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    del trainer, model, batch
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = PeftModel.from_pretrained(load(), adapter).eval()
    restored = predict(reloaded)
    assert trained == restored, "Adapter reload changed prediction"
    receipt = {
        "purpose": "infrastructure check, not a research experiment",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(),
        "gpu_count": torch.cuda.device_count(),
        "model": payload["model"],
        "model_path": str(args.model),
        "smoke_overrides": {
            "max_steps": 1,
            "gradient_accumulation_steps": 1,
            "lr_scheduler_type": "constant",
            "training_sample_count": 1,
        },
        "input_sha256": sha(args.bundle / "input.json"),
        "helper_sha256": sha(helper),
        "script_sha256": sha(Path(__file__)),
        "train_page": train["page_id"],
        "query_page": query["page_id"],
        "train_region": train["region"]["id"],
        "query_region": query["region"]["id"],
        "supervised_tokens": supervised,
        "optimizer_steps": result.global_step,
        "training_loss": result.training_loss,
        "baseline": baseline,
        "trained": trained,
        "reloaded": restored,
        "reload_identical": True,
        "elapsed_seconds": time.monotonic() - started,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "packages": {
            p: version(p) for p in ["torch", "transformers", "peft", "accelerate", "pillow"]
        },
        "adapter_sha256": {p.name: sha(p) for p in adapter.iterdir() if p.is_file()},
        "model_sha256": {p.name: sha(p) for p in args.model.glob("*.safetensors")},
        "processor_sha256": {
            p.name: sha(p)
            for p in args.model.iterdir()
            if p.suffix in {".json", ".txt"} and p.is_file()
        },
    }
    (args.output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"reload_identical": True, "loss": result.training_loss}))


if __name__ == "__main__":
    main()
