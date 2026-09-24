# Active learning for historical page-text OCR

**Current stage: two completed random-selection controls and a Qwen acquisition-strategy study.** [EXP-009](../results/EXP-009.md) evaluates LightOnOCR and [EXP-010](../results/EXP-010.md) evaluates Qwen3-VL on READ2016. [EXP-011](../results/EXP-011.md) has completed three-seed visual-diversity comparison; entropy and least-confidence repeat seeds are in progress. The research task is **whole-page transcription**, not line localization.

## Where things live

| Path | Responsibility |
| --- | --- |
| `src/active_ocr/recognition_pipeline.py` | Current common LightOnOCR/Qwen experiment runner; Modal fits/inference, local selection and scoring. |
| `src/active_ocr/acquisition_pipeline.py` | Qwen model-informed TRAIN-page scoring and selection, using the same fit/evaluation runner. |
| `experiments/recipes/read2016-page-text.json` | Shared frozen source, split, selection schedule, model revisions, prompts, LoRA, inference settings. |
| `src/active_ocr/active_learning.py` | Deterministic random and scalar-uncertainty selection; the acquisition runner supplies Qwen scores. |
| `src/active_ocr/evaluation.py` | NFC page-text, corpus CER/WER counts; no box metrics. |
| `src/active_ocr/integrations/public_dataset.py` | Pinned READ2016 preparation and provenance checks. |
| `src/active_ocr/integrations/simulation.py` | Frozen source/ground-truth oracle, page-text evaluator and synthetic fixture. |
| `src/active_ocr/integrations/storage.py`, `pipeline.py`, `models.py` | Immutable source/run metadata, stored images, validation isolation, historical schema and fixture lifecycle. |
| `src/active_ocr/entrypoints/cli.py`, `examples/simulate.py` | Offline synthetic demonstration; **not** the real GPU runner. |
| `tests/` | Source/fixture/text invariants. `experiments/EXP-*.md` and `results/EXP-*.md` hold technical evidence and readable findings. |
| `docs/verification/` | Historical independent review evidence; see the commit linked in each older experiment for removed joint-OCR source. |

Local source data is in ignored `.local/assets/read2016-1.2.0/`; `.local/prepared/read2016-v1-8db5833/` holds its checked page copy and provenance, and `.local/verification/active-learning-source/runs/` holds the frozen run database. The ignored `.local/verification/` folders use older execution IDs: `exp-011` corresponds to published EXP-008, `exp-012` to EXP-009, and `exp-013` to EXP-010. Current EXP-011 raw strategy evidence is under `al-strategies/`. The published IDs are the names of the records in `experiments/` and `results/`; do not rename older local folders because their receipts reference them.

Historic EXP-001–008 joint box/text implementations and their detailed results remain in Git history and [experiment records](../experiments/). Their runtime, remote dispatcher, box matching and joint recipe have been removed from the current package. `Box` and original source `SourceRegion` remain to validate and read the **unchanged dataset provenance**; OCR predictions and scoring are page text only. The optional `gpu` extra retains LLaMA-Factory for future work, but neither current model uses it.

## Data → acquisition → fit → evaluation

READ2016 v1.2.0 has 350 official TRAIN and 50 official VAL pages in the frozen source snapshot `781d319efaa94b68bdd8027a2ae4e01c`. Document grouping and possible model-pretraining overlap are **unknown**; no final TEST result exists. Eight fixed VAL pages (`Seite0355`–`Seite0362`) served as the recipe check, then the other 42 VAL pages were scored separately. Original READ2016 source line transcriptions are joined in their source order into NFC whole-page text. Source boxes stay in the original labels, not model targets.

The runner verifies the frozen manifest/image identities and chooses 64 **new** TRAIN pages per round with the existing seeded random selector (`824`). Only selected TRAIN text goes into the fit payload; validation pages are sent as images and scored against truth locally after inference. At stage 0, the untouched base model predicts. Stages 1, 2 and 3 each **restart from the pinned base model** and train on cumulative 64, 128 and 192 acquired pages. Each saved adapter is reloaded in a fresh model for evaluation. The experiment stops at three rounds, never at a validation threshold. Both models use the same page set, schedule, resized image longest edge (1540), Trainer + PEFT LoRA settings and inference token budget (1536); their documented model-specific chat templates/prompts still differ. Refer to the [shared recipe](../experiments/recipes/read2016-page-text.json) for exact values.

`page-text-nfc-v1` joins ordered reference lines with newlines and scores full predicted text using corpus character error rate (CER) and word error rate (WER); lower is better. Truncated/invalid outputs are empty predictions, never silently excluded. A `Prediction` uses a whole-page placeholder region solely to reuse the frozen local evaluator; this is **not a detected box**.

## Run and reproduce

The Modal image, frozen READ images and pinned Qwen weights must already exist at the volume/image IDs in the runner; LightOnOCR's pinned revision is loaded from Hugging Face. Frozen source/images must be available in the local source-run store. Install the local runner with `uv sync --locked --no-editable --extra modal --extra dev`; the GPU image already contains its own pinned Trainer/PEFT dependencies. No data, labels, credentials or large model files are tracked in Git. `--prepare-only` checks the frozen input and selected page identities without a GPU:

```sh
PYTHONPATH=src .venv/bin/python -m active_ocr.recognition_pipeline \
  .local/verification/exp-011/attempt-1/runs \
  .local/verification/exp-013/attempt-1/qwen-page-text \
  --model qwen --prepare-only
```

Remove `--prepare-only` for the baseline and three fits, or use `--through-stage 0` for a smaller first run. After all fits, pass `--holdout` to score the remaining 42 official VAL pages with the **saved** checkpoints. Change `--model lighton` and choose a distinct output directory for LightOnOCR; do not reuse another model's receipts. Running an existing result directory checks its source, config and runner hashes and replays verified stage files rather than fitting twice. Changing the runner after a run starts requires a **new output directory and experiment identity**, so the recorded hash still identifies executed code. Use the recipe and technical experiment record before running a comparison: EXP-009 originally used an earlier runner with its own SHA.

The run writes `recipe.json`, `round-0.json` … `round-3.json` and `round-N-holdout.json` under the ignored result directory. Each receipt includes source and model revisions, selected/validation IDs, metrics and raw predictions. The three LoRA adapters and duplicate receipts live durably in the existing Modal output Volume at `<run-name>/round-N/adapter`; small verified local mirrors are under ignored `.local/models/`. These adapters are **not committed**. [EXP-009](../experiments/EXP-009.md) and [EXP-010](../experiments/EXP-010.md) give the specific receipt locations, adapter hashes and remote job links. Preserve failed attempts and raw labels; do not overwrite prior outputs.

The local fixture is separate and needs no GPU:

```sh
uv sync --locked --no-editable --extra dev
uv run --locked python examples/simulate.py /tmp/ocr-simulation-example
```

## Evidence and next research step

Read [reader results](../results/README.md) for interpretation and the matching `experiments/` record for exact configuration, identity and raw-evidence locations. The 42-page slice is official **validation**, not an independent document-disjoint final test; prompts/processors, model sizes and some EXP-009 hardware also differ. The first model-informed acquisition pilot is recorded in EXP-011, but its single-seed result is mixed. Paired repeat seeds and uncertainty strategies are in progress. Avoid claiming annotation savings or strategy superiority until those comparisons are complete.
