# Active learning for historical OCR

**Current stage:** [EXP-009](../results/EXP-009.md) and [EXP-010](../results/EXP-010.md) completed whole-page random controls; [EXP-011](../results/EXP-011.md) completed the three-seed Qwen page-selection comparison. [EXP-012](../experiments/EXP-012.md) is now running a separate Qwen **known-box line-recognition** study on L4. It crops supplied READ2016 text-line boxes and returns ordered text; it does not predict coordinates or test detection.

## Where things live

| Path | Responsibility |
| --- | --- |
| `src/active_ocr/recognition_pipeline.py` | Current common LightOnOCR/Qwen experiment runner; Modal fits/inference, local selection and scoring. |
| `src/active_ocr/acquisition_pipeline.py` | Qwen model-informed TRAIN-page scoring and selection, using the same fit/evaluation runner. |
| `src/active_ocr/region_pipeline.py` | Isolated known-box crop fit, inference and page-level acquisition on Modal L4 for EXP-012. |
| `experiments/recipes/read2016-page-text.json` | Frozen settings for the whole-page controls. |
| `experiments/recipes/read2016-known-box-regions.json` | Separate EXP-012 crop, L4 training, inference and page-budget settings. |
| `src/active_ocr/active_learning.py` | Deterministic random and scalar-uncertainty selection; the acquisition runner supplies Qwen scores. |
| `src/active_ocr/evaluation.py` | NFC page-text, corpus CER/WER counts; no box metrics. |
| `src/active_ocr/integrations/public_dataset.py` | Pinned READ2016 preparation and provenance checks. |
| `src/active_ocr/integrations/simulation.py` | Frozen source/ground-truth oracle, page-text evaluator and synthetic fixture. |
| `src/active_ocr/integrations/storage.py`, `pipeline.py`, `models.py` | Immutable source/run metadata, stored images, validation isolation, historical schema and fixture lifecycle. |
| `src/active_ocr/entrypoints/cli.py`, `examples/simulate.py` | Offline synthetic demonstration; **not** the real GPU runner. |
| `tests/` | Source/fixture/text invariants. `experiments/EXP-*.md` and `results/EXP-*.md` hold technical evidence and readable findings. |
| `docs/verification/` | Historical independent review evidence; see the commit linked in each older experiment for removed joint-OCR source. |

Local source data is in ignored `.local/assets/read2016-1.2.0/`; `.local/prepared/read2016-v1-8db5833/` holds its checked page copy and provenance, and `.local/verification/active-learning-source/runs/` holds the frozen run database. The ignored `.local/verification/` folders use older execution IDs: `exp-011` corresponds to published EXP-008, `exp-012` to EXP-009, and `exp-013` to EXP-010. Current EXP-011 raw strategy evidence is under `al-strategies/`. The published IDs are the names of the records in `experiments/` and `results/`; do not rename older local folders because their receipts reference them.

Historic EXP-001–008 joint box/text implementations and their detailed results remain in Git history and [experiment records](../experiments/). Their runtime, remote dispatcher, box matching and joint recipe have been removed from the current package. `Box` and original source `SourceRegion` still preserve unchanged dataset provenance. EXP-012 additionally uses their known text-line boxes for crops; its model predicts text only. The optional `gpu` extra retains LLaMA-Factory for future work, but neither current model uses it.

## Data → acquisition → fit → evaluation

READ2016 v1.2.0 has 350 official TRAIN and 50 official VAL pages in the frozen source snapshot `781d319efaa94b68bdd8027a2ae4e01c`. Document grouping and possible model-pretraining overlap are **unknown**; no final TEST result exists. Eight fixed VAL pages (`Seite0355`–`Seite0362`) served as the recipe check, then the other 42 VAL pages were scored separately. Original READ2016 source line transcriptions are joined in their source order into NFC whole-page text. The whole-page experiments keep source boxes out of model targets. EXP-012 supplies those boxes as crop locations, never as predicted coordinates.

For the whole-page controls, the runner verifies the frozen manifest/image identities and chooses 64 **new** TRAIN pages per round with the existing seeded random selector (`824`). Only selected TRAIN text goes into the fit payload; validation pages are sent as images and scored against truth locally after inference. At stage 0, the untouched base model predicts. Stages 1, 2 and 3 each **restart from the pinned base model** and train on cumulative 64, 128 and 192 acquired pages. Each saved adapter is reloaded in a fresh model for evaluation. The experiment stops at three rounds, never at a validation threshold. Both models use the same page set, schedule, resized image longest edge (1540), Trainer + PEFT LoRA settings and inference token budget (1536); their documented model-specific chat templates/prompts still differ. Refer to the [shared recipe](../experiments/recipes/read2016-page-text.json) for exact values.

`page-text-nfc-v1` joins ordered reference lines with newlines and scores full predicted text using corpus character error rate (CER) and word error rate (WER); lower is better. Truncated/invalid outputs are empty predictions, never silently excluded. A `Prediction` uses a whole-page placeholder region solely to reuse the frozen local evaluator; this is **not a detected box**.

The separate EXP-012 recipe, exact local/remote receipts and L4 status are in [its technical record](../experiments/EXP-012.md). Its run uses the same frozen source and page budgets but different crop inputs, one training epoch and a 64-token line output cap. Whole-page and known-box scores therefore answer different tasks; compare selection strategies within each pipeline at equal page budgets.

## Run and reproduce

The Modal image, frozen READ images and pinned Qwen weights must already exist at the volume/image IDs in the runner; LightOnOCR's pinned revision is loaded from Hugging Face. Frozen source/images must be available in the local source-run store. Install the local runner with `uv sync --locked --no-editable --extra modal --extra dev`; the GPU image already contains its own pinned Trainer/PEFT dependencies. No data, labels, credentials or large model files are tracked in Git. `--prepare-only` checks the frozen input and selected page identities without a GPU:

```sh
PYTHONPATH=src .venv/bin/python -m active_ocr.recognition_pipeline \
  .local/verification/exp-011/attempt-1/runs \
  .local/verification/exp-013/attempt-1/qwen-page-text \
  --model qwen --prepare-only
```

Remove `--prepare-only` for the baseline and three fits, or use `--through-stage 0` for a smaller first run. After all fits, pass `--holdout` to score the remaining 42 official VAL pages with the **saved** checkpoints. Change `--model lighton` and choose a distinct output directory for LightOnOCR; do not reuse another model's receipts. Running an existing result directory checks its source, config and runner hashes and replays verified stage files rather than fitting twice. Changing the runner after a run starts requires a **new output directory and experiment identity**, so the recorded hash still identifies executed code. Use the recipe and technical experiment record before running a comparison: EXP-009 originally used an earlier runner with its own SHA.

The run writes `recipe.json`, `round-0.json` … `round-3.json` and `round-N-holdout.json` under the ignored result directory. Each receipt includes source and model revisions, selected/validation IDs, metrics and raw predictions. LoRA adapters and duplicate receipts live durably in the existing Modal output Volume at `<run-name>/round-N/adapter`; verified local mirrors for EXP-009, EXP-010 and EXP-011 are under ignored `.local/models/exp-ID/`. These adapters are **not committed**. [EXP-009](../experiments/EXP-009.md), [EXP-010](../experiments/EXP-010.md) and [EXP-011](../experiments/EXP-011.md) give the specific receipt locations, adapter hashes and remote job links. Preserve failed attempts and raw labels; do not overwrite prior outputs.

The local fixture is separate and needs no GPU:

```sh
uv sync --locked --no-editable --extra dev
uv run --locked python examples/simulate.py /tmp/ocr-simulation-example
```

## Evidence and next research step

Read [reader results](../results/README.md) for interpretation and the matching `experiments/` record for exact configuration, identity and raw-evidence locations. The 42-page slice is official **validation**, not an independent document-disjoint final test; prompts/processors, model sizes and some EXP-009 hardware also differ. EXP-011 has completed all three strategy comparisons across paired seeds against matched random controls. None has a consistent advantage at both tested budgets. Avoid claiming annotation savings or strategy superiority from official validation alone.
