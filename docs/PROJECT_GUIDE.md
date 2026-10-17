# Active learning for historical OCR

**Current stage:** [EXP-009–011](../results/README.md) completed whole-page controls/selection comparisons; [EXP-012](../results/EXP-012.md) and [EXP-013](../results/EXP-013.md) completed known-box recognition/selection pilots. [EXP-015](../results/EXP-015.md) and [EXP-016](../results/EXP-016.md) completed random ICL and content controls: text examples give modest gains, mismatched pairs harm OCR, and the saved LoRA reference remains stronger. [EXP-017](../results/EXP-017.md) completed the same seven conditions with LightOnOCR: all six ICL conditions worsened raw recognition under the fixed Qwen instruction, despite passed integrity checks. [EXP-018](../results/EXP-018.md) completed fixed eight-shot contrastive Qwen ICL: raw CER/WER 0.755429/1.082243 versus paired 0.798331/1.121121 and transcript-only 0.805965/1.130093 on the same VAL42 lines; saved LoRA remains stronger. Its 16-case query-image follow-up passed execution audits but failed the advance screen, so no 64-case confirmation ran. Supplied READ2016 line boxes are cropped and transcribed in order; this is recognition at known locations, not detection.

[EXP-013](../results/EXP-013.md) completed four strategy-first known-box arms at 24/48/72 pages (configured seed 824), with all 12 adapters audited and mirrored. It reused only EXP-012’s original-model validation. Uncertainty methods led at 24 pages; random led at 48 and 72. Three sampled lines score each page; all acquired lines train and all VAL42 lines evaluate. This remains a pilot with uncontrolled pre-construction adapter initialization.

## Where things live

| Path | Responsibility |
| --- | --- |
| `src/active_ocr/recognition_pipeline.py` | Current common LightOnOCR/Qwen experiment runner; Modal fits/inference, local selection and scoring. |
| `src/active_ocr/acquisition_pipeline.py` | Qwen model-informed TRAIN-page scoring and selection, using the same fit/evaluation runner. |
| `src/active_ocr/region_pipeline.py` | Known-box fit/acquisition plus the explicit `--icl-stage prepare/check/evaluate` entrypoint. |
| `src/active_ocr/icl.py` | Random line demonstrations, content controls, checked reference reuse and resumable L4 per-query inference. |
| `src/active_ocr/icl_controls.py` | EXP-016 Qwen and EXP-017 LightOnOCR ICL/content controls: preparation, readiness, VAL8 pilot, VAL42 evaluation and local reporting via `--spec`. |
| `src/active_ocr/contrastive_icl.py` | Existing crop/sampling helpers with two shared-prefix decoding branches; frozen pilot/full recipes, checked reference/readiness reuse and resumable L4 inference. EXP-018 performs no training. |
| `src/active_ocr/query_intervention.py` | EXP-018 follow-up: exact observed-prefix query swaps, matched geometry, resumable L4 scoring, saved vocabulary logits and local screening; no training or free generation. |
| `experiments/recipes/read2016-page-text.json` | Frozen settings for the whole-page controls. |
| `experiments/recipes/read2016-known-box-regions.json` | Separate EXP-012 crop, L4 training, inference and page-budget settings. |
| `experiments/recipes/read2016-known-box-al-24.json` | EXP-013 independent strategy-first 24/48/72-page settings; other training/decoding values match EXP-012. |
| `experiments/recipes/read2016-known-box-icl.json` | EXP-014 original Qwen, reused seed-824 24-page bank, 0/8/24 random line pairs/query. |
| `experiments/recipes/read2016-known-box-icl-lighton.json` | EXP-017 same bank/crops/decoder, fresh LightOnOCR zero-shot and paired/text-only/mismatched 8/24-shot replication. |
| `src/active_ocr/active_learning.py` | Deterministic random and scalar-uncertainty selection; the acquisition runner supplies Qwen scores. |
| `src/active_ocr/evaluation.py` | NFC page-text, corpus CER/WER counts; no box metrics. |
| `src/active_ocr/integrations/public_dataset.py` | Pinned READ2016 preparation and provenance checks. |
| `src/active_ocr/integrations/simulation.py` | Frozen source/ground-truth oracle, page-text evaluator and synthetic fixture. |
| `src/active_ocr/integrations/storage.py`, `pipeline.py`, `models.py` | Immutable source/run metadata, stored images, validation isolation, historical schema and fixture lifecycle. |
| `src/active_ocr/entrypoints/cli.py`, `examples/simulate.py` | Offline synthetic demonstration; **not** the real GPU runner. |
| `tests/` | Source/fixture/text invariants. `experiments/EXP-*.md` and `results/EXP-*.md` hold technical evidence and readable findings. |
| `docs/verification/` | Historical independent review evidence; see the commit linked in each older experiment for removed joint-OCR source. |

Local source data is in ignored `.local/assets/read2016-1.2.0/`; `.local/prepared/read2016-v1-8db5833/` holds its checked page copy and provenance, and `.local/verification/active-learning-source/runs/` holds the frozen run database. Older page-text subdirectories of ignored `.local/verification/` use execution IDs: `exp-011` corresponds to published EXP-008, `exp-012` to EXP-009, and `exp-013` to EXP-010. Current EXP-011 raw strategy evidence is under `al-strategies/`. New EXP-013 evidence uses `exp-013/region-exp013-v2-<strategy>-seed824/` alongside those preserved historical subdirectories. The published IDs are the names of the records in `experiments/` and `results/`; do not rename older local folders because their receipts reference them.

Historic EXP-001–008 joint box/text implementations and their detailed results remain in Git history and [experiment records](../experiments/). Their runtime, remote dispatcher, box matching and joint recipe have been removed from the current package. `Box` and original source `SourceRegion` still preserve unchanged dataset provenance. EXP-012 additionally uses their known text-line boxes for crops; its model predicts text only. The optional `gpu` extra retains LLaMA-Factory for future work, but neither current model uses it.

## Data → acquisition → fit → evaluation

READ2016 v1.2.0 has 350 official TRAIN and 50 official VAL pages in the frozen source snapshot `781d319efaa94b68bdd8027a2ae4e01c`. Document grouping and possible model-pretraining overlap are **unknown**; no final TEST result exists. Eight fixed VAL pages (`Seite0355`–`Seite0362`) served as the recipe check, then the other 42 VAL pages were scored separately. Original READ2016 source line transcriptions are joined in their source order into NFC whole-page text. The whole-page experiments keep source boxes out of model targets. EXP-012 supplies those boxes as crop locations, never as predicted coordinates.

For the whole-page controls, the runner verifies the frozen manifest/image identities and chooses 64 **new** TRAIN pages per round with the existing seeded random selector (`824`). Only selected TRAIN text goes into the fit payload; validation pages are sent as images and scored against truth locally after inference. At stage 0, the untouched base model predicts. Stages 1, 2 and 3 each **restart from the pinned base model** and train on cumulative 64, 128 and 192 acquired pages. Each saved adapter is reloaded in a fresh model for evaluation. The experiment stops at three rounds, never at a validation threshold. Both models use the same page set, schedule, resized image longest edge (1540), Trainer + PEFT LoRA settings and inference token budget (1536); their documented model-specific chat templates/prompts still differ. Refer to the [shared recipe](../experiments/recipes/read2016-page-text.json) for exact values.

`page-text-nfc-v1` joins ordered reference lines with newlines and scores full predicted text using corpus character error rate (CER) and word error rate (WER); lower is better. Truncated/invalid outputs are empty predictions, never silently excluded. A `Prediction` uses a whole-page placeholder region solely to reuse the frozen local evaluator; this is **not a detected box**.

The completed random-only EXP-012 result and exact local/remote receipts are in [its technical record](../experiments/EXP-012.md). EXP-012 itself ran random only; EXP-013 adds the other three methods with independent first batches at smaller equal budgets. Both known-box experiments use the frozen source, crop inputs, one training epoch and a 64-token line output cap. EXP-012 adds 64 pages per round; EXP-013 adds 24. Whole-page and known-box scores therefore answer different tasks; any future selector comparison needs equal page budgets and independent seeds within the same pipeline.

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

The random ICL extension reuses the exact EXP-013 24-page bank and original/LoRA controls. It has an
explicit local preparation stage, a small eight-crop L4 check, and a full-evaluation stage gated by a
matching passed check. No fitting or retrieval is performed. See [EXP-014](../experiments/EXP-014.md)
for commands and artifact paths; the separately recorded [EXP-015](../results/EXP-015.md) now supplies full VAL42 results.

## Evidence and next research step

Read [reader results](../results/README.md) for interpretation and the matching `experiments/` record for exact configuration, identity and raw-evidence locations. [EXP-011](../results/EXP-011.md) completed three paired seeds of Qwen whole-page random, k-center, entropy and least-confidence selection at 128 and 192 labelled pages; no selector consistently beat its matched random control at both budgets. [EXP-012](../results/EXP-012.md) completed a separate known-box Qwen random baseline at 0/64/128/192 pages: VAL42 CER fell from 0.8036 to 0.0944 and WER from 1.0871 to 0.3705. EXP-012 ran random only. EXP-013 now supplies a four-method known-box pilot at 24/48/72 pages; no selector led at every budget.

These studies report official **validation**, not an independent document-disjoint final test. Document grouping and possible pretraining overlap remain unknown; the whole-page and known-box tasks also differ in inputs and recipe. Do not infer annotation savings, line detection or strategy superiority from these results.

[EXP-014](../results/EXP-014.md) is the completed development/testing experiment for original-Qwen
random ICL at 0/8/24 line examples from EXP-013's 24-page bank. Local and small L4 checks passed;
[EXP-015](../results/EXP-015.md) completed and audited both full L4 passes: eight-shot CER/WER
0.749012/1.050841, 24-shot 0.755954/1.035140. Both modestly improve on reused zero-shot
0.803646/1.087103 but remain worse than the saved 24-page LoRA control 0.410374/0.825421.
More shots gave mixed CER/WER results. No new training or production-code change was needed;
all 1,722 predictions are saved and both apps stopped. Historical initialization and validation
limitations remain recorded. [EXP-016](../experiments/EXP-016.md) completed the matched content-control study after local, L4-readiness and VAL8 gates. Transcript-only raw line CER .805965/.798095 is close to paired .798331/.808924; mismatched .968874/.948695 is worse than zero .863601, while saved LoRA24 is .430499. All 4,228 outputs are verified and all three apps stopped; no new training. [Its reader report](../results/EXP-016.md) explains the raw-line versus historical page metrics.

[EXP-017](../results/EXP-017.md) extends the existing ICL worker to pinned LightOnOCR without new dependencies or another pipeline. Raw line CER is 1.523463 at zero-shot, 3.323747 at paired 8-shot and 3.597373 at paired 24-shot; text-only and mismatched conditions also worsen recognition. Every new token/context and metric count passed independent audit. Native prompt-free training makes the retained Qwen instruction a compatibility limitation, so this does not establish optimal LightOnOCR ICL performance. Raw evidence `.local/verification/exp-017/icl-exp017-lighton-seed824/`; original model `.local/assets/lightonocr-2-1b-base-c1bc8eb/`; no new adapters. All five apps stopped. A native-format LightOnOCR study remains deferred; the current completed follow-up is EXP-018.


[EXP-018](../results/EXP-018.md) uses `experiments/recipes/read2016-known-box-icl-contrastive.json` with the existing contrastive runner. All 861 full predictions passed an independent audit, including token/cache/call provenance, image-split isolation and reused-control hashes. VAL8 and VAL42 are disjoint. All new outputs ended in EOS, and the one L4 app stopped. No new models or adapters were produced. [Research analysis and next plan](../research/exp-018-contrastive-icl-analysis.md) explain the remaining gap to LoRA and the completed 16-case query-image diagnostic. The latter did not establish its proposed improvement/harm relationship and had inadequate mask-free counterfactual coverage; no 64-case confirmation was run. Raw evidence `.local/verification/exp-018/query-intervention/pilot/`; current research is explanation of context/query effects, with a new design awaiting review.
