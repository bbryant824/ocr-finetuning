# Historical OCR with VLMs — project guide

**Current stage, 18 October 2026:** engineering, fine-tuning/AL baselines and ICL controls are
recorded through EXP-017; contrastive recognition and diagnostics through EXP-021 are completed.
The current research objective is to improve inference-time OCR using useful visual/textual
context, with an explanation that can be tested beyond READ2016. [EXP-021 reader](../results/EXP-021.md)
and [technical record](../experiments/EXP-021.md) are the latest full run. The proposed token-choice
continuation study remains unapproved. Supplied line boxes mean known-location recognition,
not detection. See the [complete experiment index](../results/README.md) and
[current research index](../research/README.md); detailed numbers belong in their source records.

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
| `src/active_ocr/contrastive_diagnostic.py` | EXP-018 correct-prefix diagnostic; saved next-token scores and evaluator-only truth. |
| `src/active_ocr/contrastive_components.py` | EXP-019 controls the text-choice versus stopping rule under generated histories. |
| `src/active_ocr/contrastive_references.py` | EXP-020–021 reference/shot comparisons, exact cached query noise, checked baseline reuse and resumable L4 chunks. |
| `experiments/recipes/read2016-known-box-query-contrast-full.json` | Frozen EXP-021 0/2/8-shot VAL42 settings; other completed recipes remain in `experiments/recipes/`. |
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
| `docs/verification/` | Infrastructure verification and historical independent review evidence; see the commit linked in each older experiment for removed joint-OCR source. |

Local source data is in ignored `.local/assets/read2016-1.2.0/`; `.local/prepared/read2016-v1-8db5833/` holds its checked page copy and provenance, and `.local/verification/active-learning-source/runs/` holds the frozen run database. Older page-text subdirectories of ignored `.local/verification/` use execution IDs: `exp-011` corresponds to published EXP-008, `exp-012` to EXP-009, and `exp-013` to EXP-010. Current EXP-011 raw strategy evidence is under `al-strategies/`. New EXP-013 evidence uses `exp-013/region-exp013-v2-<strategy>-seed824/` alongside those preserved historical subdirectories. The published IDs are the names of the records in `experiments/` and `results/`; do not rename older local folders because their receipts reference them.

Historic EXP-001–008 joint box/text implementations and their detailed results remain in Git history and [experiment records](../experiments/). Their runtime, remote dispatcher, box matching and joint recipe have been removed from the current package. `Box` and original source `SourceRegion` still preserve unchanged dataset provenance. EXP-012 additionally uses their known text-line boxes for crops; its model predicts text only. The optional `gpu` extra retains LLaMA-Factory for future work, but neither current model uses it.

## Data → acquisition → fit → evaluation

READ2016 v1.2.0 has 350 official TRAIN and 50 official VAL pages in the frozen source snapshot `781d319efaa94b68bdd8027a2ae4e01c`. Document grouping and possible model-pretraining overlap are **unknown**; no final TEST result exists. Eight fixed VAL pages (`Seite0355`–`Seite0362`) served as the recipe check, then the other 42 VAL pages were scored separately. Original READ2016 source line transcriptions are joined in their source order into NFC whole-page text. The whole-page experiments keep source boxes out of model targets. EXP-012 supplies those boxes as crop locations, never as predicted coordinates.

For the whole-page controls, the runner verifies the frozen manifest/image identities and chooses 64 **new** TRAIN pages per round with the existing seeded random selector (`824`). Only selected TRAIN text goes into the fit payload; validation pages are sent as images and scored against truth locally after inference. At stage 0, the untouched base model predicts. Stages 1, 2 and 3 each **restart from the pinned base model** and train on cumulative 64, 128 and 192 acquired pages. Each saved adapter is reloaded in a fresh model for evaluation. The experiment stops at three rounds, never at a validation threshold. Both models use the same page set, schedule, resized image longest edge (1540), Trainer + PEFT LoRA settings and inference token budget (1536); their documented model-specific chat templates/prompts still differ. Refer to the [shared recipe](../experiments/recipes/read2016-page-text.json) for exact values.

`page-text-nfc-v1` joins ordered reference lines with newlines and scores full predicted text using corpus character error rate (CER) and word error rate (WER); lower is better. Truncated/invalid outputs are empty predictions, never silently excluded. A `Prediction` uses a whole-page placeholder region solely to reuse the frozen local evaluator; this is **not a detected box**.

The completed random-only EXP-012 result and exact local/remote receipts are in [its technical record](../experiments/EXP-012.md). EXP-012 itself ran random only; EXP-013 adds the other three methods with independent first batches at smaller equal budgets. Both known-box experiments use the frozen source, crop inputs, one training epoch and a 64-token line output cap. EXP-012 adds 64 pages per round; EXP-013 adds 24. Whole-page and known-box scores therefore answer different tasks; any future selector comparison needs equal page budgets and independent seeds within the same pipeline.

## Compute backend

NUS SoC is now the primary backend for future training/inference, with the fastest suitable
available resource preferred. [The cluster skill](../.agents/skills/nus-soc-cluster/SKILL.md) links
the [full credential-free guide](cluster/NUS_SOC_CLUSTER_GUIDE.md), live scheduling checks and
[workflow verification](verification/nus-soc-cluster.md). The complete original guide stays ignored
and private under `.local/private/`. Use normal SSH and Slurm in authorized discrete sessions;
respect institutional rules and do not create an external scheduler or job-management service.

The existing research entrypoints still contain Modal-specific decorators, images and Volumes.
A verified standalone cluster check establishes that workload only; a later approved experiment
must reuse its pure helpers and explicitly provide Slurm execution, inputs and durable receipts.
Historical Modal recipes/results remain reproducible; Modal is an explicit fallback.
Each cluster run also has a concise [command/session learning record](cluster/sessions/README.md),
linked from its technical/check report; [the first workflow check](cluster/sessions/2026-10-18-workflow-check.md)
shows the login-to-retrieval sequence and its actual Slurm failure/fix.

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

## Evidence and current research direction

The [results index](../results/README.md) covers EXP-001–021 and links both reader explanations and
technical records. Earlier failed runs remain evidence, not missing or silently successful studies.

| Stage | Findings and records |
|---|---|
| EXP-001–008 | Joint box/text engineering worked in later attempts but OCR quality gates failed. [Grouped reports and all technical records](../results/README.md). Historical implementation is in Git. |
| EXP-009–011 | Full-page LightOn/Qwen controls and three-seed strategy comparisons. No selector consistently outperformed matched random controls. [EXP-011 reader](../results/EXP-011.md), [technical](../experiments/EXP-011.md). |
| EXP-012–013 | Known-box Qwen fitting/AL at 64-page and then 24-page increments. EXP-013 has 12 verified adapters; no selector leads every budget. [EXP-013 reader](../results/EXP-013.md), [technical](../experiments/EXP-013.md). |
| EXP-014–017 | ICL testing, Qwen paired/text-only/mismatched controls and LightOn replication. Text context gives modest Qwen gains; mismatches harm; fixed Qwen-format LightOn ICL worsens. [EXP-016 reader](../results/EXP-016.md), [technical](../experiments/EXP-016.md); [EXP-017](../results/EXP-017.md). |
| EXP-018 | Mismatch contrast improves generated-output CER, but clean-prefix and query-image diagnostics do not establish better glyph recognition. [Reader](../results/EXP-018.md), [technical](../experiments/EXP-018.md). |
| EXP-019 | Contrast for text retains 140/144 saved pilot edits; stopping alone is insufficient. [Reader](../results/EXP-019.md), [technical](../experiments/EXP-019.md). |
| EXP-020 | Fixed reference and 0/2/8-shot pilot: query contrast helps without examples; examples add benefit; long-output sensitivity matters. [Reader](../results/EXP-020.md), [technical](../experiments/EXP-020.md). |
| EXP-021 | VAL42 confirms lower error with contrast and an additional two-shot gain. Query-noise and earlier mismatch references nearly tie; no generally superior reference is established. [Reader](../results/EXP-021.md), [technical](../experiments/EXP-021.md). |

EXP-021 uses original Qwen3-VL-4B-Instruct, the same seed-824 TRAIN24/589-line bank and 861 VAL42
queries. Ordinary→query-contrast **raw line CER** at 0/2/8 shots is 86.3601→80.9515 /
81.0403→76.4365 / 79.8331→75.4956%. Baselines zero/eight are verified reuse, not newly executed.
Two supports save 763 more character edits than zero-shot contrast; eight save another 159.
Contrast mainly reduces insertions while added supports reduce substitutions under contrast;
these are alignment-dependent error patterns, not a visual/language causal decomposition.
All 3,444 new outputs and 5,166 condition-line metric comparisons passed independent audit;
the recorded two EXP-021 app states are stopped. No model/adapter was trained in EXP-014–021.

Do not mix historical page scores (cap-masked, joined lines) with raw NFC per-line corpus scores.
The raw-line metric retains every generated string, including length-limited output. In this
metric the saved LoRA24 reference remains much stronger than ICL/contrast. Official VAL8 and
VAL42 are page-disjoint but previously examined validation; neither is an untouched final test.
Document grouping and pretraining overlap are unknown. No annotation-saving, detection,
independent generalization or repeatable strategy advantage follows from these pilots.

**Next proposal only:** at the first ordinary/contrast token disagreement, force either observed
choice and finish using a fixed ordinary or contrastive decoder. Score complete OCR outcomes,
not image-sensitivity proxies. Existing EXP-021 first eight VAL42 pages provide 105 queries,
92 divergent sequences, exact query noise and two existing endpoints; at most 184 new continuations
would be needed. The purpose is to test early path steering versus continued correction before
proposing a further decoding rule. This is unapproved and unlaunched. EXP-016's modality controls
are already completed; changing only their shot count is not the research priority. Base/chat,
blank-image fitting and broader matrices remain deferred. See [current synthesis](../research/exp-018-contrastive-icl-analysis.md).

Current raw inference evidence is `.local/verification/exp-021/query-contrast-seed824/`, with remote
originals `/exp021-query-contrast-469abd341bce/` on the existing output Volume. Source/prepared/frozen
data locations above are unchanged; original Qwen weights are `.local/assets/qwen3-vl-4b-ebb281e/`,
and LightOn weights `.local/assets/lightonocr-2-1b-base-c1bc8eb/`. Earlier trained artifacts are
verified ignored mirrors under `.local/models/exp-ID/`; exact paths/hashes are in each technical
record. Preserve historical verification-folder numbering and frozen execution copies.

[AGENTS.md](../AGENTS.md) requires Superpowers for complex work and synchronized PROJECT,
current instructions, indexes and relevant guide sections at every experiment/research handoff.
Local PROJECT/TEAM are ignored; this guide and the linked records preserve public progress.
Local changes are not GitHub updates until explicitly committed/pushed; see PROJECT for current
checkout/evidence state and approval limits. No cloud execution is implied by documentation.
