# OCR active-learning project — engineering report and guide

**Status: 19 September 2026. LLaMA-Factory migration complete; independent pipeline review passed.**
The real pipeline completed initial training on 16 pages, acquisition of 8 more pages, reset-fit
on all 24, fresh-process checkpoint reloads, validation, commit, completed-run resume and export.
The run is [EXP-005](../experiments/EXP-005.md), source
`11a58ad9ae7e7d61fc45ac05b4dfba09442b710b`. **OCR readiness failed:** only one of eight final
outputs was valid; CER was 0.999512, WER 1.0 and localization F1 zero. A working pipeline does
not establish useful OCR or an active-learning advantage.

This is the maintained document for the repository, component connections and operation.
LLaMA-Factory v0.9.5 owns CLI training and native HuggingFace `ChatModel` inference. Our code
retains selection, selected-only source-label reveal, joint evaluation and durable run accounting.
The parent local suite passed 677 cases without skips; focused corrections received independent
review, and five actual toolkit checks passed in the final Linux image.

Historical [EXP-003](../experiments/EXP-003.md) records the earlier custom-runtime round.
[EXP-004](../experiments/EXP-004.md) preserves the incomplete short-window migration attempt.
Old runs retain their exact source, checkpoints and original deadlines. Dated research and
verification/run records support this guide rather than compete with its current status.

## Contents

1. [Purpose and current stage](#1-purpose-and-current-stage)
2. [Complete repository map](#2-complete-repository-map)
3. [How the components connect](#3-how-the-components-connect)
4. [Data preparation and label isolation](#4-data-preparation-and-label-isolation)
5. [Active-learning lifecycle](#5-active-learning-lifecycle)
6. [Model runtime: LLaMA-Factory](#6-model-runtime-llama-factory)
7. [Modal execution, persistence and recovery](#7-modal-execution-persistence-and-recovery)
8. [Setup and operating procedure](#8-setup-and-operating-procedure)
9. [The completed real round](#9-the-completed-real-round)
10. [Engineering verification and resolved problems](#10-engineering-verification-and-resolved-problems)
11. [Research progress and next steps](#11-research-progress-and-next-steps)
12. [Team responsibilities and documentation policy](#12-team-responsibilities-and-documentation-policy)

## 1. Purpose and current stage

The research question is whether active learning can reduce the annotation required to adapt
OCR to historical and low-resource documents. A **page image is the acquisition unit**; its
labels are ordered line boxes and transcriptions. Existing dataset labels simulate a human
annotator: only labels for selected training pages are revealed to training. There is no need
to manually label this corpus or build a labeling frontend for the present study.

| Stage | Delivered or remaining |
| --- | --- |
| Stage 1: local foundation | Image import, shared models, selectors, metrics, SQLite/artifacts, CLI, deterministic simulation, source-label oracle and isolated evaluator. |
| Stage 2: connected real execution — complete | Pinned READ2016 conversion, verified assets, real Qwen LoRA runtime, Modal dispatch/recovery, initial baseline, one actual acquired training round, fresh-process checkpoint reload, exports and independent verification. |
| Stage 3: useful OCR and research experiments — next | Calibrate output format and decode capacity; establish an adequate initial adaptation recipe; define grouping/evaluation and real uncertainty; compare equal budgets across seeds, then datasets/models. |

The implemented real path currently supports **random acquisition only**. Least-confidence
and entropy selectors exist and work with fixture/test scores, but the model does not yet produce
reviewed acquisition scores. The two-page real run is an engineering smoke test, not an
uncertainty-strategy comparison or a sufficient initial fine-tuning campaign.

## 2. Complete repository map

The following lists the source, configuration, tests and maintained documentation. Generated
environments, raw data and Git internals are deliberately outside the tracked tree.

```text
ocr-finetuning/
├── README.md                         short entry point to this guide
├── AGENTS.md                         research/engineering rules and role ownership
├── pyproject.toml                    package, CLI scripts, extras and lint/test settings
├── uv.lock                           resolved dependency lock
├── .python-version                   local Python selection
├── .gitignore                        excludes secrets, data, runs and private coordination
├── examples/
│   └── simulate.py                   no-service fixture: generate → run → resume → export
├── src/active_ocr/
│   ├── __init__.py
│   ├── models.py                     immutable source/run/prediction contracts
│   ├── active_learning.py            deterministic random/confidence/entropy ranking
│   ├── evaluation.py                 text counts, CER/WER, IoU and curve helpers
│   ├── pipeline.py                   one orchestrator, including simulation/real lifecycle
│   ├── integrations/
│   │   ├── __init__.py               explicit imports; local tools avoid loading ML packages
│   │   ├── storage.py                SQLite records, atomic updates, hashed artifacts
│   │   ├── simulation.py             normalized source, oracle, fixture, evaluators, identity
│   │   ├── public_dataset.py         pinned READ2016 conversion and provenance verification
│   │   ├── artifacts.py              shared file inventory, hashes and environment provenance
│   │   ├── factory_model.py          LLaMA-Factory adapter: data, train, predict, checkpoint
│   │   └── modal_model.py            remote contract, coordinator, operation journal/recovery
│   └── entrypoints/
│       ├── __init__.py
│       ├── cli.py                    simulation and real create/run/status/resume/export
│       └── modal_app.py              CPU build gate, single dispatcher, isolated model children
├── tests/
│   ├── test_core.py                  shared models/selection fundamentals
│   ├── test_simulation.py            fixture lifecycle, budgets, freeze and persistence
│   ├── test_real_contract.py         baseline/real-shaped adapter contracts
│   ├── test_contract_independent.py  independent local contract checks
│   ├── test_evaluation_counts.py     text normalization, errors and undefined rates
│   ├── test_public_dataset.py        READ conversion and provenance
│   ├── test_read2016_independent.py  independent READ source checks
│   ├── test_factory_model.py         recipe, dataset rows, checkpoints and actual template cases
│   ├── test_modal_model.py           transport, identities, journal and artifact validation
│   ├── test_modal_app.py             build and worker contracts
│   ├── test_modal_independent.py     independent worker/transport/recovery checks
│   └── test_real_cli.py              real CLI wiring and export
├── docs/
│   ├── PROJECT_GUIDE.md              this maintained project document
│   └── verification/                dated evidence; PASS applies to the stated SHA/scope
│       ├── local-contract-review.md
│       ├── read2016-converter-review.md
│       ├── qwen-runtime-review.md
│       ├── modal-runtime-review.md
│       ├── modal-cpu-runtime.md      actual CPU attempt history, including failures
│       └── modal-cpu-review.md
├── experiments/
│   ├── README.md / TEMPLATE.md       small reproducibility record convention
│   ├── EXP-001.md / EXP-002.md       preserved earlier unsuccessful execution attempts
│   ├── EXP-003.md                    accepted completed real round
│   ├── assets/read2016-qwen3vl4b.json published source/file identities and staging evidence
│   └── recipes/read2016-llamafactory-joint.json  the one validated execution recipe
├── research/
│   ├── README.md / TEMPLATE.md
│   └── RES-002-real-ocr-pilot.md      dated source-backed dataset/model recommendation
├── plans/
│   └── README.md / TEMPLATE.md       future approved plans; completed plan is in Git history
├── coordination/
│   └── README.md / TASK_TEMPLATE.md  reusable team protocol, not application runtime state
├── .agents/skills/
│   ├── manager/SKILL.md
│   ├── research/SKILL.md
│   ├── planning/SKILL.md
│   ├── development/SKILL.md
│   ├── testing/SKILL.md
│   ├── experiment-platform/SKILL.md
│   └── qa-understanding/SKILL.md
└── scripts/
    ├── validate_agent_system.py      portable/local coordination integrity checks
    └── test_agent_system.py          protocol checker tests
```

Ignored local layout: `.venv/` holds local dependencies; `.local/assets/` downloads;
`.local/prepared/` normalized data; `.local/execution/` frozen execution checkouts;
`.local/verification/` configuration, receipts, runs and exports; `.local/qa-understanding/`
presentation drafts/visuals; `.agent-local/` shared private task/decision/ops records;
`var/` retained historical state; `.env` secrets, if any. These are not included in a fresh clone.
Cloud input/output Volumes are external artifacts, not repository folders.

## 3. How the components connect

```mermaid
flowchart TD
    A[Page images and original source labels] --> B[public_dataset: convert and verify provenance]
    B --> C[simulation: freeze source and LocalOracle]
    C --> D[pipeline: initial fit and acquisition rounds]
    D --> E[active_learning: choose TRAIN page IDs]
    E --> F[LocalOracle: reveal selected TRAIN labels]
    F --> D
    D --> G[modal_model: checked request and durable journal]
    G --> H[modal_app: one sequential GPU dispatcher]
    H --> I[factory_model: LLaMA-Factory LoRA fit or ChatModel prediction]
    I --> J[Immutable checkpoints and output receipts]
    J --> G
    G --> D
    D --> K[Validation predictions]
    C -->|validation truth only| L[Local evaluator: text errors and box metrics]
    K --> L
    L --> D
    D --> M[storage: pending selection and committed stages]
    M --> N[CLI: status, resume, JSON and CSV export]
```

| Component | Receives | Produces / connection |
| --- | --- | --- |
| `public_dataset.convert_read2016` | Pinned archive and safely extracted originals | Preserved `source/`, `provenance.json`, normalized `pages.jsonl`. It neither downloads nor trains. |
| `LocalOracle.freeze` / `LocalOracle` | Source manifest, images and provenance | Frozen `DatasetSnapshot`; selected-only `RevealedExample`s; isolated validation truth. |
| `Pipeline` | Frozen config, oracle, model adapter and store | Initial-fit/acquisition transitions, durable label accounting, validation, budgets and exports. No model-specific training logic. |
| `select_pages` | Available TRAIN IDs, seed and permitted pool scores | Unique selected IDs; deterministic ordering/tie handling. No labels or validation metrics. |
| `ModalModel` | Run context and explicit load/fit/predict request | Verified remote checkpoint/predictions; operation journal and known-call reconciliation. |
| `modal_app.dispatch` | Label-free metadata except selected fit examples | Fresh adapter child execution, hashed evidence and completion receipt. |
| `FactoryModel` | Pinned files, verified images, selected training targets when fitting | Upstream dataset rows and YAML, `llamafactory-cli train`, native adapter checkpoint, `ChatModel` predictions and reload diagnostics. |
| `PageTextEvaluatorV1` | Validation predictions and validation truth locally | Aggregate text edit counts, CER/WER and explicit failed-page counts. |
| `PageJointEvaluatorV1` | The same inputs | Those unchanged text counts plus separately versioned IoU-0.5 localization counts. |
| `SQLiteStore` | Validated complete state and artifact bytes | Durable JSON records and content-addressed files; compare-and-swap commit. |

One orchestrator and direct adapters keep the design shallow. There is no separate queue,
workflow service or application database for agent coordination. Fixture, contract-test and
real runs share orchestration but have explicit different run kinds; test outputs cannot silently
become real OCR evidence.

## 4. Data preparation and label isolation

### Actual dataset and source mapping

The staged corpus is **READ2016 / Bozen 1.2.0**, publisher
[Zenodo record 1297399](https://zenodo.org/records/1297399), CC BY 4.0. The retained asset
record contains exact download URLs, sizes, hashes and license observations.

| Source property | Verified value |
| --- | --- |
| Archive | `Train-And-Val-ICFHR-2016.tgz`, 493,223,531 bytes |
| Archive SHA-256 | `f4748c58af757e06804e638a6e84c2150daab19f50d55e10aac3115e6bfc1756` |
| Official membership | 350 TRAIN pages / 50 VALIDATION pages; no final-test archive downloaded |
| Source lines | 8,367 TRAIN + 1,043 VALIDATION = 9,410 |
| Preserved regular files | 804 originals; 400 known image symlinks verified and omitted |
| Source edge cases | Four empty strings and two missing baselines retained |
| Document grouping | Unknown; source `docId=-1` is not a usable document identity |

The converter uses the explicit policy `read2016-official-unknown-engineering-v1`.
Official membership is unchanged and document IDs remain null. This is an approved engineering
exception, not evidence of document-disjoint evaluation. The ordinary `known-document-v1`
policy requires known document IDs and rejects cross-split documents/images. Exact duplicate
image checks do not establish the absence of near duplicates or pretraining overlap.

PAGE XML supplies line order and literal Unicode text. Every original XML/image file is kept
byte-for-byte. The model-facing box is the pixel envelope of each source line polygon;
original polygons and partial unclear/style metadata stay in XML. No trimming, abbreviation
expansion, invented illegibility strings, label repair, crop or split changes occur. Input images
are verified for dimensions, full decoding, RGB and identity orientation for this pinned source.
Generic page-image manifests support common image files; this specific converter expects the
publisher's JPG/XML pairing.

Normalized rows contain `id`, required nullable `document_id`, `image_uri`, `width`, `height`,
`split`, `image_sha256`, ordered `regions`, source policy and provenance reference where required.
Each region has `id`, original-pixel `box`, literal `text` and `illegible`. Missing labels are
different from `regions: []`, which represents a labeled blank page. Invalid geometry, hashes,
duplicate identities, malformed order and unsupported source structures are rejected.

### What leaves the oracle

| Destination | Allowed data |
| --- | --- |
| Acquisition selector | TRAIN membership and eligible model scores; never source text or evaluation results. |
| Fit | Images and true line targets for the run's cumulative selected TRAIN pages only. |
| Prediction | Image ID, split, dimensions, checksum and image address; no XML, regions or truth. |
| Validation evaluator | Only the fixed validation subset's truth and its corresponding predictions. |
| Modal input Volume | Pinned model/processor files and content-addressed images; no oracle, XML, source manifest, DB or label dataset. |
| Final test | Not used by this completed run; final evaluation is deferred. |

Selected fit targets travel through the remote request and the first child's stdin. They are
never printed and never enter the operation journal, which keeps only a target digest. The
toolkit reads a real file, so `fit` does write a `train.jsonl` containing exactly those revealed
targets into the operation's own output directory on the output Volume. That directory is not
part of the artifact inventory returned to the coordinator, and no revealed label reaches a
prediction request, the input Volume or Git. The second reload child receives image/checkpoint
metadata, not labels. These interfaces prevent accidental leakage; they are not a security
sandbox against malicious authorized local code.

Freeze records manifest/ground-truth/image/provenance hashes, membership, seed, configuration,
model/processor revisions, code and dependency identities. Resume rechecks frozen inputs and
refuses changed sources. Original/prepared files must remain available; this is not an automatic
dataset migration or standalone cloud backup.

## 5. Active-learning lifecycle

1. **Create.** Validate and freeze inputs/configuration in a new UUID run. Creation itself does
   not load a model or submit GPU work.
2. **Zero-label baseline.** For a real/contract run, load the pinned base, predict the fixed
   validation images, evaluate locally and commit a separate round-0 baseline. No training
   labels are revealed. The fixture path has no baseline unless a separate contract is used.
3. **Initial fit (optional, `initial_batch_size > 0`).** Select a seeded random TRAIN seed set,
   reveal its labels, fit it and evaluate. This is a separate persisted `initial_fit` record, not
   the zero-label baseline: `SimulationBaseline.labelled_count` stays literally zero. Its pages
   count toward `page_budget` but not toward `max_rounds`.
4. **Select.** Choose an available TRAIN batch within remaining page budget. The first batch
   is seeded random. Subsequent supported fixture strategies use the prior round's pool scores.
5. **Simulate annotation.** The oracle reveals the selected pages' existing line boxes/text.
   Previously selected IDs, including the initial seed set, stay in the cumulative labeled set.
6. **Reset-fit.** Begin from the pinned base and train a fresh adapter on all labels revealed
   so far. This is cumulative training data, not warm-starting the previous round's adapter.
7. **Predict and evaluate.** Fixture uncertainty strategies score the remaining TRAIN pool.
   Random skips pool scoring unless explicitly requested in the fixture. The current real
   recipe requires random with no pool reporting; it always evaluates its fixed validation subset.
8. **Commit.** After source/identity/result validation, atomically store the entire new round.
   A stale concurrent writer must reload. A failed step leaves the previous committed round.
   A failed fit does not erase selection or revealed-label accounting and never reselects.
9. **Continue or stop.** Stop at page budget, round limit or exhausted pool. Completed resume
   validates inputs and returns without training again. Export reads one committed snapshot.

Selection uses sorted unique candidates, a local seeded random generator, and deterministic
page-ID ties for uncertainty ranking. Stage numbers are baseline 0, initial fit 1 where present,
then acquired rounds; the acquisition seed is `seed + committed_rounds + (1 if initial_fit)`, so
the initial fit and the first acquired round cannot draw the same order. An initial batch counts
toward the label budget. For example, batch 3 with a five-page budget produces batches of 3 and 2,
and an initial 16 with batch 8 and a 24-page budget produces 16 labels then 24.

Predictions must match the exact requested page set and run/round/checkpoint/purpose.
Purposes distinguish `pool`, `baseline_validation` and `validation`. Validation output cannot
be reused as acquisition scores. Failed outputs remain explicit `invalid_output`, `truncated`
or `refusal` records; partial malformed text is not quietly scored as successful OCR.

The evaluator `page-text-nfc-v1` joins declared line text with LF, normalizes CRLF and Unicode
NFC on a copy, and preserves case, punctuation and spacing. It sums per-page Levenshtein edits
before dividing by reference characters or whitespace-separated words. Undefined denominators
have separate defined flags; rates can exceed 1. Failed predictions count as empty hypotheses
with separate failure counts. This is an engineering text view, not official layout-aware
benchmark scoring. Edit distance now comes from RapidFuzz; the counting semantics are unchanged.

`page-joint-nfc-iou50-v1` adds separately versioned engineering localization counts
(`page-boxes-iou50-v1`) on top of those unchanged text metrics. Reference and predicted boxes are
matched one-to-one, greedily in descending IoU at threshold 0.5, with ties broken by reference
then prediction index. Unmatched predictions and unmatched references both count as errors, and a
failed page contributes zero accepted boxes, so its reference lines stay unmatched. Reported
`matched_line_cer` always travels with `matched_line_coverage`, so omitted lines cannot inflate
apparent quality. This is engineering reporting, not a final-test benchmark protocol.

## 6. Model runtime: LLaMA-Factory

`integrations/factory_model.py` is the only supported real-model path. It is a thin adapter: the
toolkit owns optimization, batching, loss masking, model loading and checkpoint serialization.
The adapter owns exactly four things — converting revealed examples to upstream rows, rendering
one YAML from the recipe, invoking the toolkit, and mapping raw responses onto the existing
`Prediction` contract. The deleted `qwen.py` was 1,567 lines; the replacement adapter and shared
`artifacts.py` helper total about 1,373 lines. The main maintenance reduction is delegating model
training and loading to upstream; this migration does not halve the whole runtime.

### The one validated recipe

`experiments/recipes/read2016-llamafactory-joint.json` is the single source of execution settings.
The native YAML is rendered from it per operation, so there is no second configuration that can
disagree. Its digest travels inside every checkpoint binding, and the coordinator recomputes that
binding locally, so a worker running a different recipe cannot produce an accepted result.

| Setting | Value |
| --- | --- |
| Version IDs | `read2016-llamafactory-joint-v1`; `llamafactory-lora-sft-v1`; `llamafactory-hf-greedy-v1`; evaluator `page-joint-nfc-iou50-v1` |
| Toolkit | LLaMA-Factory 0.9.5, template `qwen3_vl_nothink` for training and inference, `infer_backend=huggingface` |
| Model task | Ordered line text and boxes as strict JSON; coordinates normalized to 0–1000 relative to the original page |
| Image view | Full page; the toolkit resizes by pixel budget, `image_max_pixels` 1,048,576 and `image_min_pixels` 65,536, identical in training and inference |
| Sequence limits | Prompt ≤2,048; target ≤4,096 including EOS; total ≤6,144; `cutoff_len` is set to 6,144 |
| Trainable adapter | BF16 LoRA rank 8, alpha 16, dropout 0 on language `q,k,v,o` projections; vision tower frozen |
| Optimization | 3 epochs, batch 1, accumulation 4, AdamW, learning rate 3e-5, linear schedule, warmup ratio 0.05, gradient checkpointing, no packing, no automatic validation split |
| Decoding | Greedy, ≤4,096 new tokens |
| Key pins | llamafactory 0.9.5, Transformers 5.6.0, PEFT 0.18.1, Accelerate 1.11.0, TRL 0.24.0, Torch 2.8.0; exact resolution in `uv.lock` |

The upstream range caps Transformers at 5.6.0 and PEFT at 0.18.1, so the project's previous newer
pins were lowered rather than forced. The native CLI check also exposed the upstream
Torch 2.9.x/Conv3D rejection; Torch/TorchAudio 2.8.0 and TorchVision 0.23.0 avoid that guard. These are engineering settings, not claimed optimal
hyperparameters. EXP-004 measured actual training and initial validation; its acquired round was incomplete.

### Length preflight is mandatory

The upstream supervised processor silently truncates anything longer than `cutoff_len`. Before any
optimizer step, `fit` encodes every selected example through the actual toolkit template and
tokenizer and records `prompt_tokens`, `target_tokens` and `total_tokens` per page. A page over the
limits raises and the fit fails; nothing is truncated, dropped or substituted, and a needed recipe
correction must be recorded before execution rather than applied mid-run. Those measured lengths
are persisted in the checkpoint's training record. Preflight also executes the upstream SFT
processor and checks its full token IDs and labels: only the complete response, EOS and native
trailing newline are supervised. The target limit conservatively includes that newline.

### Known upstream decoding limitation

LLaMA-Factory's HuggingFace engine decodes generated text with
`clean_up_tokenization_spaces=True`, which rewrites `" ."`, `" ,"`, `" ?"`, `" !"` and a few English
contraction patterns. References are not rewritten, so measured CER slightly penalizes
transcriptions containing those sequences. This is recorded in the recipe's `known_limitations`
and is a measurement caveat, not a repaired prediction.

### Assets

The pinned assets are unchanged: `Qwen/Qwen3-VL-4B-Instruct`, model and processor revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`. Its two weight shards are approximately
8.875 GB. Publisher metadata declares Apache 2.0; the pinned repository had no LICENSE file,
so the staging record distinguishes publisher metadata from separately retained Apache text.
Model files are downloaded and checksum-verified separately; they are never committed to Git.

### Operations

`load_base` verifies pinned files and publishes a base
checkpoint manifest. `fit` writes `train.jsonl` and a minimal `dataset_info.json` into a unique
per-operation directory, renders `resolved.yaml`, runs the CLI as a subprocess argument array
with captured exit status and log, then records the native adapter plus a manifest holding base
and processor revisions, recipe and environment identity, selected page IDs and source hashes,
measured preflight lengths, the toolkit's own optimizer-step and loss summary, and artifact
hashes. Upstream demo datasets and automatic validation splitting are excluded. A finite, nonzero LoRA B tensor is required under the verified native default initialization,
where B starts at zero. Random nonzero LoRA A alone does not count as training. Checkpoint IDs are content hashes and reload uses
ordinary upstream adapter support; there is no checkpoint-config rewriting and no hard-coded
tensor topology check.

Prediction parses only the specified JSON structure and validates original-pixel geometry. It
preserves raw generated output and the finish reason, and records `truncated` or `invalid_output`
instead of inventing regions or confidence. `ChatModel.get_scores()` is a reward-model interface
and is deliberately not used as OCR confidence, so real acquisition still supports random only.
One loaded model is reused across the pages of a single operation.

Remote fitting writes one probe after training, then loads the published adapter in a second
fresh interpreter and writes a second probe. Greedy decoding must reproduce the first probe's
status, finish reason, response length, response digest and parsed regions exactly, from two
distinct process IDs. The same selected TRAIN page is a diagnostic, not independent evaluation:
this proves the native checkpoint reloads consistently, not that the transcription is correct.

## 7. Modal execution, persistence and recovery

The local coordinator needs ordinary Python dependencies plus the optional Modal SDK. Heavy ML
dependencies live in the locked Linux image. Imports do not implicitly authenticate, deploy or
load model weights.

```text
Local computer                         Modal
Pipeline + LocalOracle
  │ selected-only fit / image-only predict
ModalModel + SQLite operation journal ──► one dispatch Function
  ▲                                      │ checks identity, deadline and files
  │ verified results/artifacts             ├─ fresh child: toolkit operation
  │                                       └─ fit only: second child reload probe
  └─────────────────────────────────────── immutable completion and hashed artifacts

Input Volume: /inputs  (read-only model + images)
Output Volume: /outputs (one run's checkpoints, probes, receipts and control)
```

The two Volumes are distinct. Mounts expose only `/bundles/<bundle-hash>` and `/runs/<run-uuid>`.
Only an explicit eight-file runtime allowlist, the recipe, small processor assets and synthetic CPU tests
enter the image build; no repository-directory upload or real label dataset is used.
The input inventory checks every allowed path/size and hashes requested images when consumed.
The model child independently checks pinned model/processor bytes before use.

The worker requests one L40S, 2 CPU and 32 GiB, with max 1/min 0/buffer 0 containers and
retries 0. Invocation and aggregate defaults remain 2,400 seconds; an explicitly configured
run may use up to 7,200 seconds, with a 300-second startup timeout and termination reserve.
The full migration verification uses 7,200 for both `aggregate_gpu_seconds` and `timeout_seconds`.
The coordinator persists the absolute run deadline before first submission. Existing runs cannot
extend that deadline. Children share the earlier of that deadline and the invocation timeout minus
a ten-second shutdown margin. These bounds reduce exposure; they are not a hard dollar cap.
Queue/startup, billing lag and storage need separate accounting.

The parent never holds a Torch/CUDA model. Each operation uses a fresh child; fitting then
starts a second fresh child in the same Function invocation for reload verification. The
parent enforces deadlines and terminates/reaps child process groups. A result is accepted
only after referenced bytes, ownership, checkpoints and probe evidence verify.

SQLite stores `simulation`, `model-control` and `model-operation` records in the existing
JSON record table. Operation IDs hash semantic inputs, including selected-target digests,
not raw target text. Checkpoints, prediction receipts and probe arrays are immutable artifacts.
The worker publishes `complete.json` last and commits the Volume before returning completion.

| Situation | Recovery behavior |
| --- | --- |
| Known call still running | Reattach using its recorded provider call ID within the original deadline. |
| Completed operation | Reverify and reuse its result; a later validation/round-commit failure need not retrain. |
| Spawn acknowledgement lost | Mark unresolved/UNKNOWN and reconcile; do not blindly resubmit. |
| Deadline/cancellation | Persist cancellation request, cancel known call and confirm termination separately. |
| Partial checkpoint/failed operation | Retain evidence; no automatic overwrite, retry or accepted completion. |
| Round committed but client response lost | Reopen the committed run, rather than applying the round twice. |

This is durable recovery, not a promise of exactly-once paid execution. Uncertain provider
termination remains an operator reconciliation issue. Old failed runs and their original
deadlines/journals remain intact. A new recipe requires a new identity and run, not mutation
of the evidence for a completed run.

## 8. Setup and operating procedure

### Local fixture: immediately runnable without cloud services

Use Python 3.11 and `uv`, from the repository root:

```sh
uv sync --frozen --no-editable --extra dev
uv run --frozen python examples/simulate.py /tmp/ocr-simulation-example
```

Use a new destination. The example generates seven TRAIN pages and separate validation/test
pages, selects 3 then 2, fits a deterministic fixture, resumes and exports. Fixture predictions
are synthetic and do not measure OCR quality. No token, model download or GPU is needed.

The same lifecycle is available as individual commands:

```sh
uv run --frozen active-ocr simulation fixture /tmp/ocr-source
uv run --frozen active-ocr simulation start /tmp/ocr-source/pages.jsonl /tmp/ocr-runs \
  --strategy entropy --batch-size 3 --page-budget 5 --rounds 3 --seed 824
# Set RUN_ID to the UUID printed by start.
uv run --frozen active-ocr simulation run /tmp/ocr-runs "$RUN_ID" --one-round
uv run --frozen active-ocr simulation resume /tmp/ocr-runs "$RUN_ID"
uv run --frozen active-ocr simulation status /tmp/ocr-runs "$RUN_ID"
uv run --frozen active-ocr simulation export /tmp/ocr-runs "$RUN_ID" /tmp/ocr-export
```

`--initial-batch-size N` adds the separate seeded initial-fit stage before acquisition. With
`--initial-batch-size 16 --batch-size 8 --page-budget 24 --rounds 1` the run commits an
`initial_fit` record at 16 labels and then one acquired round at 24, and `rounds.csv` reports
`record_type` `initial_fit` then `round`. Leaving it at the default 0 preserves the old lifecycle.
New initial-fit runs skip zero-shot inference. Selection is persisted before oracle reveal;
`pending_selection.revealed_ids` retains the cumulative consumed-label count on a failed fit.
Reopening retries the same selection. Successful stages clear the pending record. Stage numbers
are 1 and 2, while acquired-round count is `len(rounds)`, zero then one.

### READ conversion

Stage the archive and model files from the URLs and hashes in the
[asset record](../experiments/assets/read2016-qwen3vl4b.json); the converter itself does not
download or extract them. Preserve originals and use safe archive extraction. From a clean
reviewed checkout, with at least 2 GiB scratch headroom and a new output path:

```python
from pathlib import Path
from active_ocr.integrations.public_dataset import convert_read2016
from active_ocr.models import SourcePolicy

manifest = convert_read2016(
    Path(".local/assets/read2016-1.2.0/Train-And-Val-ICFHR-2016.tgz"),
    Path(".local/assets/read2016-1.2.0/extracted"),
    Path(".local/prepared/read2016-new"),
    source_policy=SourcePolicy.READ2016,
)
```

Conversion publishes `pages.jsonl` last. It has no overwrite/resume mode; interrupted attempts
need a new destination. Existing verified preparation can be reused unchanged.

### Real runtime bootstrap

The real execution path uses this sequence. A fresh clone is **not** a one-command cloud deployment:
it needs authenticated Modal access, verified assets and the observed build/deployment records.
These are explicit operator inputs; the CLI does not invent provider IDs or bootstrap resources.

1. Freeze a clean source checkout and local environment. Install the `modal` extra separately
   from `gpu`. Export Linux build requirements from the frozen lock:

   ```sh
   uv export --offline --frozen --format requirements-txt \
     --no-emit-project --no-header --no-annotate \
     --extra dev --extra gpu --extra modal \
     --output-file /approved/staging/requirements-linux.txt
   ```

2. Form `BuildSpec` from that source SHA, `RUNTIME_CODE_FILES`, lock, exported requirements,
   `tests/test_factory_model.py`, `experiments/recipes/read2016-llamafactory-joint.json` and the
   pinned small processor manifest. Resolve the reviewed App and two
   Volume names; record actual distinct IDs. Reuse a verified input bundle where available;
   otherwise `input_upload_files` / `upload_input_bundle` verify and upload the explicit allowlist.
3. `entrypoints.modal_app.build_image` performs the actual CPU build. It requires checkout,
   exported requirements, processor root, observed App and output Volume. The build runs
   `tests/test_factory_model.py -k test_actual_`: every selected case must pass, with no skips
   or failures. It returns observed image ID, `BuildReceipt` and report. Retain and independently
   verify receipt/report bytes and actual package inventory; a cached image without evidence
   is not accepted. The CPU build has no GPU or real image/label/4B-weight payload.
4. Freeze the receipt's source/package/build hashes, model manifests, recipe, validation subset
   and image-based deployment reference in `ExpectedIdentity`, `RealOCRConfig` and
   `SimulationConfig`. `real create` generates the actual run UUID with no model work.
5. Save canonical `RuntimeSettings` for that UUID, input bundle, observed image and Volumes.
   `create_app(settings, settings_file=...)` constructs the Function; explicit `app.deploy`
   deploys it. Check collision/replacement safety first. The per-run settings file is a deferred
   mount, preserving the accepted image ID. Record actual Function/App/mount identities in
   `DeploymentObservation`. Constructor success is not deployment verification.
6. Run local then provider preflight, followed by the bounded run. Keep settings, deployment,
   frozen checkout and client environment with the run. Stop/reconcile actual failures before
   another attempt. Do not extend old deadlines or overwrite old artifacts.

These schemas and helpers live in `integrations/modal_model.py` and `entrypoints/modal_app.py`.
They are the authoritative executable contracts. Local examples below use placeholders for
**existing verified** configuration and paths, not a script that creates them automatically:

```sh
export PYTHONPATH=src
python -m active_ocr.entrypoints.cli real create "$MANIFEST" "$RUNS" "$CONFIG"
# Set RUN_ID to the created UUID and prepare its SETTINGS and observed DEPLOYMENT as above.
python -m active_ocr.entrypoints.cli real preflight "$RUNS" "$RUN_ID" "$SETTINGS"
python -m active_ocr.entrypoints.cli real preflight "$RUNS" "$RUN_ID" "$SETTINGS" \
  --deployment "$DEPLOYMENT" --provider
python -m active_ocr.entrypoints.cli real run "$RUNS" "$RUN_ID" "$SETTINGS" "$DEPLOYMENT"
python -m active_ocr.entrypoints.cli real status "$RUNS" "$RUN_ID"
python -m active_ocr.entrypoints.cli real resume "$RUNS" "$RUN_ID" "$SETTINGS" "$DEPLOYMENT"
python -m active_ocr.entrypoints.cli real export "$RUNS" "$RUN_ID" "$EXPORT"
```

`real run --one-round` performs one orchestration step: an initial fit when
`initial_batch_size` is positive, otherwise the historical zero-shot baseline. `real run` without it continues until configured
completion. `real reconcile ... OPERATION_SHA --call-id OBSERVED_CALL_ID` attaches known work;
`--completion` verifies saved completion evidence, and `--cancel` requests cancellation.
None is a blind in-place retry command. Read `--help` for exact positional arguments.

Exports are `results.json` (full committed run), `rounds.csv` (baseline when present, initial fit when present, then acquired rounds), and
`validation.json` (fixed membership/count). Use a new export directory. A completed real resume
still checks exact source/dependencies: run it from its frozen execution checkout, even when
current main differs only in documentation.

### Supported commands and removed services

The CLI now exposes only `simulation` and `real`. The old Label Studio client/template,
HTTP GPU client/server, polling controller, manifest importer, YAML/service configuration,
Docker/Compose files, obsolete domain records and their exclusive tests were removed.
`Pipeline` takes one `SQLiteStore`; it no longer constructs annotation or HTTP clients.
The local package needs Pillow, Pydantic and Typer; Modal and ML remain separate extras.
There are no placeholder service endpoints left in the supported execution path.
Historical datasets, checkpoints, run stores and measured results were not deleted. Frozen
execution checkouts are still used to inspect or resume runs requiring their original identity.

## 9. The completed real round

### Current LLaMA-Factory run — EXP-005

Run `cdc42baf72544791b6cdba6e6ca31066` used READ2016 full pages and the pinned Qwen3-VL-4B.
Initial 16 and acquired 8 TRAIN pages were unique; only those 24 labels were revealed. The same
8 official validation pages were evaluated after both fits. Document grouping remains unknown:
these are engineering results, not a document-independent benchmark.

| Measurement | Initial fit: 16 pages | Acquired round: 24 pages |
| --- | ---: | ---: |
| Native optimizer updates | 12 | 18 |
| Changed LoRA B tensors | 144 | 144 |
| Native training seconds | 62.2985 | 92.4838 |
| Valid / invalid / truncated outputs | 2 / 4 / 2 | 1 / 3 / 4 |
| Character edits / reference characters | 4094 / 4099 | 4097 / 4099 |
| CER | 0.998780 | 0.999512 |
| WER | 1.0 | 1.0 |
| Box F1 | 0 | 0 |

Both checkpoints reproduced their greedy probes in fresh processes. Initial fit plus one acquired
round committed; completed resume made no additional model calls, and JSON/CSV exports matched.
Final inventory showed zero active containers. The run took 26.24 minutes; provisional App cost
was USD 0.9512 and aggregate project gross usage USD 4.2823, with billing lag and retained storage
reserves. Detailed identities, artifacts and cost caveats are in EXP-005. Independent artifact review passed, including recomputation of the reported joint metrics. These poor predictions
remain failures; no JSON repair or model-quality improvement is claimed.

### Historical custom-runtime run — EXP-003

### Setup

| Item | Actual run |
| --- | --- |
| Evidence | [EXP-003](../experiments/EXP-003.md), 17 September 2026 |
| Runtime UUID | `c241b57754f94b86a389ed20842ec9e7` |
| Source | `1a03f77519439c0601a83c8c3b4670ca896de337` |
| Strategy / seed | Random / 824 |
| Budget | Batch 2, total revealed pages 2, maximum acquired rounds 1 |
| Selected TRAIN | `Seite0286`, `Seite0332`; 25 source regions each |
| Fixed VALIDATION | `Seite0355`, `Seite0356` from the official validation split |
| Fit | Reset LoRA from pinned base, three epochs, three optimizer updates |
| GPU | One NVIDIA L40S; CUDA 13.0, driver 580.95.05 |
| Runtime environment | Python 3.11.12; exact 96-package fingerprint retained |
| Final state | One committed round; 2 revealed / 348 remaining TRAIN pages; stop `page_budget` |

Four remote operations completed: load base, baseline prediction, fit including fresh reload,
and post-fit prediction. The CLI run took **613.011 seconds (about 10.2 minutes)**. The five
run/status/resume/status/export commands exited zero; preflight also passed.
Fresh-process completed resume preserved the same run, operation and provider call IDs and
submitted no new model operation. JSON, CSV and validation membership exports agree.

Training produced six finite page losses:
`1.532149, 1.406864, 1.389118, 1.517770, 1.360519, 1.500381`.
There were 13,005 supervised tokens; all 144 adapter tensors changed and frozen-base checks
passed. These few losses are diagnostics, not a learning curve or quality improvement result.

The adapter file was 23,614,424 bytes. Fresh reload used different interpreters and reproduced
the exact tensors, generated IDs, status and regions. Probe logits had maximum absolute
difference **0.0**. The probe output itself was truncated, so matching it demonstrates save/load
consistency, not correct transcription.

### OCR results

| Validation measure | Base model | After selected-page fit |
| --- | ---: | ---: |
| Valid page outputs | 0 / 2 | 0 / 2 |
| Invalid JSON / truncated output | 1 / 1 | 1 / 1 |
| Character errors / reference characters | 981 / 981 | 981 / 981 |
| Word errors / reference words | 151 / 151 | 151 / 151 |
| CER | 100% | 100% |
| WER | 100% | 100% |

`Seite0355` failed the output contract; `Seite0356` reached the generation limit, both before
and after fitting. Failed outputs have empty accepted regions, explaining the measured error
rates. This does **not** show that every raw generated character was wrong; raw malformed or
partial outputs were intentionally excluded from accepted OCR.

Each selected prompt used 812 tokens. Targets including EOS used **2,206 and 2,129 tokens**,
both longer than the unchanged **2,048-token generation cap**. This is concrete evidence that
the recipe needs output/decode calibration. It does not prove that a larger cap alone will
solve formatting, recognition or layout errors. No settings were changed midway through this run.

### Time, memory and cost

| Model operation telemetry | Elapsed seconds | Peak allocated bytes |
| --- | ---: | ---: |
| Base load | 28.017 | 8,875,651,584 |
| Baseline prediction | 72.631 | 9,335,729,664 |
| Fit, before fresh reload | 148.407 | 15,119,284,224 |
| Post-fit prediction | 102.957 | 9,359,336,448 |

Model-stage timings are not total provider lifecycle or billed durations. The largest measured
allocation was **15.12 GB decimal**; this is not total device memory or proof that an untested
smaller GPU will work. L40S was retained for the bounded verified fit; cheaper hardware can be
assessed later without claiming it has already been tested.

Provisional run App cost was **USD 0.35677159**: GPU 0.30240138, CPU 0.01467034 and memory
0.03969987. Latest aggregate metered compute was **USD 0.99 before credits**, with USD 0 billed
after credits at that observation. Storage and final billing were unsettled; a USD 2.16 storage
reserve remained. The authorized ceilings were USD 5 for first verification and USD 30 total.
Zero active containers were observed at completion, with minimum containers zero. This report
does not start further workloads or promise zero continuing storage cost.

### Reproducibility anchors

The full run record retains checkpoint, settings, environment and artifact identities. Key anchors:

| Artifact | SHA-256 |
| --- | --- |
| Prepared source manifest | `ac0a02a218a2142c6ee92ae964e7828ddfc65545dd6ca12f695815bb4493bdd6` |
| Ground truth | `7636a6bbb33979326acf1904e6ea7cc5c67bdabaabc22077d0c36aa5fa1bdbc5` |
| Base checkpoint | `e783f52809eeda0fe459856243c8b84a6f4a952bce49ff4af762b00b2a0a2b09` |
| Adapter checkpoint | `b726e06ce06de3e052de914461dd3ec0b644b59249fce58a3fa9040b6dc1d660` |
| Saved result summary | `a474d49715d32634c3e1b22992041c40f7cc3ba68df8aa4ed0dfb0a38dc388ab` |
| Exported rounds CSV | `f628eec19e5283c6247d7496c50a3a11e6946a47f113e0dc3679ff397a6969c1` |

On the original workspace, retained evidence is under `.local/verification/exp-007/run-3/`:
`runs/simulation.sqlite3`, `handoff-index.json`, `result-summary.json`, `terminal/remote/`,
command records and `export/`. These ignored artifacts are intentionally absent from GitHub.
Their checksums support verification by someone with artifact access; the public repository
alone does not contain the private local run store or downloaded checkpoints.

## 10. Engineering verification and resolved problems

**LLaMA-Factory migration status.** The toolkit adoption is implemented and locally tested. Every
upstream contract it relies on was read from the v0.9.5 source tree rather than from moving `main`
documentation: the released `examples/train_lora/qwen3vl_lora_sft.yaml`, the `qwen3_vl_nothink`
template registration, `ChatModel.chat(..., images=...)` and its `Response` fields, the sharegpt
multimodal `dataset_info.json` schema, and the supervised processor's silent `cutoff_len`
truncation. Actual local CPU checks exercise the pinned processor/template and native PEFT
initialization/update/save plus the actual toolkit CLI and fresh-process native adapter reload
on a tiny random Qwen3-VL. All five named checks also passed in the Linux image with zero skips.
EXP-004 verified initial training/reload and validation but stopped before its acquired round.
EXP-005 completed both fits, evaluations, one acquired round, resume and exports. Source and CPU
passes alone are not the basis for that result; actual saved runtime artifacts are retained.
The duration correction at `11a58ad` passed 32 author checks and six independent checks while
preserving defaults and immutable deadlines.

The preceding maintenance candidate `4925b720ebfb585e2befad60660dda5a663bfaa8` removed 13 legacy
files and corrected the output-capacity mismatch. Its full suite passed 718 cases with 11 explicit
ML skips, and independent review passed 26 focused checks. Those statements apply to that SHA and
to the now-removed runtime.

Verification progressed from cheap local checks to actual execution. Scope matters: a fixture
pass is not a GPU pass, and an engineering pass is not a scientific finding.

| Layer | Evidence established |
| --- | --- |
| Local contracts | Atomic baseline/rounds, selection/label isolation, identity checks, failure cases, resume/export and exact metric counts; [review](verification/local-contract-review.md). |
| Actual source | All 400 pages / 9,410 lines converted, 804 original files checked, full freeze and fresh-process reopen; [review](verification/read2016-converter-review.md). |
| Qwen offline (superseded) | Processor/target/mask/geometry, trainable topology, strict checkpoint/reload and rejection behavior for the removed runtime; [review](verification/qwen-runtime-review.md). |
| Modal integration | Operation ownership, deadline/preflight ordering, label-free boundaries, artifact verification and recovery; [review](verification/modal-runtime-review.md). |
| Historical Linux CPU image | All 11 processor/tiny-model cases passed with zero skips/failures on the accepted image; [CPU review](verification/modal-cpu-review.md). |
| Historical adapter fix | Independent 16-case focused review plus actual 36-layer export regression within the same 11-case CPU gate. |
| Historical 4B/GPU round | Completed training/reload/baseline/validation/persistence/resume/exports; independently inspected artifacts without rerunning training. |

Earlier attempts exposed real integration issues: provider mount/publication semantics,
processor normalization representation, and PEFT's compression of explicit LoRA target names.
They were diagnosed against observed evidence and corrected before the accepted run. The final
production fix was eight lines in the existing adapter save path, preserving strict reader
validation. Failed attempt records remain in EXP-001/002 and CPU history; they are not erased or
counted as successful rounds. Detailed earlier PASS/FAIL statements apply to their original SHA.

For changes to application behavior, use focused relevant tests, then broader checks warranted
by the change. Ordinary local verification:

```sh
uv run --frozen pytest
uv run --frozen ruff check src tests examples
python3 scripts/validate_agent_system.py
```

Actual ML cases require their pinned environment and processor assets; missing-dependency skips
are unverified scope, not model passes. There is no configured type checker. Documentation-only
edits need link/content/diff checks; they do not justify another GPU run or redundant test campaign.

## 11. Research progress and next steps

Research has selected and staged an initial corpus/model pair, recorded source/license/revision
evidence and examined future alternatives in the dated [pilot note](../research/RES-002-real-ocr-pilot.md).
That note is rationale and proposals, not the current execution status or an approved final
comparative methodology. READ/Qwen was a feasibility choice; historical-handwriting accuracy
and lack of pretraining overlap were never assumed.

**The immediate research blocker is useful structured OCR, not pipeline connectivity.**
The next work should remain small and sequential:

1. **Establish useful OCR before comparing strategies.** The toolkit recipe preserves the corrected
   output capacity, but EXP-005 still produced invalid/truncated output and zero localization F1.
   Diagnose model quality under a separately recorded research recipe; do not silently repair
   outputs or source labels.
2. **Establish adequate initial adaptation.** Choose a modest, fixed training subset and training
   budget; verify usable line/text output and held-out validation behavior before spending on
   acquisition comparisons. Three updates on two pages are not this baseline.
3. **Freeze comparative methodology.** Establish document grouping or explicitly limit the claims;
   define the final-test protocol, layout/text metrics, common initial labeled set, equal page
   budgets, seeds, reset-fit policy and stopping rules. Keep final test isolated from calibration.
4. **Implement real uncertainty only when needed.** Specify how token-level evidence becomes a
   page-level confidence/entropy score and how failures are handled. Test actual scores before
   releasing the existing uncertainty selectors for real model runs.
5. **Run fair comparisons, then replication.** Compare random and candidate strategies under the
   same conditions across multiple seeds; export quality versus labeled-page curves with
   uncertainty. Change datasets/models in separate controlled replications, not simultaneously
   while diagnosing the first working baseline.

No new GPU result is claimed for the v2 correction. No claim is currently supported for
selection superiority, annotation-time savings, useful OCR,
multi-seed stability or document-independent generalization. Revealed-page counts simulate
annotation budget; human annotation seconds remain unknown. No research campaign or extra cloud
job is authorized merely by this report. Dataset/model replacement remains possible through
the existing boundaries without adding speculative infrastructure.

## 12. Team responsibilities and documentation policy

| Existing role | Completed/current responsibility |
| --- | --- |
| Manager | Priorities, releases, source/evidence reconciliation, integration and user-facing project state. |
| Research | Source-backed corpus/model recommendation and methodological limitations; later interpretation. |
| Planning | Concrete data/model/runtime contracts and acceptance plan, now implemented for Stage 2. |
| Development | Local simulation, converter, model adapter and coordinator/CLI implementation and fixes. |
| Testing & Code Quality | Independent checks of exact candidates, actual source/CPU evidence and completed run artifacts. |
| Experiment & Platform | Modal build/worker boundary, staged assets, bounded execution, receipts and provisional cost observations. |
| QA & Understand | Independent explanations, documentation and mentor slides with clear visuals and qualified results. |

The seven persistent roles are initialized; live task IDs, paths and assignments stay in ignored
shared local memory. Their protocol is in [coordination/README](../coordination/README.md).
That operating protocol and [AGENTS](../AGENTS.md) remain necessary instructions, not duplicate
project reports. Production ownership and independent review remain in effect.

Maintain this guide as the single current architecture/progress entry point. Keep new results
in concise dated experiment records, then update the relevant guide summary. Preserve original
failure/review evidence and scientific citations. Do not append routine task histories, machine
inventories or raw model outputs here. Removed implementation guides/plans remain recoverable
from Git history; their removal does not delete data, checkpoints or cloud resources.
