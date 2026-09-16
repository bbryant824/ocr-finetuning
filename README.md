# Active Learning for OCR Annotation

A small research pipeline for studying whether active-learning strategies can reduce the amount
of human annotation needed to adapt OCR models to historical and low-resource documents.

The intended current path is a **local active-learning simulation**: existing source true labels
simulate annotation, and a deterministic fixture model exercises the complete loop without services.
The pinned READ2016 data preparation path is independently verified. Real OCR execution and the
Modal connection are the next integration milestone.

## Quick start: no-service simulation

```bash
uv sync --no-editable --extra dev
uv run python examples/simulate.py /tmp/ocr-simulation-example
```

Use a new output directory. The example generates seven train pages plus separate validation/test
pages, selects 3 then 2 train pages under a five-page budget, reveals source labels, fits on the
cumulative set, records synthetic pool predictions, reopens/resumes, and exports JSON/CSV.
No credentials, downloads, Label Studio, network, torch or GPU are needed.

For separate start/run/resume/status/export commands, normalized JSONL format, the optional
validation hook, and adapter/retry contracts, see [the simulation guide](docs/simulation.md).
Run `uv run pytest` and `uv run ruff check src tests examples` for verification.

**Fixture output is not OCR performance or measured human effort.** Model OCR regions are empty;
confidence/entropy values are deterministic synthetic test values. Budgets count revealed pages,
annotation duration stays unknown, and validation metrics are absent unless explicitly supplied
by a caller-defined evaluator. Final-test labels never enter training or selection.

## Current status

The local simulation freezes input membership, source labels/checksums and config; uses the existing
random/least-confidence/entropy selectors; enforces split/run/prediction ownership; and commits each
round atomically in a dedicated SQLite database. Failed rounds can be retried from committed state.

The READ2016 1.2.0 converter preserves the official 350 training / 50 validation pages and all
9,410 literal source lines. Independent checks cover conversion, source checksums, full freezing
and reopening in a fresh process. Document grouping remains unknown, so this dataset is admitted
for engineering verification only. See the [converter review](docs/verification/read2016-converter-review.md)
and [real-pipeline plan](plans/PLAN-003-real-ocr-modal.md) for evidence and remaining gates.

The pinned Qwen runtime implements model loading, reset LoRA fitting, strict OCR output parsing
and immutable checkpoints. It passed [independent offline review](docs/verification/qwen-runtime-review.md);
actual Linux processor/tiny-model checks and 4B/CUDA execution remain unverified. The Modal adapter,
operation recovery and real-run CLI are the next integration step. See the [runtime guide](docs/qwen-runtime.md).

The earlier optional Label Studio/remote-worker flow is retained below for compatibility. Its
GPU job endpoints remain placeholders. No real-model experiment or selection-quality result is claimed.

## Structure

```text
src/active_ocr/
├── models.py             # Shared pages, annotations, predictions, and experiment state
├── pipeline.py           # Coordinates one annotation/training/selection loop
├── active_learning.py    # Query strategies only
├── evaluation.py         # Research metrics only
├── config.py             # Small validated YAML configuration
├── integrations/
│   ├── simulation.py     # True-label oracle, fixture model and synthetic input generator
│   ├── public_dataset.py # Pinned READ2016 conversion and provenance checks
│   ├── local_data.py     # Imports a manifest and stages images for Label Studio
│   ├── storage.py        # SQLite state and local artifacts
│   ├── label_studio.py   # Label Studio API and payload conversion
│   ├── gpu_client.py     # Lightweight client for the remote worker
│   └── qwen.py           # Isolated GPU-only model boundary
└── entrypoints/
    ├── cli.py            # Manual research commands
    ├── controller.py     # Optional polling process
    └── gpu_server.py     # Remote GPU HTTP process
```

There are no domain, port, repository, service, or workflow abstraction layers. Integrations are
passed directly into `Pipeline`, which is enough to replace them with small fakes during tests.

## Optional legacy service flow

```text
page manifest
    ↓
select batch ───────────── active_learning.py
    ↓
Label Studio annotation ─ integrations/label_studio.py
    ↓
remote Qwen training ──── integrations/gpu_client.py → integrations/qwen.py
    ↓
score unlabelled pages
    ↓
next batch or finish

evaluation.py reads experiment outputs separately and never affects selection.
```

Each strategy is stored as its own experiment run. Runs with the same seed receive the same
deterministic random first batch, making random, least-confidence, entropy, and future strategies
directly comparable.

Three data-isolation rules are enforced:

- Predictions are stored as `experiment_id:round_number:page_id` and selection reads only the
  current experiment, round, and model.
- Human annotations may be reused globally, but a training request contains only pages listed in
  that experiment's `labelled_page_ids`.
- Dataset splits are calculated from `document_id`, so every page from one source document stays
  in the same split.

## Optional service setup

Python 3.11 and [uv](https://docs.astral.sh/uv/) are expected.

```bash
uv sync --no-editable --extra dev
cp .env.example .env
uv run active-ocr setup
docker compose up -d label-studio
```

Create a Label Studio project, use `config/label-studio.xml` as its labeling configuration, and
set its project ID and API token in the configuration/environment.

Page images are imported through a JSONL or CSV manifest. Each row must provide a path relative
to the manifest and the original document ID:

```jsonl
{"path":"book_01/page_001.png","document_id":"book_01"}
{"path":"book_01/page_002.png","document_id":"book_01"}
```

Dataset splits are assigned from `document_id`, so pages from one source document cannot cross
between training, validation, and test data.

Imported images are copied into `var/documents/pages/` and referenced through Label Studio's
`/data/local-files/` endpoint. The default Compose mount exposes that directory read-only to
Label Studio. If `documents` is changed in `default.yaml`, set `ACTIVE_OCR_DOCUMENTS_DIR` to the
same host directory before starting Compose.

## Legacy service commands

```bash
uv run active-ocr import PATH_TO_MANIFEST.jsonl
uv run active-ocr experiment start "Random baseline" --strategy random
uv run active-ocr experiment status EXPERIMENT_ID
uv run active-ocr tick EXPERIMENT_ID
```

The optional controller repeatedly performs the same one-step `tick` operation:

```bash
uv run active-ocr-controller
```

The remote GPU service is installed separately so local development does not install PyTorch:

```bash
uv sync --no-editable --extra gpu
uv run active-ocr-gpu
```

Its health endpoint works now. Model job endpoints remain explicit placeholders; the new Qwen
runtime is intended for the separately reviewed Modal path.

## Runtime files

SQLite state, imported data, predictions, checkpoints, and reports belong under `var/`, which is
ignored by Git. API tokens belong in `.env` and are also ignored.

## Persistent FYP agent system

Use **Manager** for project control and **QA & Understand** for explanations. The seven existing
Codex tasks use a shared, ignored `.agent-local/` directory for durable memory and native task
messages for communication. Existing tasks require no repeated setup.

The [operating guide](coordination/README.md), [constitution](AGENTS.md), role Skills,
templates and checks are versioned. Machine/task identifiers, assignments, routine progress,
setup history and operational inventory stay local. Research notes, approved plans and small
reproducibility records belong in Git when ready; datasets, checkpoints and raw logs stay ignored.

Run `python3 scripts/validate_agent_system.py` for portable file checks; `--live` additionally
requires and validates the installed local team. A fresh clone contains reusable instructions,
not this computer's live agent memory. See the guide for backup and worktree behavior.
