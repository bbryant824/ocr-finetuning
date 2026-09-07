# Active Learning for OCR Annotation

A small research pipeline for studying whether active-learning strategies can reduce the amount
of human annotation needed to adapt OCR models to historical and low-resource documents.

Label Studio provides line-box and transcription annotation. The local pipeline manages data and
experiments, while Qwen training and inference run in a separate remote GPU process.

## Current status

The initial local pipeline is implemented. It can import a page manifest, copy images into
Label Studio's mounted data directory, persist research state, create strategy runs, select
annotation batches, exchange Label Studio payloads, submit remote jobs, and advance the
experiment loop when used with compatible services or test fakes.

Qwen loading, fine-tuning, and inference are intentionally left for the model-development phase.
The active-learning module currently contains only simple random, least-confidence, and entropy
baselines so future research algorithms have one focused extension point.

## Structure

```text
src/active_ocr/
├── models.py             # Shared pages, annotations, predictions, and experiment state
├── pipeline.py           # Coordinates one annotation/training/selection loop
├── active_learning.py    # Query strategies only
├── evaluation.py         # Research metrics only
├── config.py             # Small validated YAML configuration
├── integrations/
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

## Pipeline

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

## Setup

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

## Commands

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

Its health endpoint works now. Model job endpoints remain explicit placeholders until Qwen
fine-tuning and inference are implemented.

## Runtime files

SQLite state, imported data, predictions, checkpoints, and reports belong under `var/`, which is
ignored by Git. API tokens belong in `.env` and are also ignored.
