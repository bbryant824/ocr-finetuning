# Local true-label active-learning simulation

This path verifies the loop using source true labels as a simulated annotator. The built-in
`deterministic-fixture-v1` model produces empty OCR regions and synthetic hash-based scores.
These are **not OCR quality, calibrated uncertainty, measured annotation effort, or evidence that
one acquisition strategy is better**. No downloads, services, credentials, network, torch or GPU
are needed. Real adapters and final-test evaluation remain future work.

## Flow and boundaries

`Pipeline` coordinates: freeze inputs → select train pages → oracle reveal → reset-fit on cumulative
revealed examples → predict remaining train pool → atomically commit the round → repeat.
The existing selectors implement random, least-confidence and entropy. All strategies start with
the same seeded random batch for matching train membership. Source order and dataset/GT hashes do
not seed selection or scores. Later fixture scores depend only on revealed training examples.
Random skips prediction unless `report_predictions` is enabled.

Code lives in [Pipeline](../src/active_ocr/pipeline.py), shared [models](../src/active_ocr/models.py),
[oracle/model adapters](../src/active_ocr/integrations/simulation.py) and the existing
[SQLite store](../src/active_ocr/integrations/storage.py). Older service methods remain compatible
and are not called by this path.

`LocalOracle` owns source labels. `reveal(ids)` accepts only train IDs, and Pipeline requests its
cumulative selected IDs, including reconstruction on resume. `fit(examples, seed=...)` receives
exactly those image/true-region pairs and returns a nonempty model ID. `predict(pages,
experiment_id=..., round_number=..., model_id=...)` gets image metadata only. Its results must cover
the requested pool exactly with matching run/round/model and valid scores for the configured
selector. Global page/annotation/prediction records from other runs are not consulted. These are
cooperative adapter boundaries, not a security sandbox against malicious local code.

## Normalized JSONL source

`active-ocr simulation fixture NEW_DIRECTORY` generates seven train images, one validation image
and one test image, all distinct, with known synthetic labels. Its `pages.jsonl` is a full example.
Each row contains:

- `id`, `document_id`, `image_uri` (local path relative to manifest, or absolute path).
- `width`, `height`, explicit `split` (`train`, `validation`, `test`), `image_sha256` of file bytes.
- `regions`: a list of `{id, box: {x, y, width, height}, text, illegible}` source records.

Unknown fields, duplicate page/region IDs, unknown splits, cross-split document IDs or identical
image bytes, corrupt images, bad dimensions, nonfinite/out-of-bounds boxes and incorrect hashes
are rejected. No split policy is inferred. Near duplicates/differently encoded copies are not
detected; external datasets still need provenance/leakage review. Missing/null `regions` means
missing labels and is rejected when selected (or when validation evaluation needs them). `[]`
is a valid blank-page annotation. Source text/Unicode and flags are preserved, including empty
illegible text; legacy `Region` normalization is not applied. Raw manifest bytes are preserved too.

## Persistence, budgets and retry

`Pipeline.for_simulation(directory)` uses `directory/simulation.sqlite3` and content-addressed
artifacts. Use a dedicated directory, not the legacy runtime location. Creation freezes manifest
bytes, membership/splits, image bytes and SHA-256s, full config/seed, backend and fit policy.
Python/Pydantic/Pillow versions and Git revision are recorded (`-dirty` for source/config changes;
installed-package runs without Git report an unknown revision). Source manifest/images must remain
available and unchanged. Each round/resume checks source and frozen inputs. Changed GT, membership
or bytes fail instead of silently adapting. Resume uses stored settings; `step_simulation(config=...)`
optionally asserts an exact match. Different backend, fit policy or evaluator ID is rejected.
Use a new run for changed inputs/settings. Direct SQLite tampering is unsupported.

One SQLite JSON record per run contains the snapshot and complete round history. A transaction
compares the previous payload before replacing it with the new complete state. Failure before
commit leaves the old round; loss of the response after commit leaves one full committed round.
No partial durable reveal/budget/score set exists. A stale concurrent writer must reload. Supported
execution is sequential and synchronous; every fit must reset and consume cumulative examples.
The fixture is stateless. Retry may repeat oracle/model calls; remote exactly-once side effects
and real checkpoint recovery are not implemented. This small snapshot design favors readability
over scalability to large datasets/runs.

The initial batch counts toward `page_budget`. Each batch is truncated by remaining budget/pool;
seven pages, batch 3, budget 5 gives 3+2. Empty pool, exhausted budget or reached round limit stops.
Zero budget/rounds is a valid no-op; negative values and nonpositive batch sizes are invalid.
Completed resume verifies inputs but does not train again. Counts mean simulated labelled **pages**;
annotation seconds remain unknown/null.

Export writes one committed snapshot to `results.json` and `rounds.csv`: settings, seeds, input/GT
identities, backend/fit policy, selections, cumulative counts, model IDs, predictions and optional
validation outputs. JSON also has full image metadata/environment. Export requires a new directory
and can be regenerated from SQLite after an interrupted export. SQLite is the source of truth.

## Optional validation hook

By default `validation_metrics` is null and validation predictions are empty. A caller can provide
a `ValidationEvaluator` with a versioned `identifier` and
`__call__(examples, predictions) -> dict[str, float]`. Freeze its ID in `SimulationConfig.evaluator_id`
and pass `evaluator=` to step/run, including resume. The model predicts validation images separately;
only the evaluator gets validation truth from the oracle. Metrics must be finite named numbers;
evaluation failure prevents round commit. Metrics do not feed selection or tune settings. There
is no final-test label access. Callers must document their matching/normalization/metric policy;
none is invented by this hook. The CLI fixture configures no evaluator.

## Commands and adapter points

After installing local dependencies, for example `uv sync --no-editable --extra dev`:

```bash
uv run python examples/simulate.py /tmp/ocr-simulation-example
uv run active-ocr simulation fixture /tmp/ocr-source
uv run active-ocr simulation start /tmp/ocr-source/pages.jsonl /tmp/ocr-runs \
  --strategy entropy --batch-size 3 --page-budget 5 --rounds 3 --seed 824
# Replace RUN_ID with the UUID printed by start:
uv run active-ocr simulation run /tmp/ocr-runs RUN_ID --one-round
uv run active-ocr simulation resume /tmp/ocr-runs RUN_ID
uv run active-ocr simulation status /tmp/ocr-runs RUN_ID
uv run active-ocr simulation export /tmp/ocr-runs RUN_ID /tmp/ocr-export
```

Use new fixture/example/export directories. No `.env` is necessary. The
[example](../examples/simulate.py) shows the Python API too. Later model adapters implement
`SimulationModel` with versioned backend identity, reset-fit semantics and validated predictions;
dataset adapters emit this normalized format without silently changing GT/splits. Real model
loading/training, uncertainty definitions, remote execution and final-test policy need separate review.

Verification: `python -m pytest`, `ruff check src tests examples`, and the example/CLI commands.
Tests cover oracle/model isolation, hidden-label perturbation, matched first samples, interleaved
runs, invalid predictions, failures before/after commit, stale writers, mutation, restart, optional
validation, budget boundaries and exports. This verifies fixture engineering only; independent
review is required before integration.
