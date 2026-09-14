# Local true-label active-learning simulation

This path verifies the loop using source true labels as a simulated annotator. The built-in
`deterministic-fixture-v1` model produces empty OCR regions and synthetic hash-based scores.
These are **not OCR quality, calibrated uncertainty, measured annotation effort, or evidence that
one acquisition strategy is better**. No downloads, services, credentials, network, torch or GPU
are needed. Production real adapters and final-test evaluation remain future work. The Python API also
provides the explicit synthetic contract path described below.

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
cumulative selected IDs, including reconstruction on resume. `fit(examples, seed=..., experiment_id=..., round_number=...)` receives
exactly those image/true-region pairs and returns a nonempty model ID. `predict(pages,
experiment_id=..., round_number=..., model_id=..., purpose=...)` gets image metadata only. Its results must cover
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
installed-package runs without Git report an unknown revision). **Fixture resume requires unchanged code,
software environment and versioned adapters.** Fixture code/software provenance is captured at
creation; fixture resume does not check it against the current installation. The explicit contract
path adds identity checks, as described below. Start a new run after code or dependency
upgrades; mixed-version resume is not verified. Source manifest/images must remain available and unchanged. Each round/resume checks source and frozen inputs. Changed GT, membership
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
For fixtures, zero budget/rounds is a valid no-op; negative values and nonpositive batch sizes are invalid.
Completed resume verifies inputs but does not train again. Counts mean simulated labelled **pages**;
annotation seconds remain unknown/null.

Export writes one committed snapshot to `results.json` and `rounds.csv`: settings, seeds, input/GT
identities, backend/fit policy, selections, cumulative counts, model IDs, predictions and optional
validation outputs. JSON also has full image metadata/environment. Export requires a new directory
and can be regenerated from SQLite after an interrupted export. SQLite is the source of truth.

## Optional validation hook

By default `validation_metrics` is null and validation predictions are empty. A caller can provide
a `ValidationEvaluator` with a versioned `identifier` and
`__call__(examples, predictions) -> dict[str, int | float]`. Freeze its ID in `SimulationConfig.evaluator_id`
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
review accepted implementation `2d24c6493ce8051e9a6ae95d9d30fd5f7b8e55d7` with the code/environment
resume limit above. All 63 tests and Ruff passed; independent probes checked separate-process CLI,
SQLite rollback and concurrent stale writes, reset-fit retries, held-out validation isolation and
3+2 page-budget completion. This acceptance covers the documented synchronous fixture scope.


## Local baseline and real-adapter contracts

**Implementation state:** local records, boundary checks and synthetic adapter execution exist.
Qwen/Modal adapter construction and execution, weight-byte verification, source conversion, remote
operation recovery and uncertainty remain unavailable. The CLI still constructs fixtures only.
These contracts make no OCR, GPU, learning or annotation-time claim.

`SimulationRun.kind` is explicit:

| Kind | Configuration and execution |
| --- | --- |
| `fixture-simulation-not-ocr-evidence` | Default fixture backend, no real recipe or baseline. |
| `adapter-contract-test-not-ocr-evidence` | Frozen real-shaped recipe plus an explicitly supplied synthetic adapter. |
| `real-ocr-simulated-annotation-v1` | Can represent a frozen recipe/run locally; execution raises `NotImplementedError` in this slice. |

`create_simulation(manifest, config, kind=RunKind.CONTRACT_TEST)` freezes a `SimulationConfig`
whose `real` is a `RealOCRConfig`. Its backend/evaluator must match that recipe. This path currently
requires random selection, `report_predictions=False` and `reset-fit-cumulative-v1`; validation
runs after every fit despite skipping pool predictions. An omitted adapter never becomes a fixture.

`RealOCRConfig` contains backend, recipe version, model/processor repository and commit revisions,
training/decode policy IDs, evaluator ID and `ExpectedIdentity`. The latter freezes clean source
SHA, local dependency SHA-256, model/processor commit and file-manifest hashes, recipe/evaluator
versions, and optional code-bundle/remote-dependency/build-spec hashes and deployment reference.
Model/processor pins and recipe/evaluator versions must agree between these records.

`local_contract_identity()` reports source revision and the SHA-256 of canonical compact JSON
containing Python version and sorted installed distribution name/version pairs. Names use lowercase
and collapse runs of hyphen/underscore/dot to hyphen. A dirty checkout (including untracked,
nonignored files) or unavailable Git cannot match a clean expected source SHA. Creation checks
local identity without calling a model. Each step checks local identity, explicit adapter kind,
backend, fit policy, complete recipe and its declared identity before model work, between load/fit
and prediction, and before commit. Completed resume checks inputs/config/identity before returning.
These declarations do not verify actual checkpoint bytes; a later real adapter must do that.

Execution telemetry is optional and absent initially: device, driver/CUDA, free/peak memory,
elapsed time, call ID and billed cost. Adapters may provide an `ExecutionTelemetry` as `telemetry`
after execution. It is captured on the baseline/round but is never equality-compared against an
expected GPU instance or a previous timing/memory observation. No GPU inspection is required.

The adapter metadata is `kind`, `backend`, `fit_policy`, `real_config` and `identity`. In addition
to the owned `fit`/`predict` signatures above, it implements `load_base(*, experiment_id) -> str`.
Real-contract model IDs must match `checkpoint:sha256:<64 lowercase hex digits>`. This is a format
check, not evidence of immutable weight storage. `FixtureModel.fit` still permits direct callers
to omit ownership; ownership does not change its synthetic hash.

Creation performs no model work. It requires nonempty validation membership and stores
`baseline=None`; zero page budget, zero rounds or no training pages do not complete it yet.
The first `step_simulation` loads the base, predicts validation metadata at round 0 with purpose
`baseline_validation`, evaluates through the oracle, and atomically commits only a
`SimulationBaseline`. It has zero labels and no selected/revealed IDs. The step returns immediately,
even with remaining budget. Zero-budget/round/no-train runs complete in that same baseline commit.
The next step starts acquired round 1. `seed + len(rounds)` excludes the baseline, so matching
fixture and contract runs select the same batches with the same seed and membership. Fits always
receive cumulative selected TRAIN examples. No test page or hidden training label is predicted,
revealed or used to form model inputs during the baseline.

Prediction purpose is `pool`, `validation` or `baseline_validation`; baseline purpose is legal
exactly at round 0, other purposes require a positive round. Acquired rounds are positive.
New boundary results must match exact requested pages, run, round, model and purpose. Page results
are reordered to requested metadata order after exact-set/duplicate checks; line order is preserved.
Successful real-contract regions require unique IDs and finite, positive boxes bounded by original
image pixels. Wrong/missing/extra/duplicate pages prevent commit. Old predictions without purpose
remain readable as `pool`, including old fixture validation history; stored history is not relabelled.
The compare-and-swap expected value preserves old field presence so old fixture runs can resume.

Status is `ok`, `invalid_output`, `truncated` or `refusal`, with optional raw-output artifact and
finish reason. Failed outputs remain page records and evaluate as empty text hypotheses, with
separate failure counts. Partial text on failed output is not scored. Valid blanks and failed
outputs therefore remain distinguishable. Raw evidence collection belongs to the later adapter.
Structured region coordinates must be finite for every status before any simulation commit;
malformed failed output is rejected atomically, while valid failed records retain their raw references.
Legacy scoring accepts only pool-purpose results before publishing any predictions or experiment
state, and acquisition reads exclude other purposes. Older omitted-purpose records still mean pool.
Local failures leave the prior run intact; competing callers can compute, but only one can commit.
After a lost post-commit response, reload the run before deciding whether to request the next step.
Repeated remote side effects are not prevented by local SQLite compare-and-swap.

## Page text counts and exports

The contract path defaults to `PageTextEvaluatorV1` (`page-text-nfc-v1`), a fixed engineering
validation view. It is not official benchmark scoring or an approved final-test methodology.
It joins declared line texts with LF, changes CRLF to LF and applies Unicode NFC to a copy.
Case, punctuation, spacing, blank lines and literal illegibility strings remain. Ground-truth
`SourceRegion` is never converted through the legacy `Region` illegibility substitution.

Per-page character and Unicode-whitespace-token Levenshtein edits are summed before dividing:
`cer = char_edits / reference_chars`, and similarly for WER. Empty references contribute insertion
counts and zero denominator. This is a micro aggregate across pages, not a mean of page rates or
a concatenation that permits edits across page boundaries. Rates may exceed 1. Metrics contain
integer `pages`, `char_edits`, `reference_chars`, `word_edits`, `reference_words`, `cer_defined`,
`wer_defined`, `invalid_output_pages`, `truncated_pages`, `refusal_pages` and `failed_pages`.
`cer`/`wer` are present only when their respective denominator is positive. Whitespace-only truth
can define CER while leaving WER undefined. Absent pages or missing annotations raise errors;
existing blank references are valid. All returned values are finite, with no NaN/null rate values.
The legacy standalone `character_error_rate` empty-reference shortcut remains unchanged.

JSON includes the separate baseline, all rounds, purpose/status, counts and optional telemetry.
CSV preserves existing columns and adds `record_type`, validation `purpose`, per-page
`validation_statuses` JSON, the count/defined-flag columns and CER/WER. Baseline exports exactly one
row with round 0, zero labels and empty selected/revealed arrays; acquired rows have type `round`.
Annotation seconds and undefined rates are blank; defined flags distinguish absent rates from zero.
Fixture exports have only their actual round rows. Export still reads one committed snapshot and
requires a new directory.

## Contract verification

Run `python -m pytest tests/test_real_contract.py tests/test_evaluation_counts.py`, then the full
`python -m pytest` suite and `ruff check .`. No configured type checker is present.
The synthetic adapter and its deliberately stubbed source/model identities are confined to tests;
`test_separate_process_resume_and_export` demonstrates API baseline/acquisition resume across
fresh Python processes without ML imports. It does not bypass identity checks for application use.

| Accepted plan checks | Reproducible coverage |
| --- | --- |
| L1 | Existing CLI/API/legacy suite, old-schema SQLite resume, no ML imports in child processes. |
| L2–L3 | No selection/reveal/fit at baseline, zero paths, load/fit/predict/evaluator/CAS failures, response loss and two competing callers. |
| L4–L5 | Ownership/purpose/coverage/geometry rejection, ordering, positive rounds, seed/budget/cumulative-fit invariance and no random pool prediction. |
| L6 | Every identity field, recipe policies, code/dependency/dirty-source changes, adapter mutations, optional varying telemetry and unavailable production execution. |
| L7–L8 | Hand-computed Unicode/blank/whitespace/micro/failure counts, raw source preservation and held-out/hidden-label perturbation. |
| L9 | Separate-process baseline/round resume, JSON/CSV rows and undefined-rate flags, source/identity refusal on completed runs, no overwrite. |

These tests establish local contract behavior only. Independent Testing must review the exact
implementation candidate before acceptance; earlier fixture acceptance does not approve this extension.
