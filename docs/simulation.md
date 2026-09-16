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
Qwen/Modal adapter construction and execution, weight-byte verification, remote
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

These tests establish local contract behavior only. Manager accepted corrected implementation
`dee5e1a5ecbbc326921e0f4019792c8f7e5deab2` after independent Testing passed the 197-test
candidate suite and 40 independent cases. The [review](verification/local-contract-review.md)
records resolved defects, exact reproduction and limits. Real model/GPU execution remains unavailable.


## READ2016 engineering source conversion

`integrations.public_dataset.convert_read2016` implements one pinned mapping for READ2016 1.2.0,
Zenodo record1297399, archive SHA-256
`f4748c58af757e06804e638a6e84c2150daab19f50d55e10aac3115e6bfc1756`.
It requires explicit `read2016-official-unknown-engineering-v1` policy. This permits engineering
verification of the official350 TRAIN/50 VALIDATION pages with grouping recorded **unknown**;
it does not establish document-independent evaluation. TEST content is excluded. The converter
checks the original archive, staged files and image/XML pairs; it does not download or extract.
The400 known relative image symlinks are verified against regular same-split images and omitted.
All804 regular files, including doc.xml/list exports, are copied byte-for-byte without hard links.

Legacy `Page` and its importer still require known document IDs. Only `SourcePage` and
`SimulationPage` support a required-but-null document field under the explicit READ policy.
The strict default `known-document-v1` continues to reject null documents and cross-split known
documents/images. READ requires null on every page, no TEST, consistent row/config/snapshot policy,
and the pinned provenance/membership. Neither filenames nor `-1` become document identities.
Duplicate image content anywhere in READ is rejected; strict known-group behavior is unchanged.

The converter requires PAGE2013-07-15 XML with exact same-split JPG basename pairing. Region order
comes from complete unique indexed references with matching custom indices; explicit region gaps
are retained. Per-region line order must be contiguous from zero. Every line is included, with
its original ID and one literal Unicode element, including blank strings. Standard XML decoding
applies, with no trimming, Unicode normalization, abbreviation expansion or illegibility marker
insertion. Partial unclear/style/abbreviation/sic spans remain raw XML, not invented whole-line
flags; `illegible=False` means not explicitly flagged. Missing Baseline or structure is preserved.
Unknown nodes, annotation attributes/blocks, ambiguous text/order, DTDs and entity declarations fail.

Line polygons remain in original XML. The normalized box is their positive bounded pixel envelope
(maximum minus minimum), with no +1, rounding or clamping. JPEG RGB, single-frame, full decode,
matching dimensions and absent/identity EXIF orientation are required. Nonidentity declared rotation,
unsupported modes and bad images are rejected. No image rotation/crop/resize or source split changes.

Output consists of `source/PublicData/...`, a deterministic `provenance.json`, and `pages.jsonl`.
Provenance has a fixed schema/mapping version, pinned source identity, clean converter code SHA,
engineering/grouping limitations, sorted804-file size/hash inventory and ordered page mappings/counts.
It contains no transcript copies, absolute paths, timestamps or runtime UUIDs. Every row hashes the
same relative provenance file; the manifest is not included in that file's inventory. Original XML
preserves all polygons/custom metadata for audit. Predict/fit page projections explicitly omit the
provenance reference; fitting receives only selected TRAIN line targets and validation truth still
goes only to the evaluator. These cooperative local interfaces are not a sandbox against adapters.

The destination must be new, even if an existing directory is empty. At least2GiB scratch headroom
and a clean Git checkout are required. Conversion reserves the destination exclusively, builds in a
temporary sibling, validates copied bytes, flushes files and publishes `pages.jsonl` last. Its presence
is the completion boundary. Handled failures clean invocation-owned files; unrelated contents are
retained with an incomplete-path note. An interrupted/crashed conversion requires a fresh path;
there is no overwrite/resume mode. Concurrent contenders cannot replace the winner. Source paths
are safe relative paths without symlink components. Original files remain unchanged.

Direct invocation (after independent converter review and structural-smoke release):

```python
from pathlib import Path
from active_ocr.integrations.public_dataset import convert_read2016
from active_ocr.integrations.simulation import LocalOracle
from active_ocr.models import SimulationConfig, SourcePolicy
from active_ocr.pipeline import Pipeline

policy = SourcePolicy.READ2016
manifest = convert_read2016(
    Path(".local/assets/read2016-1.2.0/Train-And-Val-ICFHR-2016.tgz"),
    Path(".local/assets/read2016-1.2.0/extracted"),
    Path(".local/prepared/read2016-v1"),
    source_policy=policy,
)
pipeline = Pipeline.for_simulation(Path(".local/verification/read2016-v1"))
snapshot = LocalOracle.freeze(manifest, pipeline.store, source_policy=policy)
LocalOracle(snapshot)  # structural integrity only; no model work
# A later authorized simulation must set SimulationConfig(source_policy=policy).
```

Freeze/reload/pre-commit recheck the manifest, provenance, all originals, literal XML-to-row mapping,
source/frozen metadata and actual stored image bytes. Missing original files prevent resume.
Config/snapshot/page policies must agree even on completed runs. Existing fixture JSON defaults to
strict policy and retains historical compare-and-swap compatibility. JSON exports carry policy and
provenance in config/dataset; CSV adds `source_policy`, `document_grouping` and `engineering_only`.
A fixture using this source remains fixture evidence. CLI start keeps strict policy and cannot
automatically admit READ; no CLI or model-execution feature is added here.

Author C1–C7 checks are in `tests/test_public_dataset.py` and the existing simulation/independent
contract tests. Run `python -m pytest tests/test_public_dataset.py tests/test_simulation.py`, the full
suite from a clean committed checkout, and `ruff check .`. Synthetic400-page fixtures exercise
publication/integrity without relaxing the public pinned-archive API. Read-only source diagnostics
matched350/50 pages,8367/1043 lines,four empty strings and two missing baselines. Independent actual-data conversion/freeze/reopen (C8) has since passed, as recorded below.
No real model, GPU, cloud, OCR quality or active-learning result is established.

The pinned READ converter passed independent C1–C8 verification at
`8db583326e55b307df487d57d4c03be1d6e12689`:350train/50validation pages,9410literal lines,
804unchanged originals and successful full freeze/fresh-process reopen. See the
[structural review](verification/read2016-converter-review.md) for counts, hashes, commands
and limits. This establishes engineering data readiness, not OCR/model execution.

## Real adapter and bounded validation subsets

The `real` CLI group now connects the exact identity-checked Modal adapter; fixture and local
contract-test flows remain available. See [Modal runtime](modal-runtime.md) and its deployment
bootstrap for required build evidence, settings, observed resource identities and execution gates.
No real execution is implied by local author tests.

`SimulationConfig.validation_page_ids` optionally freezes an ordered nonempty unique subset.
Every ID must belong to the frozen validation split at creation and resume. The same pages, in
that order, go to model prediction and the isolated evaluator; omitted means all validation pages.
This does not edit source membership, ground truth or final-test policy. Exports preserve the run
JSON and add `validation.json` with membership/count; CSV includes the same information.
