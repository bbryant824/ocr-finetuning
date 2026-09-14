# Plan — real OCR with simulated annotation on Modal

Owner: Planning. Version: 0.1, 2026-09-14. Status: DRAFT, phase A.
Inspected code: `a6f31e42b9bbcb33ad2773b91845fa66f9824e34`.
Manager acceptance: NONE. Dataset/model/method/resource acceptance: PENDING.
Research input: RES-002, provisional recommendation; Platform input: EXP-002, phase-A
preflight received. Final scientific/resource evidence and Manager acceptance remain pending.
This version specifies proposed engineering contracts and bounded workloads; it neither
authorizes implementation nor launches or approves a paid experiment.

## Goal and current system

Deliver one reproducible path from public page images and source line annotations to actual
model adaptation, validation OCR metrics and a learning curve against revealed-page budget.
Use the existing local oracle and Pipeline. Keep one synchronous model adapter backed by one
Modal GPU Function, existing SQLite state, and one Volume where feasible. Start with random
acquisition; add a scored comparator only after an exact score definition is accepted and tested.
Implement one concrete recipe with plain records/functions and explicit identity equality checks;
no capability registry, generic deployment framework or automatic GPU/model fallback.

The proposed 4/8/12-page **engineering pilot** has only 1/2/3 optimizer updates per fit under
the example one-epoch recipe. It verifies execution, saved predictions, metrics and recovery;
it cannot establish adequate training or an active-learning advantage. A later comparison needs
calibrated training duration, a larger accepted subset, paired seeds and a separately costed release.

HYPOTHESIS for the later comparison: a specified image-only uncertainty strategy lowers
validation character error at matched revealed-page budgets versus random acquisition under
paired seeds and identical reset-fit conditions. A working random pilot tests feasibility, not
this hypothesis. A negative comparison is a valid result. Revealed pages measure simulated
annotation, never human minutes or proven annotation-time savings.

### Inspected implementation and gaps

| Existing code / behavior | Required extension for real evidence |
| --- | --- |
| `integrations/simulation.py`: `SourcePage`, `_read_source`, `LocalOracle.freeze/reveal/evaluate_validation` | One source-specific public adapter; explicit source reading order, provenance, image transforms and source-split audit. Existing checksum/document/exact-byte leakage checks remain. |
| `SimulationModel.fit(examples, seed)` and `predict(pages, experiment_id, round_number, model_id)` | Real reset-fit, immutable checkpoints, explicit runtime/model recipes and remote operation ownership. |
| `FixtureModel` returns hash scores and empty regions | Retain fixture backend. Real backend must never substitute these scores or claim empty fixture output is OCR. |
| `Pipeline.create_simulation/step_simulation/run_simulation` | Base evaluation, real-backend selection, strict real-run identity checks and remote-effect reconciliation before retry. |
| `SimulationRun.kind` defaults to fixture for every backend; `SimulationConfig` has only backend/fit-policy/evaluator IDs | Explicit run kind/schema version and frozen typed real-model, training, decoding, score, evaluation and runtime identity. |
| `Prediction.round_number >= 1`; no baseline record; zero budget immediately completes | Separate baseline record with zero revealed pages and round-zero prediction ownership; no fake first acquisition round. |
| Random skips pool prediction unless reporting enabled; validation hook is separate | Preserve this efficiency. Require baseline and per-round validation for real result runs even with random selection. |
| `Prediction.regions` has positive-size boxes but no image-bound/finite-box/output-order check | Validate every output against original page coordinates, unique region IDs, text/order/schema and explicit generation-failure status. |
| `QwenRunner` load/train/predict are placeholders; `parse_output` only checks a JSON regions list | Implement only the chosen model in its integration; parsing alone is not functioning OCR. |
| `runtime_identity()` records Git plus Python/Pydantic/Pillow; resume checks backend/fit policy and config, not runtime identity | Guard code, full environment, deployed image, revisions and adapter policies on every real restart/call. |
| `SQLiteStore.compare_and_swap` atomically commits whole rounds | Retain. It does not prevent duplicated paid work before a round commits; add a small local operation journal in the same store. |
| `evaluation.py` has per-string CER, IoU and trapezoidal area; no WER/corpus aggregation/line assignment | Versioned validation policy with explicit text/order/empty cases and complete detection accounting. |
| CLI `simulation run/resume` constructs the fixture implicitly | Add explicit real configuration/adapter construction; no implicit fallback to fixture. |

Evidence inspected: the files above, `models.py`, `entrypoints/cli.py`, `config.py`,
`integrations/storage.py`, `tests/test_simulation.py`, `pyproject.toml`, and
[simulation guide](../docs/simulation.md). The tests cover fixture isolation, ownership, source
mutation, resume and transaction failure. They do not demonstrate real OCR, CUDA training or
Modal recovery. No test suite or GPU workload was executed while writing this plan.

Exclusions: a Label Studio service, HTTP GPU server, queue/broker, new database service,
dashboard, distributed training, concurrent experiments, full parameter sweep, new acquisition
algorithm, final-test evaluation, and automatic use of gated assets. Legacy integrations remain
outside the new path. Preserve all existing fixtures and accepted same-version simulation behavior.

## Exact design

### 1. Public assets and local oracle

Provisional first pair from Research: READ2016/Bozen **1.2.0**,
[Zenodo 1297399](https://zenodo.org/records/1297399), with
[Qwen3-VL-4B-Instruct at ebb281ec70b05090aa6165b016eac8ec08e71b17](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/ebb281ec70b05090aa6165b016eac8ec08e71b17).
Use that model repository revision for processor/tokenizer too, subject to resolved file hashes.
Research reports CC BY 4.0 train/validation archive of 493,223,531 bytes with publisher MD5
`654f2d2c62055f1847f65ba83bd5d744`; compute SHA-256 on staged bytes. Do not mix its changed
GT with the older Zenodo 1164045 release. Research's archive metadata verification remains
provisional evidence here; Planning's direct Zenodo fetch was rate-limited and no archive was
inspected. Planning verified the pinned model tree/Apache-2.0 listing; Research reports
8,875,719,344 bytes of weight shards. Neither establishes page OCR/geometry quality.
Keep test downloads excluded. Florence-2-base-ft is Research's proposed smaller comparator,
not an automatic fallback; any word/region-to-line mismatch requires a method decision.

Research must supply the exact dataset release, license, download URLs, approximate bytes,
document grouping, official split identities and annotation semantics. Platform verifies access,
space and hashes after Manager's asset release. Candidate corpus/model names are not accepted
identities; do not use default 70/10/20 splitting on an actual corpus merely because it exists.

Proposed source-specific `normalize_source(...)` in new `integrations/public_dataset.py`
produces the existing `SourcePage` JSONL plus a small provenance JSON. Choose exactly one source
parser after acceptance, not a registry/framework. Validate:

- Stable corpus-qualified page/document/line IDs; derive documents from source evidence, never
  one document per page to bypass leakage checks. Preserve official splits. If they mix a real
  document across splits, stop and propose a decision; no silent repartition.
- Source text and illegibility flags verbatim. `regions=None` means unavailable; `regions=()`
  means an explicitly annotated blank page. No text normalization during import.
- Preserve an explicit line order from source reading-order metadata. Record the rule and its
  version; reject ambiguous order for page-transcript evaluation until Research resolves it.
- `Box` describes an axis-aligned line rectangle in original image pixels. For a polygon
  source, preserve the original annotation file/hash and describe an enclosing-rectangle
  projection as a lossy adapter view, subject to methodology acceptance. No fabricated boxes
  from transcription length; word/baseline annotations are not silently reclassified as lines.
- Verify image bytes, orientation, width/height, colour conversion and TIFF frame selection.
  First slice accepts one explicit PNG/JPEG/TIFF page per record. Reject ambiguous multipage
  TIFF; PDF rasterization is out of scope. Hash original and any derived image plus transform.
- Audit exact-byte duplicates, document leakage and source provenance; review near-duplicate
  scans/contact sheets before acceptance. Exact-byte checks alone cannot prove independence.

Provenance fields: source release/URL/license, archive SHA-256 and bytes, adapter version/code
SHA, source annotation checksums, ordered source-to-page mapping, derived-image checksums,
split manifest/hash, subset rule/hash, counts and known exclusions. All derived subsets are
document-preserving, selected without labels or model scores, and explicitly approved before use.
Source test partition stays local and excluded from all phase-A/B remote work.

`LocalOracle.freeze` continues to own full truth locally. The frozen source, original corpus
archive, SQLite run dump and local memory must never be uploaded or included in a Modal Image.
The upload allowlist is constructed from image-only page records, not by copying the run's
artifact directory: that directory currently also contains `simulation-source` with all labels.

### 2. Frozen real-run configuration and baseline

Add one typed optional `RealOCRConfig` to `SimulationConfig` in `models.py`; fixture defaults stay
compatible. The table groups fields for readability, not a requirement for six classes. Keep a
single executable recipe and direct validations in existing modules. Required real fields:

| Record | Required identity/configuration |
| --- | --- |
| Model recipe | backend/version; model repo and immutable commit; processor/tokenizer repo and commit; file checksums; architecture; precision; adapter target names/rank/alpha/dropout; initial adapter if any; preprocessing/prompt/output-schema versions |
| Training recipe | reset-fit policy; epochs or fixed optimizer-step policy; batch/accumulation; optimizer/lr/scheduler/warmup/weight decay/clip; seed; image/token limits; masking/truncation/illegibility/blank-page rules; package-lock hash |
| Decode/score recipe | decoding algorithm, max output length, stop tokens, score definition or `none`, text-token mask, normalization and page aggregation versions |
| Evaluation recipe | evaluator ID/version, normalization, source/prediction line order, matching threshold/algorithm, corpus aggregation, explicit validation IDs/hash, final-test disabled |
| Runtime identity | clean source SHA and source bundle hash, local lock/environment fingerprint, remote Image/deployment identity, resolved remote package versions/CUDA/device details, schema version |
| Resource limits | approved GPU/device count, CPU/RAM, per-call startup/execution limits, absolute phase deadline, operation limit and total currency ceiling plus estimate provenance |

Real creation fails on missing required values, `unknown`/dirty code, mutable revisions,
unsupported score/backend combination or absent nonempty validation set. Define
`kind = real-ocr-simulated-annotation-v1` only for an explicitly real backend with verified
identity; keep `fixture-simulation-not-ocr-evidence` for fixture runs. This is an experiment-kind
marker, not an assertion of model quality or successful completion.

Proposed `SimulationBaseline` is separate from `SimulationRound`: base model ID, validation
predictions/metrics, zero labelled count, and operation/artifact references. Real configs require
`evaluate_base=True`. Add `load_base(*, experiment_id: str) -> str` on the real adapter, producing a checkpoint
identity without revealing labels or training. Extend `Prediction.round_number` to allow zero,
but permit zero only for baseline validation in Pipeline; preserve positive legacy-job/round
checks explicitly. `SimulationRound.number` remains positive.

`step_simulation` first validates identity and obtains/commits the baseline atomically if absent.
Only after that does it apply budget/round stopping. Thus zero-budget real runs produce baseline
metrics, no `fit` or acquisition calls, and complete; fixture zero-budget behavior remains unchanged.
Baseline predictions never seed acquisition. First acquisition remains seeded random using the
current sorted membership and `seed + completed_round_count` rule.

### 3. Model and prediction contracts

Keep `SimulationModel` synchronous. To make fitting ownership explicit, propose adding
`experiment_id` and `round_number` keyword arguments to `fit` alongside `seed`:

```python
fit(examples: tuple[RevealedExample, ...], *, seed: int,
    experiment_id: str, round_number: int) -> str
predict(pages: tuple[SimulationPage, ...], *, experiment_id: str,
        round_number: int, model_id: str) -> tuple[Prediction, ...]
```

Update FixtureModel/test adapters mechanically; ownership must not affect fixture/random seeds.
The real adapter also exposes the owned `load_base` call and its concrete frozen recipe checked
by a simple constructor/factory. It returns only after remote output verification. Every predict call loads or
verifies exactly the supplied checkpoint, including after a fresh process/container start.

The proposed `ModalSimulationModel` maps `SimulationPage` to a strict `RemotePage` containing
only `id`, `document_id`, `width`, `height`, `image_sha256`, and safe image key. Strip
`source_image`, host `image_uri`, arbitrary filesystem paths and truth-bearing fields. A
predict request has no `regions` field; schema uses `extra=forbid`. Fit serializes only the
cumulative `RevealedExample` set returned by `oracle.reveal`, with TRAIN-only and uniqueness
checks. Hidden-label/whole-manifest hashes remain local; do not use them as seeds or model input.

Use one typed `RemoteRequest` with a discriminated `op = load_base | fit | predict`, and
`RemoteResult`, defined in `models.py`:

| Envelope | Content |
| --- | --- |
| Common request | schema, operation key, run UUID, round (0 only baseline), approved code/recipe identity, resource/deadline limit, attempt ID; owner fields never become model features |
| load_base | pinned base/processor/checksum recipe only |
| fit | ordered selected TRAIN examples, selected-truth digest, recipe, seed and pinned base ID; no validation/test/unselected targets |
| predict | image-only pages, exact model ID, decode/score recipe and purpose `pool` or `validation`; no targets/crops from truth |
| Success | echoed identity/ownership, verified model ID or exact prediction coverage, artifact checksums, elapsed time/device/peak VRAM, finite loss/update summary for fit |
| Failure | categorized code (`input`, `identity`, `oom`, `timeout`, `model`, `storage`, `generation`), safe message, terminal/reconciliation state; never traceback dumps of labels/secrets |

Operation keys hash canonical sorted-key JSON of the inputs actually consumed, recipe/code,
owner and operation; request ordering is explicit. Training RNG depends on the paired seed and
training recipe, not operation key/run UUID. Resource/attempt bookkeeping is separate from the
semantic fit digest so reconciliation cannot silently change the learned model.

`model_id` is `checkpoint:sha256:<canonical-manifest-hash>` resolving through a verified
manifest, never an arbitrary supplied path. A base manifest has base/processor revisions and
file checksums; a fitted manifest additionally has selected IDs/truth digest, fit recipe/seed,
actual optimizer steps/trainable parameter names, adapter/full-weight file hashes, preprocessing
identity and completed status. Hash canonical relative keys/bytes, not machine paths. Only
complete verified checkpoints are loadable; partially written attempts are not candidates.

Output region coordinates are original-page pixels with finite positive extents and bounds;
IDs are unique within a page. Return regions in explicit predicted reading order. Resize/pad/
tile transforms are inverted from saved transforms, never from GT. Training and prediction use
the identical accepted processor/prompt/schema path. `parse_output` must perform complete
schema/geometry/coverage checks, not merely identify JSON.

Add a small generation status to real predictions: `ok | invalid_output | truncated`, plus an
artifact reference for raw output/finish reason. On random validation, invalid/truncated output
is retained, counted as a failure and scored as empty text (no silently dropped pages); never
reinterpret parser failure as a successful blank page. An actual blank OCR result may be `ok`
with empty regions. Scored acquisition fails closed on unusable score/output; do not invent a
score or silently switch to random. Missing/wrong-owner/duplicate pages are contract errors and
prevent round commit. The smoke requires all requested outputs structurally valid before pilot.

Model choice is pending. For a page-generative model, supervise the accepted complete page
response from revealed lines only. For a line recognizer, hidden GT line boxes are forbidden
even to create inference crops. It needs an image-only detector with its own pinned provenance,
or an explicit change to a known-layout/transcription task accepted by Manager/user. Never wrap
a page transcript in a fabricated full-page line box to pass the schema. If Research recommends
text-only page OCR, revise the output contract/evaluation and obtain a method decision first.

### 4. Real training and checkpoint correctness

Each fit reloads the same pinned base and fresh trainable adapter/optimizer/scheduler, then
trains on exactly the cumulative revealed pages. Previous round weights are not the next
round's starting point. No validation early stopping, best-checkpoint selection or unseen-label
training; select the terminal step fixed in the recipe. Equal-budget strategies share the same
recipe and training-page count; record target tokens/optimizer steps because page lengths differ.

Conditional candidate recipe for a feasible generative model: LoRA rank 8/alpha 16/dropout 0,
batch 1, accumulation 4, one epoch per fit, AdamW lr `1e-4`, constant schedule, warmup 0,
weight decay 0, gradient clip 1, bf16 if supported, frozen vision encoder. Target modules must
be explicit verified language-attention names after model acceptance; do not apply a blanket
suffix matcher to the vision tower. These are PROPOSALS, not measured stable hyperparameters;
a simpler recognizer needs its own recipe. Pin and test the exact library/model combination.

Supervised loss uses target response tokens only: ignore prompt, image/control and padding
tokens; padding positions use `-100`. Include the agreed response stop token. For structured
page generation, the train target includes required text and coordinate serialization; scoring
text-only tokens is a separate later policy. Verify labels by decoding a synthetic batch and
inspecting the mask. Never train on validation crops or use true pool line count to configure
inference. Keep blank-page targets deliberate and source illegibility behavior versioned.

Declare image/sequence caps before the run. For Qwen-sized feasibility only, initial proposed
caps are 1024-pixel long side, batch 1 and 2048 generated tokens; target-token limit 4096 must
be checked with image+prompt expansion. These may lose detail or truncate dense historical
pages and are not scientific defaults. Calibrate on separate train development pages. If a
page exceeds caps, stop/record the failure; do not drop it or crop with hidden annotations.
Changing caps after calibration produces a new frozen recipe/run.

Prove training by finite nonzero loss on supervised tokens, finite gradients, recorded positive
optimizer steps and at least one intended trainable tensor changed. Prove reset by comparing
initial base/adapter hashes on every fit. Save then load in a fresh process and verify tensor
hashes plus deterministic decoded output on a fixed train-only probe. Repeated same-recipe fits
need empirically declared tolerance; GPU nondeterminism is disclosed, not called bitwise resume.

### 5. Modal boundary and data access

Proposed new `integrations/modal_model.py` owns the local adapter and request reconciliation;
new `entrypoints/modal_app.py` owns one Function operation dispatcher. Model logic remains in
`integrations/qwen.py` if Qwen is selected (otherwise one chosen-model module, not both).
The local orchestrator waits synchronously; it does not need an HTTP endpoint.

One Volume layout: `/images/<sha>`, `/base/<revision>/...`,
`/checkpoints/<attempt>/...`, `/outputs/<operation>/<attempt>/...`. All finalized assets have
manifests/checksums. Selected labels exist only in fit arguments and ephemeral working memory;
never save trainer datasets, target-containing caches, full prompts or example transcripts to
the Volume/logs. Checkpoint weights may learn revealed text, which is intended. Local oracle
truth, validation metrics/labels and final-test assets are not mounted. Build the Modal Image
from an explicit code/dependency allowlist, never the repository root or `.local`.

Use fixed safe image keys resolved beneath approved mount roots; reject absolute paths,
`..`, escaping symlinks, mismatched hashes and dimensions. Verify request image IDs against
its declared allowlist before model access. For the single dispatcher, keep immutable image/base
subtrees read-only where supported and writes confined to derived outputs. Mount restrictions
are established at Function/container configuration, not dynamically by a payload flag.
If different per-operation OS permissions become necessary, split into two functions only after
showing the need; selected-only payloads and absence of GT from all mounts remain mandatory.

Modal documents read-only/subdirectory Volume mount options; Platform reports SDK 1.5.5
signature support, while cloud enforcement/runtime checks remain pending. Close files before refresh; commit completed artifacts
before reporting success. Filesystem visibility and concurrent writes need explicit handling,
not an assumption that returning a filename publishes bytes. These requirements follow
[Volume mount API](https://modal.com/docs/sdk/py/latest/Volume) and
[Volume persistence guidance](https://modal.com/docs/guide/volumes), read 2026-09-14.

Use one active orchestrator and one active GPU input/container per approved pilot; no `.map`
fanout or concurrent adapter calls. Explicit `max_containers=1`, `min_containers=0` and no
input-concurrency expansion bound the proposed deployment. Scaling limits are not a monetary
ceiling; confirm settings against the pinned SDK and measure cold-start/idle charges.
See [Modal scaling](https://modal.com/docs/guide/scale).

### 6. Local journal, crash reconciliation and resume

Use existing SQLite `records` with kind `model-operation`, not another store/service.
Proposed `RemoteOperation` holds semantic request hash, phase/owner, attempt ID, state,
FunctionCall ID when known, receipt/artifact hashes, absolute deadline, estimated/reserved cost
and terminal error. Record fit completion separately from the atomic scientific round.

| Interruption | Required next behavior |
| --- | --- |
| Before journal reservation | No request sent; normal retry. |
| Reserved but submission/ID outcome unknown | Mark/retain `submit_unknown`; stop. Reconcile receipts and provider call inventory, then explicitly resolve. Never blind-resubmit a potentially billable fit. |
| Call ID persisted, local process gone | Reattach to that call and inspect terminal state; wait does not launch a replacement. |
| Worker finishes, response missing | Read verified receipt/checkpoint for the same request and attempt; reuse completed result. |
| Fit finished, prediction/evaluator fails | Reuse fitted checkpoint and any complete prediction receipt; retry only the failed stage within remaining approval. No new reveal count committed. |
| Partial checkpoint / corrupt receipt | Quarantine attempt logically; refuse loading; preserve evidence, and request explicit bounded retry if needed. |
| Round committed, caller lost response | Reload run and journal; return durable round, do not retrain it. |
| Stale second local caller | Compare-and-swap journal reservation fails before submit; reload. Whole-round CAS remains final guard. |

Adapter uses `.spawn()` solely to obtain and persist `FunctionCall.object_id`, then waits with
bounded `.get()` calls. `FunctionCall.from_id` restores a known call on restart. A client wait
timeout alone is not evidence of remote cancellation. See the
[FunctionCall API](https://modal.com/docs/sdk/py/latest/FunctionCall), read 2026-09-14.

Worker verifies request identity, checks a completed receipt before work, writes an attempt
checkpoint/result then its completion manifest, and commits before returning. Single-writer
operation ownership and immutable attempt outputs are required; a Volume is not a lock service.
Uncertain external execution cannot be made exactly once by local CAS. Operator reconciliation
is an intentional bounded failure mode, preferable to extra infrastructure or silent duplicate cost.

Set application retries to zero for the initial smoke; catch expected OOM/data failures as
terminal results. This does NOT eliminate platform rescheduling after container crashes:
[Modal retries](https://modal.com/docs/guide/retries) documents that behavior. Every attempt
checks a persisted absolute deadline before training and between steps; limit total phase time,
not just successful calls. Platform must verify a controllable stop/cancel path for startup
crash loops before a paid release. If this cannot be demonstrated, the paid phase stays blocked.

An execution timeout is per attempt and excludes scheduling; configure a startup limit too.
After deadline, cancel the known call and verify terminal state; allow for shutdown/accounting
latency in the estimate. These are not exact invoice guarantees. See
[Modal timeouts](https://modal.com/docs/guide/timeouts), read 2026-09-14.

Before real resume, compare frozen source/GT/images/config as today plus code, local lock,
model/processor revisions, adapter/evaluator/score/preprocessing policies and remote deployment
image/environment fingerprints. Each worker response echoes the actual identity. Refuse any
mismatch before fit or metric publication. Resume only with the original environment or create
a new run; no `--ignore-identity` path in this slice. A restored checkpoint skips completed work;
interrupted within-fit optimizer/RNG-state continuation is deferred—an approved failed attempt
restarts from base and is recorded, not described as exact continuation.

### 7. Evaluation and score policy

Proposed evaluator identifier: `page-text-layout-v1`; acceptance pending Research. Raw source
labels remain unchanged. Evaluation creates a temporary view: Unicode NFC and CRLF-to-LF,
preserve case/punctuation/spacing, join ordered line texts with one newline; no GT-crop OCR.
Source illegibility markers remain literal until a different explicit policy is accepted.
Record counts of illegible/blank/missing labels. Do not silently apply current `Region`'s
`[ILLEGIBLE]` insertion to source truth; `SourceRegion` preserves the source.

Primary proposed corpus CER = sum of per-page character edit distances divided by sum of
reference code points, not mean of page CERs. WER uses the same normalized page strings,
Unicode-whitespace tokenization, and sum token edits / sum reference tokens. Empty reference
pages contribute insertions to numerators; both-empty contributes zero. An all-empty validation
reference makes CER/WER undefined and fails evaluator eligibility rather than reporting zero.
Metrics may exceed 1. Current `character_error_rate` empty-string shortcut is insufficient for
this corpus aggregation; add/reuse an edit-count helper, not reverse its normalized result.

Evaluate all fixed validation pages; invalid/truncated OCR counts as failed/empty prediction,
plus report invalid-output/truncation rates. Never retain only successful pages. Ordered page
text penalizes missed/extra lines without giving the recognizer GT line alignment. For layout
diagnostics, form IoU pairs, sort by decreasing IoU with stable predicted/reference ID ties,
greedily match unused pairs with IoU >= 0.5. Report matched count, precision/recall/F1 and mean
matched IoU; explicitly label this a deterministic greedy rule, not optimal assignment.
Missed reference and extra predicted lines remain in denominators. If both sets are empty,
report counts and omit page-level ratio averaging; aggregate corpus counts, with undefined
all-empty layout rates represented by omitted keys plus counts. Do not return NaN to the current
finite-float evaluator API. Matching is diagnostic and does not reorder model input or mask CER.

The first real pilot has `score=none`, random strategy, null confidence/entropy, pool reporting
off. Validation still predicts. Do not compute a confidence from teacher forcing against pool GT.

One possible later score, requiring RES-002 acceptance and tokenizer verification: for generated
transcription token positions T only, use unfiltered next-token model distributions at temperature
1. Confidence `exp(sum(log p(y_t))/|T|)`; entropy
`sum(-sum_v p_t(v) log p_t(v))/ (|T| log V)` for vocabulary size V > 1. Exclude prompt, image,
padding, EOS/control, JSON keys/coordinates and syntax. Define character spans of generated text
values, map to tokens with tested offsets, and exclude boundary tokens that cross syntax/text;
record included counts. Aggregate all included tokens on the page equally (length/layout bias is
an explicit confound); do not average per-line values without a revised version. Empty T, invalid
schema, nonfinite distributions or truncation fail score eligibility. Computing full distributions
may increase memory; no retention of full logits required. This proposal is not enabled until
Research confirms masking/aggregation, numerical tests verify [0,1], and measured overhead fits
the approved resources. Keep selectors unchanged; absent scores fail their existing contract.

### 8. Ownership and implementation slices

All files below are proposed changes, not completed work. Manager assigns task IDs and exact
ownership before dispatch; no overlapping writes to `models.py`, Pipeline or dependencies.

| Ordered slice | Owner / proposed files and functions | Exit gate |
| --- | --- | --- |
| A. Asset decision and normalization | Research recommendation; Platform download/provenance. Development new `integrations/public_dataset.py`, source-fixture tests | Accepted license/revision/split/order, exact counts/hash mapping; faithful import and leakage audit. Bulk transfer only after asset release. |
| B. Real identity, baseline and evaluator | Development `models.py`, `pipeline.py:create/step/_validate_simulation_predictions/export_simulation`, `simulation.py:SimulationModel/FixtureModel/runtime_identity`, `evaluation.py`, simulation tests | Offline zero-budget and real-kind tests; metric golden cases; no oracle reveal at baseline; legacy fixtures unchanged. |
| C. Page OCR and model artifact | Development chosen-model integration (`qwen.py` if selected), model tests | Offline parser/preprocessing tests; pinned load/predict contract and trained-target masking tests. Real quality still unverified. |
| D. Modal transport and recovery | Platform owns new `entrypoints/modal_app.py`; Development owns new `integrations/modal_model.py`, `models.py` operation records and tests. Development alone owns `pyproject.toml`/locks after agreed Platform requirements | Fake transport covers every journal failure row and payload/mount audit. Platform confirms SDK/image/limits/auth. Independent review before paid call. |
| E. Base smoke and real fit | Development implements train/save/load in chosen integration; Platform executes approved calibration; Testing reviews exact candidate | Real base output, loss/gradient/update and fresh checkpoint reload evidence under cost cap. No pilot if output task/geometry invalid. |
| F. CLI and complete random pilot | Development `entrypoints/cli.py`, new explicit real example YAML, `docs/simulation.md`, new small result export/plot helper under `scripts/` | Explicit backend dispatch, identity-safe resume, genuine baseline+round metrics, resource accounting and reproducible exports. |
| G. Paired comparison | Research final score/method; Development score extraction only if accepted; Testing verifies; Platform executes reviewed runs | Equal budgets/seeds/recipe, required source/method decisions, failures included; Research interprets without demanding a win. |

Prefer small independently reviewable Development candidates. Slice B can use fakes without a
GPU; real model implementation waits for A and accepted Research inputs. Slice D consumes B/C
schemas without changing them concurrently. Resource image pins are agreed with Platform but
committed by the designated file owner. No task release is implied by this table.

CLI proposal preserves `simulation fixture/start/run/resume/status/export`; add explicit
`--real-config PATH` on start and reconstruct the frozen selected adapter on run/resume.
Missing Modal SDK/auth is an actionable error before paid submit, never a fixture fallback.
Add a read-only `simulation preflight` producing resolved config, source/image counts, revision
checks and cost proposal; add an explicit operation reconciliation command only for journal
unknown outcomes. The stored recipe governs resume, not changed CLI defaults.

## Experiment and resources

### Staged workload proposal

All counts below are bounded proposals, not existing data or permission to repartition it.
Choose document-preserving subsets without labels/model scores; if exact counts cannot be
formed from accepted source groups, stop and revise counts explicitly. Subsets must contain
available labelled TRAIN/VALIDATION pages; final-test pages stay unused. Calibration uses a
separate train development document pool excluded from pilot acquisition; record its IDs and
separate engineering label exposure. Calibration checkpoints never initialize the pilot.

| Stage | Pages / rounds / seeds | Work and purpose |
| --- | --- | --- |
| Offline contract rehearsal | Existing synthetic fixtures | No cloud; errors, payload/GT isolation, identity/resume and zero-budget cases. |
| Authorized calibration smoke | 4 train development pages, 2 validation pages; seed 824; 2 acquisitions of 2 pages; random; no test | Base validation on 2, reset-fit on 2 then 4, validate both; total 6 validation page predictions, 2 fits, plus 2 fixed train-probe predictions for save/load comparison. Hard cap 10 optimizer updates per fit; record actual epochs/steps, not a convergence result. |
| Random engineering pilot | 24 train acquisition pages, 6 validation pages; seeds 824/825/826; batch 4, budget 12, 3 rounds | 9 fits total at cumulative 4/8/12; base + 3 validation passes per seed = 72 validation page predictions. No pool prediction. Recipe frozen after calibration; no within-pilot tuning. This is not an adequately trained comparison by default. |
| Optional comparator rehearsal | Same accepted 24/6 subset, seeds 824/825/826, batch/budget/rounds and frozen recipe | Another 9 fits/72 validation predictions; if scoring after all rounds, 20+16+12 = 48 pool predictions per seed, 144 total. Verify score plumbing/fairness; no AL conclusion. Omit terminal pool pass only as an explicit tested optimization applied consistently. |

Reference workload bounds assume unique checkpoint fits per seed/strategy, no caching discount,
no automatic retry and sequential execution. With one epoch and batch 1/accumulation 4 on a
page-generative model, the pilot has 1/2/3 optimizer updates per fit and 18 across three seeds;
framework treatment of final partial batches must be fixed. A line-based architecture changes
this calculation and needs a new explicit step budget before approval.

Start random only. Add comparison after score acceptance and random pilot recovery evidence.
Three seeds are a pilot, not strong population-level statistical evidence. Share fixed initial
batch membership across strategies; final data/model recommendations and exact revisions remain
pending. No validation-driven acquisition, stopping or hyperparameter choice within paired runs.

Scale-up gate: after engineering acceptance, use a separate train development subset to measure
loss/update curves, target coverage, runtime and validation behavior for increasing fixed training
durations under a capped calibration budget. Research/Planning must explain why the chosen
duration is adequate (or openly characterize undertraining), based on learning curves rather
than the presence of one nonzero gradient. Freeze the resulting epoch/step recipe for every
strategy at a given page budget; checkpoint selection remains terminal-step, no per-strategy
early stopping. This calibration is additional work, separately estimated/approved, and its
labels/checkpoints never initialize the comparison. Engineering metrics remain labelled as such.

Concrete later comparison proposal, conditional on enough document-consistent pages: 100 train
pool pages, 20 validation pages, batch 10, budget 50, five rounds, paired seeds 824–828. Two
strategies mean 50 fits, 1200 validation page predictions including each baseline, and 1750
scored-pool page predictions if terminal scoring is retained (90+80+70+60+50 per scored seed).
For calibrated E epochs and page batches of 1 with accumulation 4, quote actual implementation
step rounding and observed token lengths at budgets 10/20/30/40/50; do not extrapolate the
engineering pilot into this cost. Verify/approve this separate recipe, subset and cost envelope
before release. If available source groups cannot support these exact counts, revise the proposal
explicitly. These data can support a scoped report with limitations; five seeds and small data do
not guarantee significance or generalization. Full-corpus or replication work remains later.

### Memory, runtime and cost bounds

ESTIMATE, conditional on a 4-billion-parameter candidate: two-byte base weights alone require
about 8 GB decimal, before activations/KV cache/workspaces/optimizer/adapters. This arithmetic
does not establish fit on a 16/24 GB GPU. LoRA reduces trainable-state cost but not base weights
or all activations. Platform supplies model-specific peak allocation/reservation and appropriate
GPU/CPU/RAM/disk; chosen GPU, usable VRAM, training time and approved currency ceiling remain UNKNOWN.
No GPU selection is accepted from the configured Qwen name alone.

Proposed limits for pricing: one GPU, one active call; calibration execution cap 10 minutes
per operation, startup cap 5 minutes and whole calibration deadline 45 minutes. No automatic
retry, OOM-based upsizing or spending beyond the approved phase. These limits are deliberately
conservative caps, not expected times or spend authorization. No new expensive call when its
worst-case reservation exceeds the remaining cap. Whole-call fit must be interruptible/check
deadlines between training steps. Any required larger cap goes back for review before launch.

Platform's quote must separately include billed GPU seconds, CPU core-seconds, memory GiB-seconds,
startup/idle tails, storage GiB-days/retention and any transfer/build costs at dated official rates.
For measured per-fit times F2/F4 in calibration, price the pilot from measured throughput and
accepted page/token distribution, not linear extrapolation alone. Use a conservative factor
(proposed 2x measured work), explicitly add cold starts/checkpoint I/O, and cap calls/deadline.
Report `estimate = sum(resource_time × verified_rate) + storage + transfer + contingency`,
with every quantity/unit/source visible; dollar total remains pending Platform evidence and user
ceiling. Function timeouts alone do not cap invoice or repeated crash retries.

Pricing basis, checked 2026-09-14: L4 `$0.000222/s`, A10 `$0.000306/s`, L40S `$0.000542/s`,
CPU `$0.0000131/physical-core/s` (2 vCPU), RAM `$0.00000222/GiB/s`, Volume `$0.09/GiB/month`.
Source: [Modal pricing](https://modal.com/pricing). Ignore account allowances/credits in estimates.
Platform's proposed fixed smoke envelope is one L40S, 2 physical cores and 32 GiB RAM:
`$2.301264/hour` at those rates. The proposed 45-minute total billed-compute envelope gives
`$1.73`; add Platform's illustrative CPU build `$0.053`, storage/transfer and contingency.
A 20 GiB Volume for 7 days adds approximately `$0.42` at 30-day proration before allowances.
Platform proposes a **USD 5 gross smoke ceiling**, unapproved. Its **USD 20 pilot proposal**
assumes 2–4 GPU hours (compute `$4.60–9.21`); it is not yet costed for this plan's larger
comparison. These are estimates, not measured training times or an assurance of L40S fit.
Requote from calibration before scaling; neither ceiling authorizes execution or automatic
upsizing. Check input CPU/RAM units and hard limits in the pinned SDK at implementation.

Successful calibration requires margin below the selected device's usable memory and permitted
time; reserve proposed 20% VRAM headroom before pilot sizing. On OOM, quota/auth error, repeated
restart, invalid output, checksum mismatch, or exhausted deadline/budget, stop and preserve the
attempt. Do not automatically switch model, precision, token/image limits or dataset pages.

### Results and interpretation

Export new-directory JSON plus flat per-run/per-checkpoint CSV with run kind, dataset/subset/
split hashes, model/processor/base/checkpoint IDs, code/environment/recipe identity, strategy,
seed, baseline/round, revealed pages/IDs, cumulative training tokens/optimizer steps, CER/WER,
layout counts/metrics, invalid/truncated counts, fit/predict time, peak VRAM, operation IDs,
cost estimate and observed billed cost when available. Human annotation seconds stay null.
Export raw OCR and artifact hashes for reproducibility; publish only license-permitted small
derived examples and summaries, never a dump of oracle labels or local operational records.

Plot engineering CER (lower is better) against 0/4/8/12 revealed pages. Show each seed and mean with sample
standard deviation (n=3); label the small sample. The later proposed comparison uses 0/10/20/30/40/50
and n=5. If comparing, pair each seed at each budget.
Report normalized area under the error curve over [0,12] using the existing trapezoidal helper
divided by 12 (or [0,50] divided by 50 for the later proposal); lower is better. Do not average areas on missing/misaligned budgets or treat
failed runs as zero error. Report attempted/completed/failed counts and reasons; reruns require
the same recipe or are a separately versioned experiment. Threshold-budget savings can only
be claimed for a preregistered threshold reached within observed budgets, without extrapolation.

Final-test evaluation is not part of this pilot. After methods/configurations/checkpoints and
selection rules are frozen, obtain explicit methodology approval for a separate evaluator-only
test run. Do not use that test result to choose seeds, thresholds, checkpoints or strategies.
Potential base-model pretraining contamination of a public corpus remains a disclosed unknown.

## Verification and acceptance

### Trace a representative page

Illustrative page `corpus:doc-A:p3` is an accepted TRAIN record with a verified image hash,
original dimensions and local ordered source line labels. Normalizer records its source mapping;
oracle freezes bytes. Only its `RemotePage` image metadata may enter base/pool prediction.
With seed 824, the existing selector chooses its ID from train membership; only then does
`reveal` include its source lines in the cumulative fit tuple. Fit request hashes the selected
targets/recipe, journals the operation, obtains the Modal call ID, resets the base, trains,
and returns a verified checkpoint manifest. A new container predicts a fixed validation page
from its image and that checkpoint; it never receives validation line boxes. Geometry is mapped
back to original pixels; local evaluator alone joins source reference lines and computes metrics.
After identity/data recheck, the round commits once. Resume reconciles the same checkpoint and
receipts before deciding whether more work is necessary. This is an interface walkthrough,
not a claim that this real page or run already exists.

### Required evidence per boundary

| Boundary | Test / acceptance evidence |
| --- | --- |
| Source fidelity | Tiny source-format fixture with real parser semantics: verbatim text/flags/order, polygon projection audit, dimensions/hash mismatch, duplicates and document split violations; no silent repartition. |
| Truth isolation | Perturb hidden/validation truth in separate synthetic runs: same training membership/seed/revealed labels must produce same fit payload and acquisition; validation changes only metrics. Payload and built Image/Volume inventory contain no full GT/source/DB or hidden GT crops. |
| Baseline/kind | Zero budget still evaluates base once with no oracle reveal/fit; first acquired IDs unchanged with baseline enabled; fixture remains visibly fixture; no fallback on missing real dependencies. |
| Training | Loss masks decoded and checked; only selected examples; no validation callbacks; reset initial hashes; nonzero updates; fresh load tensor hashes and fixed-probe decoding agree within declared precision policy. |
| Predictions | Golden resize/pad/box transforms; invalid JSON/IDs/order/bounds/NaN/truncation; exact run/model/round/page coverage; valid blank output distinguished from parse failure; line recognizer never sees GT inference boxes. |
| Metrics | Hand-computed insertion/deletion/substitution cases, blank/nonblank, unequal reference lengths proving corpus rather than macro CER, Unicode/whitespace/order cases, unmatched/duplicate lines and all-empty denominator policy. |
| Scores (later) | Analytic uniform/one-hot distributions, stable log math, finite [0,1], tested token spans, no prompt/GT/coordinate tokens, empty/truncated rejection; recorded score overhead and a defined rank/tie result. |
| Remote journal | Inject each interruption from section 6, including lost ID, completed fit plus failed evaluator, committed round plus lost response, corrupt checkpoint and stale writer before submit. Prove no blind resubmit on unknown outcome. |
| Identity | Change code/package lock/model revision/prompt/processor/evaluator/remote Image and confirm resume refuses before billable work; identical identity allows separately restarted CLI execution. |
| Cloud/resource smoke | Exact reviewed SHA, deployed image ID, mount inventory, real device/peak VRAM/time/loss/update/outputs, cost/attempt count, verified cancellation/deadline behavior and fresh checkpoint reload. Fakes cannot satisfy this row. |
| End-to-end results | Baseline + every committed budget present, selected labels counted once, no test access, failed rows disclosed, JSON/CSV/curve agree, separate process resume agrees with receipts; independent Testing reviews exact candidate. |

Run the relevant new tests and full current suite plus Ruff; no configured type checker was
found in inspected configuration. The existing 63-test independent review is historical evidence
for the simulation candidate, not a PASS on these proposed changes. Independent Testing must
review each significant implementation candidate and the real-smoke evidence before pilot release.

Phase-A completion means this bounded draft is available for review. Implementation acceptance
requires accepted Research/Platform inputs, exact file ownership/slice release, and independent
code evidence. Paid release additionally requires assets, method/identity/resource recipe, user
spend ceiling and tested stop/recovery controls. Result delivery means verified real base and
adapted predictions plus reproducible validation table/curve; it does not require an AL gain.

### Decisions still required for version 0.2

| Pending decision/evidence | Owner / consequence |
| --- | --- |
| First dataset/model pair, immutable revisions/licenses/downloads and document/split/order semantics | Research + Platform, Manager acceptance; no real parser/model selection assumed. |
| Full page boxes+text versus detector+recognizer versus explicitly changed text-only task | Research + Manager/user method decision; cannot conceal GT-crop advantage. |
| Training/processor limits, target serialization, normalization/matching and later score mask | Research + Planning review; version/freeze before any paired run. |
| Installed Modal SDK/auth/permissions, mount options, Image pinning, GPU and cancellation | Platform evidence; unavailable capability is a blocker, not inferred from plugin installation. |
| Smoke/pilot subset manifests, calibration exclusions, exact prices/currency ceiling and deadline | Platform proposal + explicit required approval; counts here do not authorize split changes/spend. |
| Exact real-code candidate, independent review and Manager release | Development/Testing/Manager; this document alone does not release work. |

Do not wait indefinitely to publish phase A. Preserve this version and incorporate accepted
input in a reviewed successor with a change record; never silently rewrite a running protocol.
