# Plan — real OCR with simulated annotation on Modal

Owner: Planning. Version: **0.4**, 2026-09-16. Status: **Qwen specification proposed; converter specification accepted; candidate 1A completed**.
Manager reviewed v0.2 at `d659c48f38639e54e19ac6aceff1b5adf853355a`
and accepts its roadmap and bounded local candidate 1A under the user's engineering authorization.
Candidate 1A at `dee5e1a5ecbbc326921e0f4019792c8f7e5deab2` passed independent Testing;
Manager accepted and integrated it with review evidence `2db380a497ca641d8e1df0cda773074e0d306040`.
See the [exact review and limitations](../docs/verification/local-contract-review.md). Engineering-only
READ source use is now user-approved; Manager accepted the exact converter specification at
`653fe7dff2018de1e35c91d616c24ed848681a16` and releases candidate 1B for implementation/review. Later implementation, methodology and
execution retain their stated gates.
Phase-A v0.1 remains in Git history. Real model execution is still unavailable.

Inspected control for v0.4: `fad7cae5baf5d7e30ecf9a772152363ab9ca6902`, including accepted
local contracts, converter specification and two-Volume runtime correction. Earlier scopes below
are retained unchanged; the new proposal is candidate 2 only and awaits Manager acceptance.
Scientific input: accepted [RES-002](../research/RES-002-real-ocr-pilot.md) at
`8a173197fd4ef12f91da3fde0d0197bc4bda21b6`. Development, independent Testing and QA
preparation informed this reconciliation; they are not verdicts on a real implementation.

## Goal, decisions and current evidence

Build on one Pipeline, the source-label oracle and existing SQLite. Implement three sequential
candidates: **local contracts → Qwen → Modal and CLI**. Source conversion is a separate gated
part of candidate 1; it must not hold up records, baseline and metric work on synthetic data.
Use plain records/functions, one concrete recipe and direct equality checks. No registry,
capability framework, service, extra database, queue or automatic model/GPU fallback.

The first real result is pinned base and adapted page OCR with validation metrics and verified
checkpoint/recovery evidence. It is not necessarily useful OCR or evidence of an AL advantage.
The later falsifiable hypothesis is that an accepted uncertainty strategy lowers validation error
at matched revealed-page budgets versus random with paired seeds and the same reset-fit recipe.
A null/worse result is admissible. Revealed pages are simulated annotation budget, not human time.

### What v0.2 resolves

| Issue in v0.1 / differing proposals | Single v0.2 decision |
| --- | --- |
| Baseline could share a step with acquisition; positive rounds were described as already constrained | Commit baseline in its own step and return; add an explicit positive `SimulationRound.number` constraint. Prediction purpose determines whether zero is legal. |
| Remote CUDA/device/peak-memory identity required before a first run | Freeze expected code/dependency/model/build identities without GPU work. Observed runtime telemetry is separate and never equality-compared as an identity. |
| Seven overlapping workstreams | Three sequential candidates, with a small exact local slice 1 below. PAGE conversion, real model and remote work have their own gates. |
| LoRA r8/alpha16/one epoch versus Research r16/alpha32/three epochs | Adopt r16/alpha32/three epochs as the sole future calibration proposal, unmeasured. Explicit text q/v module names remain a candidate-2 prerequisite. |
| Several incompatible page schedules | One stage table below: 2-page smoke; 10/20 random engineering pilot; conditional 20/40/60 paired comparison. Other v0.1/Platform counts are superseded as execution proposals. |
| Greedy versus maximum-IoU layout matching | No layout scoring in slice 1. Later use the precisely defined maximum-total-IoU rule below; remove greedy as a default. |
| Reject versus represent all-empty reference metrics | Keep edit counts; omit each undefined rate independently and emit numeric defined flags. Never store NaN or invent a rate. |
| Character offsets for uncertainty | Later original-token serialized-byte-span rule from Research, not decode/retokenize; random/score=none now. |
| Dataset metadata treated as still entirely unknown | Archive/structure evidence is now available. Document independence and a few semantic mappings remain unresolved; no fabricated document IDs. |

### Inspected implementation

Candidate 1A is implemented and independently passed: separate baseline, positive acquired rounds,
prediction purpose/failure checks, expected-identity guards and page-text counts. The review above
records its exact scope and limitations; integrated main has 237 passing tests per Manager evidence.
`SourcePage` still inherits required `Page.document_id`; `_read_source` rejects cross-split document
IDs/image hashes; freeze/reload hashes manifests/images but has no source admission policy or XML
provenance. Legacy `local_data.py` hashes known documents to splits and must remain unchanged.
Qwen/Modal execution and the READ converter remain missing. Planning inspected current models,
oracle, coordinator, importer, tests/config and actual source. This plan-only turn ran no application
suite, conversion or model/cloud workload.

### READ2016 asset evidence and the grouping boundary

The selected staging pair is [READ2016 1.2.0](https://zenodo.org/records/1297399), CC BY 4.0,
and Qwen/Qwen3-VL-4B-Instruct / processor at
`ebb281ec70b05090aa6165b016eac8ec08e71b17`, Apache-2.0. Asset selection is established;
loadability, model quality and scientific suitability are not.

Platform's actual archive audit records 493,223,531 archive bytes, publisher MD5
`654f2d2c62055f1847f65ba83bd5d744`, SHA-256
`f4748c58af757e06804e638a6e84c2150daab19f50d55e10aac3115e6bfc1756`, and 804 regular
extracted files totaling 499,592,094 bytes. All 400 archive symlinks were omitted after checking
that their image targets exist; originals remain unchanged. No test subtree was staged.
Planning read the actual archive/page audit records and checked XML multiplicity/metadata.
This is structural evidence, not an independent repeat of every download checksum test.

Completed staging provenance is published at
`6317fe0c26c008d8e853a1ca8e6df254a72cab5a:experiments/assets/read2016-qwen3vl4b.json`.
Both model shards match their expected hashes; the index/header audit covers 713 tensors.
Manager independently rehashed the archive and all 15 model/support files against that record.
No model was loaded or executed. The Apache-2.0 declaration comes from publisher metadata;
the separately retained standard license text is not a license file from the pinned repository.

| Observed property | Consequence |
| --- | --- |
| 350 TRAIN / 50 VALIDATION images/pages; 8,367 / 1,043 lines | Preserve official partitions and all source records; no implicit 70/10/20 split. |
| PAGE namespace `2013-07-15`; complete indexed region and line reading order recorded by Platform | Use `RegionRefIndexed` plus line `custom` reading-order indices, validate uniqueness/coverage; region index gaps are preserved, line indices are contiguous. XML sequence is not the semantic rule. |
| Every line has exactly one `TextEquiv` and one `Unicode` (Planning XML check) | Select that Unicode verbatim for this source version; reject future ambiguous alternatives instead of choosing by confidence. |
| Four empty Unicode lines: one TRAIN, three VALIDATION | Preserve empty strings. They are annotated empty lines, not missing annotations or blank pages. |
| Two TRAIN lines lack `Baseline`; coordinates and reading-order metadata exist | A polygon-envelope view need not fabricate baselines. Missing structure type is a separate metadata issue, not a reason to drop text. |
| No geometry issues or cross-split exact image duplicates/filename overlap in audit | Useful checks only; near-duplicate and source-document independence are not proved. |
| Both `doc.xml` exports have `docId=-1`, title `page`, split-local `pageNr` | These are placeholders. Filename sequence, page number and one-ID-per-page are not document provenance. |

**Current disposition (v0.3):** the user has explicitly admitted these official partitions
with grouping unknown for engineering verification only. Candidate 1B below specifies nullable
simulation document IDs plus a narrow frozen policy; no placeholder IDs, silent leakage-check
waiver or final-test use. This successor resolves the prior source-representation decision for
review without changing original XML/images/split membership or claiming document independence.

## Approved first implementation scope — candidate 1A: local contracts only

Manager has released this bounded candidate under existing engineering authorization after
reviewing the exact successor and acceptance map. It is independent of Modal login, weights,
corpus conversion, document grouping, layout scoring and uncertainty research.

### Owned files and exclusions

Development is the sole writer to the following paths during candidate 1A:

| File / functions | Exact allowed change |
| --- | --- |
| `src/active_ocr/models.py` | Plain real-config/expected-identity/baseline records; explicit run kind, prediction purpose/failure status; positive acquired rounds. Compatible defaults for existing fixture data. |
| `src/active_ocr/integrations/simulation.py` | Small identity comparison and owned model signatures; fixture compatibility; `PageTextEvaluatorV1` using metric primitives; keep oracle reveal/split enforcement unchanged. |
| `src/active_ocr/pipeline.py`: `create_simulation`, `step_simulation`, `_simulation_stop`, `_validate_simulation_predictions`, `export_simulation` | Baseline-only atomic step, real/test-kind fail-closed adapter selection, purpose/ownership/identity checks and baseline export. |
| `src/active_ocr/evaluation.py` | Raw character/token edit counts and fixed text-view aggregation; keep existing per-string CER behavior for callers. No layout assignment or score extraction. |
| `tests/test_simulation.py`, new `tests/test_real_contract.py`, new `tests/test_evaluation_counts.py` | Synthetic/model-double coverage of the acceptance map below; no model download/import/network. |
| `docs/simulation.md` | Record exact new API/baseline/export behavior, fixture limitations and what is still unsupported. |

No edits to production Qwen, Modal/legacy GPU clients or entrypoints, storage implementation,
source converter, selectors, global Settings/default YAML, dependencies/lock or other roles'
artifacts. No real CLI backend or new commands yet; existing CLI continues fixture behavior and
uses extended status/export transparently. If an excluded file is demonstrably required, send
Manager a precise scope adjustment before editing; no speculative helper module/registry.

### Records and API semantics

Manager correction scope after independent review of candidate
`3c86330b82121bd86c7e6a9d926103d8d62b23f0` (2026-09-14): the candidate is not accepted.
The corrected candidate `dee5e1a5ecbbc326921e0f4019792c8f7e5deab2` subsequently passed these
checks and was accepted; the original FAIL remains in the review history.
Failed-status predictions must not persist nonfinite structured geometry that becomes JSON null
and prevents reload. Reject nonfinite coordinates atomically for every status; preserve valid
failed-output records and their raw-evidence references. The existing successful-layout checks
remain. Also permit a narrow purpose-only change to `Pipeline.poll_job` and `_predictions_for`:
legacy scoring must reject non-pool results before any publication, and acquisition reads must
exclude non-pool stored records. Missing purpose in historical records still defaults to pool.
Development may add focused coverage in `tests/test_pipeline.py`; independent review regressions
remain Testing-owned. This exception does not authorize other legacy service or storage work.
Re-review the exact corrected candidate, including full regression and the independent failure cases.

Keep `SimulationConfig` and `SimulationRun` as the run containers. Add optional `RealOCRConfig`
with explicit backend/recipe version, pinned model/processor revision, training/decode/evaluation
policy IDs and `ExpectedIdentity`. Use typed explicit fields, not arbitrary executable imports or
a generic capability system. Candidate 1A can represent recipes; it cannot execute Qwen/Modal.

`ExpectedIdentity` contains clean source SHA, canonical local dependency/version fingerprint,
model+processor commit/file-manifest identities and immutable recipe/evaluator versions. Future
Modal recipes also bind code-bundle/dependency/build-spec hashes. These are knowable without
executing a GPU. A resolved immutable deployment/image reference may be recorded from nonbillable
metadata when available; missing authentication must not prevent local contract testing. A
future paid submit requires the expected deployment/build identity to be resolved and frozen;
never mutate an existing run to substitute a new environment.

Actual device name, driver/CUDA observation, free/peak memory, elapsed time, call ID and billing
are **execution telemetry**, initially absent. Do not demand them at run creation or compare
memory/timing/device instance across resume. Verify actual code/packages/model/processor/build
against their frozen expectations before model work; check precision/device support as an
execution prerequisite, not equality to a previously measured GPU instance. No GPU discovery
call or prior successful model load is needed for local creation. All identity comparisons are
small explicit field comparisons; no identity service/framework.

Explicit run kinds: existing `fixture-simulation-not-ocr-evidence`,
`adapter-contract-test-not-ocr-evidence` for synthetic doubles, and
`real-ocr-simulated-annotation-v1` for a configured real backend. Validate kind/backend pairing;
never infer genuine OCR from a numeric metric or nonempty model ID. Test doubles exercise the
same real-contract branch but exports visibly remain contract tests. Default fixture creation/
execution remains unchanged. A missing explicitly requested adapter is an error, not FixtureModel
fallback. Production real construction remains unavailable until later candidate release.

Extend the existing model signatures with ownership/purpose (mechanical fixture/test updates):

```python
load_base(*, experiment_id: str) -> str  # real/test adapter only; no training
fit(examples, *, seed: int, experiment_id: str, round_number: int) -> str
predict(pages, *, experiment_id: str, round_number: int,
        model_id: str, purpose: PredictionPurpose = POOL) -> tuple[Prediction, ...]
```

Real/test adapters require ownership. FixtureModel may retain optional ownership keyword
defaults for existing direct callers; those values never enter its synthetic hash/seed. Pipeline
supplies explicit ownership for all new calls. No legacy fixture caller must pretend to own a
remote operation to keep running.

Add `PredictionPurpose = pool | validation | baseline_validation`; default `pool` preserves old
legacy predictions. Allow `Prediction.round_number >= 0` with invariant: baseline purpose iff
round is zero; other purposes require positive round. Explicitly add `SimulationRound.number >= 1`.
Existing legacy jobs are positive-round/pool; reject a baseline-purpose result there. Old fixture
validation records lacking purpose remain readable; validate purpose on newly produced records
and strict real-path boundaries, not by relabelling historical artifacts. No baseline/pool/round
validation result may be substituted for another purpose.

`SimulationBaseline` stores model ID, exact ordered validation predictions and metric counts/
rates, with fixed labelled_count=0. It is separate from `rounds`; no selected/revealed IDs.
Candidate 1A validates an immutable model-reference format in its real/test path; it does not
claim to verify actual weight bytes. Manifest creation/load verification belongs to candidate 2.
Fixture model identifiers/metrics remain compatible.

For real/test predictions add explicit `status = ok | invalid_output | truncated | refusal`
(default `ok`) and optional raw-output/finish-reason artifact reference; preserve raw evidence
in later real execution. Invalid/refused/truncated output is a failed prediction, never a valid
blank annotation. Its all-page text hypothesis is empty under this proposed engineering view;
report failure counts separately. This measures omissions, not a bound on hypothetical
hallucinations. Missing/wrong-owner/duplicate/extra pages are boundary errors and prevent commit.
Real/test `ok` geometry must be finite, positive, bounded by original dimensions and uniquely
identified, with order preserved from the model. Do not normalize `SourceRegion` through
`Region`'s illegible-text substitution. Later adapter serialization must preserve the agreed
literal source flag/text representation; no invented `[ILLEGIBLE]` target for empty source text.

### Baseline is one durable step

1. Creation freezes inputs/config/expected identity and creates a run with `baseline=None`.
   If baseline is requested (mandatory for real/test result runs), do **not** mark zero-budget/
   zero-round/no-train runs complete yet. Reject missing/empty validation membership at creation;
   available blank references are permitted. No model call occurs during creation.
2. The first `step_simulation` checks frozen inputs/identity/adapter/evaluator, calls `load_base`,
   predicts only ordered VALIDATION pages at round 0/purpose `baseline_validation`, validates
   coverage/ownership/geometry, and evaluates locally. It never calls `reveal`, `fit` or selection.
3. Recheck inputs, CAS-commit **only** the baseline; apply stop reason within that same updated
   record if budget/round/train exhaustion already applies. Return immediately, even if more
   budget remains. Baseline failure leaves the old run unchanged; concurrency can only commit
   one baseline. Remote side-effect reuse is a candidate-3 responsibility, not promised by this CAS.
4. Next step acquires positive round 1. Seed uses `seed + len(rounds)`; baseline is not a round,
   seed offset, reveal count or score source. Later fits receive cumulative selected TRAIN only.
   Random/score=none skips pool prediction; validation still runs after each real fit.
5. Completed resume revalidates source/config/identity but performs no load/predict/fit. Failed
   baseline/round can retry locally; preexisting fixture zero-budget behavior remains a no-op.

Export baseline in JSON and one CSV row with `record_type=baseline`, round 0 and zero labels;
round rows get `record_type=round`. Preserve existing columns and blank annotation_seconds;
add purpose/status/count fields and defined-rate flags. Export creates a new directory and
reads one committed snapshot. Undefined rates are blank/absent with explicit flags/counts,
not zero. Default fixture exports keep only their actual round rows.

### Local evaluation view: one version, no layout in slice 1

Proposed `PageTextEvaluatorV1.identifier = page-text-nfc-v1`. The version is an engineering
validation view, not official benchmark scoring or final-test approval. Normalize only a copy:
CRLF→LF and Unicode NFC; join declared ordered line texts with LF, preserve case, punctuation,
spacing and literal illegibility text. No modernization, dehyphenation or order repair from GT.
Blank line strings remain in the join; source arrays are never mutated. For missing annotations
raise an error; for present empty strings compute counts normally.

Compute per-page Levenshtein character edits and Unicode-whitespace-token edits separately.
Return finite numeric `pages`, `char_edits`, `reference_chars`, `word_edits`, `reference_words`,
`cer_defined`, `wer_defined`, `invalid_output_pages`, `truncated_pages`, `refusal_pages` and
`failed_pages`. Flags are 0/1. Add `cer` only when reference_chars>0 and `wer` only when
reference_words>0. Counts are exact integers in the returned metric map; adapters storing float
metrics must preserve these small integers exactly. CER/WER may exceed 1. All-empty reference
is valid count evidence with undefined rates; whitespace can define CER while WER is undefined.
Absent validation pages are ineligible, unlike an existing all-empty validation reference.

Corpus CER = sum edits / sum reference chars, WER analogously; never average per-page CER.
Empty reference pages contribute insertions and zero denominator. Use raw edit-count helpers;
existing CER's empty shortcut cannot be inverted. Failed predictions contribute empty hypotheses
plus failure flags, never dropped pages. Evaluation receives only validation truth; acquisition
never sees metrics or targets. Keep the finite-number evaluator interface: no NaN/null rates
in its dictionary. JSON/CSV/report consumers respect the separate defined flags.

### Candidate-1A exact acceptance map

Each test below is required on the exact implementation candidate; no real-readiness claim.
Testing independently reviews focused tests, full existing suite and Ruff once code is ready.

| ID | Required pass evidence |
| --- | --- |
| L1 compatibility | Existing fixture CLI/API/import/legacy fake loop unchanged; fixture import requires no torch/transformers/modal. Old stored fixture records deserialize; default budget-zero fixtures have no new calls. |
| L2 baseline step | Instrument model/oracle/selector: creation calls none; first positive-budget step commits baseline only; second step first acquisition. Budgets 0, rounds 0 and no TRAIN still commit exactly one baseline before completion. No reveal/fit or test-page access at baseline. |
| L3 atomicity | Inject load/predict/evaluator/pre-CAS failures: no baseline/round published; post-CAS lost response reloads one record. Two baseline callers: one commit; loser reloads. This proves local state, not exactly-once remote execution. |
| L4 ownership | Reject wrong run/model/round/purpose, missing/extra/duplicate pages, baseline round>0, nonbaseline round0 and acquired round<=0; reject malformed real/test geometry. Validate ordered output against requested page order (or deterministically reorder by that explicit order after exact-set validation). |
| L5 sampling | Same train membership/seed with/without baseline produces the same first batch; later seed offsets depend only on acquired rounds. Full cumulative examples, unique page budget, partial last batch/exhaustion. Random null scores, no pool prediction; baseline never used by selectors. |
| L6 identity | Expected code/dependency/model/processor/policy mismatch refuses before calls/commit. Telemetry absent at creation is valid; different elapsed/peak-memory observations do not create identity mismatch. Explicit real adapter missing fails; injected test adapter exports contract-test kind. |
| L7 text counts | Hand-count edits for insert/delete/substitute, unequal lengths proving micro vs macro, NFC combining characters, CRLF, blank lines, Unicode whitespace, literal illegibility and order. One empty reference + one nonempty includes insertions; all empty omits both rates; whitespace-only defines CER but not WER. Rates can exceed1. Failed page counted once, not discarded. |
| L8 isolation | Across separate runs perturb hidden/validation labels without changing revealed examples: training inputs/selection remain identical; validation perturbation changes only metrics. No GT-derived crop/order/line-count input to predict; oracle checks remain intact. |
| L9 exports/resume | Separate-process synthetic API harness resumes baseline then rounds; JSON/CSV contain exactly one budget-zero row, correct counts/flags/kind, empty seconds, no overwrite. Identity/source mutation refuses even completed resume; fixture rows preserved. |

The release candidate needs reproducible commands/results and an exact SHA, with warnings
explicitly accepted by Manager. Success means local contract engineering only. No corpus,
Qwen quality, training, CUDA, checkpoints-on-disk, Modal auth or cancellation PASS is implied.

## Later candidates and stage-specific gates

### Candidate 1B — READ2016 converter and engineering-only source policy

**Accepted v0.3 specification; implementation and independent review pending.** On 2026-09-16
the user admitted the official
READ2016 1.2.0 TRAIN/VALIDATION partitions for engineering verification with document grouping
explicitly unknown. Original labels/membership stay unchanged; final test is excluded. This
resolves source-use permission, not converter correctness or scientific independence. Candidate
1A is completed; candidates 2/3 remain unimplemented. No recipe, evaluator or score change here.

#### Narrow schema and frozen policy

Introduce `SourcePolicy` with exactly `known-document-v1` (default) and
`read2016-official-unknown-engineering-v1`. The latter binds READ2016 1.2.0, the archive SHA-256
above, official train/validation membership, `grouping=unknown`, `engineering_only=true`, no test,
and the mapping below. It is not `allow_unknown=True` or a generic split-check bypass.

Keep **`Page.document_id` required and non-null**: legacy `local_data.load_image_pages` and its
document-hash split remain untouched. Override only `SourcePage` and `SimulationPage` with
`document_id: str | None`, still required, nonempty when a string. Add `source_policy` with the
strict default to these two classes, `DatasetSnapshot` and `SimulationConfig`. Under the default,
require a known nonempty document ID and keep cross-split document checks. Under the READ policy,
require **null on every page**, reject TEST and any mixed policy/known-ID row, and do not insert
`-1`, a filename, a page ID or a split name as a document identity. This intentionally supports
one admitted all-unknown corpus, not arbitrary mixed corpora. Defaults keep historical JSON
readable; omission of the document field remains an error, never implicit admission.

Add one plain `SourceArtifact` record (`uri: str`, `sha256: SHA256`) for the converter's
`provenance.json`. `SourcePage.source_provenance` and `DatasetSnapshot.source_provenance` are
optional with default `None`; require them for READ and forbid them under the strict policy in
this slice. Every READ row references the same relative `provenance.json` and hash. The snapshot
retains that reference relative to `source_manifest`. It is label-bearing local provenance and
must **not** be copied onto `SimulationPage`, fit examples or prediction requests. Freeze must
explicitly omit `regions` and `source_provenance` when projecting pages; do not let the existing
`model_dump` copy all new source fields indiscriminately. Raw XML/polygons stay oracle-side.

The provenance file uses a fixed schema, not arbitrary metadata: schema/mapping version1,
source policy, dataset record/version/archive hash, converter code SHA, the limitations above,
sorted regular-file inventory (relative URI, bytes, SHA-256), and ordered page-to-image/XML
mapping with split and observed counts. It contains no transcript copies; original XML already
preserves them. Inventory covers all 804 copied regular source files, including both `doc.xml`
exports. No absolute paths, runtime IDs or timestamps in deterministic output bytes. The
normalized manifest hashes this file; the file does not hash the manifest (avoid a hash cycle).

Extend `_read_source(data, directory, *, source_policy=known-document-v1)` and
`LocalOracle.freeze(manifest, store, *, source_policy=known-document-v1)`. Explicit caller policy,
all row policies, provenance policy and snapshot policy must agree; never infer admission from
nullable fields. READ source/provenance URIs must be safe relative paths beneath the dataset
directory, with no traversal, absolute path or symlink component; keep legacy path behavior
unchanged. Keep the known-document and image-content maps separate: only the admitted
unknown document check is inapplicable, never the image check. Existing cross-split duplicate
image rejection stays strict; READ additionally rejects duplicate image hashes anywhere and
requires exact row coverage against its provenance page map. Reject wrong source/version/archive
identity, missing/changed provenance or raw files and any TEST member. This is integrity against
accidental changes, not protection against a malicious adapter rewriting all local evidence.

`Pipeline.create_simulation` passes `config.source_policy` to freeze. `SimulationRun` validates
config/snapshot/page policy equality; `LocalOracle.__init__` reloads with the frozen policy,
rechecks provenance/inventory and compares source/frozen page metadata (ID, document ID, split,
size, image hash and policy), not just the current set-of-IDs check. Retain existing manifest,
frozen-image and pre-commit source validation. Check policy again on resume, including complete
runs; no runtime option can loosen it. Preserve `exclude_unset=True` historical CAS compatibility.
JSON exports carry policy and provenance in existing config/dataset objects; CSV adds
`source_policy`, `document_grouping` and `engineering_only` columns, with `known`/false for strict
and `unknown`/true for READ. Run kind still independently distinguishes fixture/contract/real.
A synthetic model run on READ is still fixture evidence. Conversion does not enable real execution.

#### Observed source → normalized fields

Planning inspected all 400 actual PAGE XMLs and JPEG headers, line attributes and marginal/missing
metadata examples. Counts below are structural observations, not a semantic transcription audit.
A direct check corrects v0.2's overly strict region-contiguity assumption: TRAIN Seite0111 has
indices 0,2,3,4,5,6,7 and Seite0184 has 0,1,3,4; all regions are referenced exactly once and
custom indices agree. Sorting these explicit indices is unambiguous; never renumber or invent
missing regions. All per-region line indices are contiguous. The 400-page/9,410-line check also
confirmed unique line IDs per page, single plain Unicode elements and positive bounded envelopes.

| Source field / observed condition | Fixed mapping and refusal rule |
| --- | --- |
| `PublicData/Training` 350 pages / `Validation` 50; 8,367 / 1,043 lines | Map only to TRAIN/VALIDATION, preserve membership; no random split or filtering. Stable page ID `read2016-v1.2.0:<train-or-validation>:<XML-stem>` identifies a page, never a document. Emit TRAIN then VALIDATION, each sorted by exact stem. |
| PAGE namespace `2013-07-15`; one Page per XML | Require this namespace/shape and exact XML-stem/image-basename pairing. Resolve JPG only in the same split's `Images`, never via omitted symlinks or arbitrary `imageFilename` paths. |
| `ReadingOrder/OrderedGroup/RegionRefIndexed` and region `custom readingOrder` | Require one flat indexed group, unique nonnegative integer indices and references covering all TextRegions; require matching region custom indices. Gaps are allowed and preserved (observed on two pages). Order by explicit indices, not XML order or box position. |
| All 9,410 lines have per-region `custom readingOrder` | Require exactly one integer index per line, unique complete contiguous indices from zero within its region; concatenate regions then lines. Preserve original line ID, require uniqueness across each page. Reordered XML nodes with unchanged indices preserve ordered line targets; raw XML/provenance hashes correctly change. |
| One line-level `TextEquiv/Unicode` per line; four empty strings | Read that Unicode literally after standard XML character/entity decoding; an existing empty element means `""`. Missing/multiple Unicode/TextEquiv, nested markup or ambiguity is an error. Never use duplicated region-level TextEquiv text, strip whitespace, expand abbreviations, normalize Unicode or append markers. Raw XML bytes preserve XML newline/entity syntax. |
| Line Coords polygons have 4–118 vertices | Preserve original points in copied XML; use envelope `x=min(xs), y=min(ys), width=max(xs)-min(xs), height=max(ys)-min(ys)` in pixel coordinates, no +1/rounding/clamping. Require finite integer pairs, at least three distinct vertices, positive envelope and bounds `0<=x<=W`, `0<=y<=H`. This is a declared lossy box view, not a new source annotation. |
| Two TRAIN lines lack Baseline; two different lines lack custom structure | Keep all four affected lines and their text/Coords; baseline and structure are not required to make a line target. Do not infer missing types from parent/position or fabricate baselines. Reject absent line Coords. |
| Paragraph/page-number/heading/marginalia regions; 499 marginalia regions overall | Include every line, including marginalia, numbers, headings, empty strings and struck-through text. Region polygons, PrintSpace and region text are provenance only; never turn them into extra line targets or prediction crops. |
| `unclear`20, `abbrev`1715, `textStyle`224 and `sic`2 custom span blocks; six standalone TextStyle elements | Preserve full custom/XML, including expansions/style/span offsets. These are not whole-line illegibility flags. No explicit whole-line illegibility was found: emit `SourceRegion.illegible=False` meaning “not explicitly flagged,” not certified legibility. Do not interpret partial unclear spans, missing Baseline or empty text as true. Unknown line-flag semantics require review, not a guessed true/false mapping. |
| All 400 images: JPEG, RGB, single frame, no EXIF orientation; PAGE sizes agree; no rotation attributes observed | Preserve bytes and raw orientation; no `exif_transpose`, rotation, crop, deskew or resize. Require JPEG/RGB/single frame, full decode, PAGE dimensions, absent/identity EXIF orientation and absent/zero declared rotation. Reject any nonidentity transform or unsupported mode instead of silently repairing it. |

Only reading-order indices are interpreted from `custom`; retain known opaque structure/style/
unclear/sic/abbrev blocks unchanged in XML. Reject duplicate/malformed reading-order blocks,
unsupported structural nodes/order schemes or newly encountered annotation block names pending
mapping review. Do not classify semantic legibility from text content. Existing source-derived
literal text feeds the unchanged `page-text-nfc-v1` evaluation copy; converter text remains literal.

#### Callable conversion and output integrity

Proposed public function in new `integrations/public_dataset.py`:

```python
convert_read2016(archive: Path, extracted: Path, output: Path, *,
                 source_policy: SourcePolicy) -> Path  # returns output/pages.jsonl
```

Only the exact READ policy is accepted. Use existing Pillow/Pydantic and standard-library
XML/tar/hash/filesystem code; no new dependency, network fetch or importer framework. Verify the
archive's pinned bytes/SHA-256 and stream its regular-member hashes, then compare the staged
`extracted/PublicData` bytes against that inventory. Reject unsafe/duplicate member names, hard
links, special files, extra/missing regular files, TEST content and symlinked staged paths.
The archive's 400 known relative image symlinks are deliberately omitted only when their
normalized targets are the matching regular JPG in the same split, matching prior staging.
Never extract/follow those links. Ignore normal directory entries. Enforce 804 regular files,
350/50 image/XML pairs and the pinned archive identity at the public entry point; parser unit
fixtures exercise lower-level helpers without adding a production bypass or alternative archive.
Reject XML DTD/entity declarations; use no network or external entity resolution.

Reserve the destination with exclusive `mkdir(exist_ok=False)`; reject an existing directory
(including an empty one), file or symlink. Build in an invocation-owned temporary sibling; copy
all 804 regular files byte-for-byte under `source/PublicData`, parse **those copied bytes**, and
build inventory/provenance plus a pending manifest referencing
`source/PublicData/<split>/Images/...`. All output-relative paths stay inside that fresh dataset.
Hash copied bytes against the verified archive, fully decode each image once, check counts/order/
geometry and re-read the pending manifest/provenance with the explicit policy before publication.
Move completed source/provenance into the reserved directory, then atomically publish
`pages.jsonl` **last**. Its presence is the dataset completion boundary; an empty/pending output
directory is never consumable. A crash before that boundary requires a fresh output path; no
resume/overwrite mode. Report the owned incomplete path rather than deleting unrelated content.
On handled failure clean only this invocation's files; concurrent creation loses at exclusive
mkdir and cannot touch the winner. No directory rename may replace an existing destination.
Revalidate copied bytes before publication and bytes saved by oracle freeze against their frozen
hashes to catch mutation between validation and copying. Original files are never modified or
hard-linked. Output is reproducible for identical source/mapping/code SHA regardless of output
directory or enumeration order. Flush completed output before publishing its manifest; no claim
of recovery from filesystem/hardware failure beyond the existing local persistence guarantees.

One documented direct invocation is sufficient; **no CLI change in this slice**:

```python
from pathlib import Path
from active_ocr.integrations.public_dataset import convert_read2016
from active_ocr.integrations.simulation import LocalOracle
from active_ocr.pipeline import Pipeline

policy = "read2016-official-unknown-engineering-v1"
manifest = convert_read2016(
    Path(".local/assets/read2016-1.2.0/Train-And-Val-ICFHR-2016.tgz"),
    Path(".local/assets/read2016-1.2.0/extracted"),
    Path(".local/prepared/read2016-v1"), source_policy=policy,
)
pipeline = Pipeline.for_simulation(Path(".local/verification/read2016-v1"))
snapshot = LocalOracle.freeze(manifest, pipeline.store, source_policy=policy)
LocalOracle(snapshot)  # reload integrity check; no model or training call
```

An API-created simulation must additionally set `SimulationConfig(source_policy=policy)`;
existing CLI start supplies the strict default and therefore refuses this source. This is
intentional until an explicit source-policy CLI is separately released; do not auto-enable it.
Actual conversion structural smoke follows reviewed code and Manager release, with no GPU or
model dependency. It prepares all 400 pages but does not run a learning experiment or expose
labels to a model. Budget estimate: under 0.6 GB converted output plus under 0.6 GB image/source
snapshot, sequential image decoding, no cloud spend; require 2 GiB available scratch headroom.
These are disk estimates from the 499,592,094-byte source, not measured converter runtime/RAM.

#### Exact ownership and independent acceptance

Development alone owns this bounded implementation scope, released by Manager after exact v0.3 review.

| Files | Allowed changes |
| --- | --- |
| `src/active_ocr/models.py` | SourcePolicy/SourceArtifact, simulation-only nullable document field, config/snapshot/page consistency and provenance reference. Keep Page/legacy validation strict. |
| `src/active_ocr/integrations/simulation.py` | SourcePage fields; explicit policy-aware `_read_source`, freeze/reload and provenance verification/projection. Preserve selected-TRAIN and validation-only oracle boundaries. |
| New `src/active_ocr/integrations/public_dataset.py` | Single pinned READ converter, fixed source mapping, safe deterministic output; directly callable, no generic registry. |
| `src/active_ocr/pipeline.py` | Pass/compare source policy at creation/resume/pre-commit; add export disclosure columns only. No model/evaluator/selection redesign. |
| New `tests/test_public_dataset.py`; `tests/test_simulation.py` | Synthetic source/parser/integrity/compatibility and API propagation coverage below. Existing independent tests remain untouched. |
| `docs/simulation.md` | Direct invocation, output/provenance policy, limitations, compatibility and observed structural smoke reproduction once executed. |

No legacy importer, entrypoints/CLI, storage implementation, Qwen, Modal, selector, evaluator,
config defaults outside these records, dependencies/lock, source labels or split edits. Testing
owns a separate `tests/test_read2016_independent.py` and small review record when assigned.

| Check | Required independent evidence at exact implementation SHA |
| --- | --- |
| C1 strict compatibility | Actual historical fixture JSON resumes/exports with strict defaults; Page and legacy import reject null/missing document IDs. Known documents crossing splits and identical cross-split images still reject. All existing 1A regressions pass. |
| C2 admission | Null documents without explicit READ caller/config/row/snapshot policy reject. Mixed policy/known-ID rows, fake `-1`, TEST, wrong source/version/hash or inconsistent provenance coverage reject. READ duplicate image content rejects even within one split. |
| C3 lossless text/order | Synthetic shuffled XML nodes retain explicit index order. Reject duplicate/missing/unknown references or gapped line indices; accept preserved gapped region indices with complete coverage. Preserve exact Unicode, whitespace, blank lines, abbreviations and struck-through/unclear span text; all provenance XML bytes match input. No extra region-level targets. |
| C4 geometry/orientation | Hand-computed nonrectangular/marginal polygon envelope; preserve missing Baseline/structure; reject nonfinite/degenerate/out-of-bounds/missing Coords, mismatched dimensions, bad JPEG, nonidentity orientation/rotation, unsupported mode and ambiguous text. No hidden rotation or +1. |
| C5 provenance/mutation | Tamper with manifest, policy, provenance, source XML/doc.xml, original/frozen image or copied dimensions: fail freeze/resume/pre-commit before model/budget publication. Test mutations during freeze as well as after; match bytes actually persisted. Different output roots produce identical manifest/provenance bytes for the same clean code SHA. |
| C6 filesystem/atomicity | Archive/member/path traversal, staged symlink, duplicate/missing/extra member, target collision (including empty dir/symlink), I/O interruption and concurrent publication never overwrite source/destination or publish partial output. Test DTD/entity refusal without external access. |
| C7 isolation/export | Spy fit sees selected TRAIN SourceRegions only; predict pages contain no provenance/XML/polygon/text/region count. Validation truth reaches only evaluator; reveal validation rejects. Fresh-process snapshot/run reload and JSON preserve null document IDs; JSON/CSV retain policy and engineering limitation; fixture kind is never relabelled real. |
| C8 actual structural smoke | After reviewed code release, independently compare all 350/50 rows and 8,367/1,043 lines to copied original XML order/literal text/envelopes and image hashes; all four empty lines and two missing baselines retained, all804 originals hash-identical, no TEST. Freeze/reopen full dataset under explicit policy. Record command, clean SHA, archive/provenance/manifest/ground-truth hashes and counts locally; public evidence contains counts/hashes only. No OCR quality claim. |

Run focused author/independent cases, full suite and configured Ruff at the final candidate;
there is no configured type checker. Reject implementation on any compatibility/leakage/integrity
failure. Technical acceptance requires independent Testing and Manager acceptance; the subsequent
actual structural smoke supplies source execution evidence. No further source-use permission
question is needed within this admitted scope. Comparative independence, final test, model and
provider execution remain separate gates. Publish no full normalized manifest or raw source GT.

### Candidate 2 — concrete Qwen runtime, training and checkpoints

**v0.4 proposal for exact-version review.** Implement the model boundary after converter acceptance;
this section does not release implementation or a paid run. Source admission, the two-Volume /
one-dispatcher architecture and USD30 total / USD5 first reviewed smoke remain accepted. Keep
candidate3 responsible for transport/journal/real CLI and enabling `RunKind.REAL` in Pipeline.
The existing `QwenRunner` and loose `parse_output` are unused placeholders, not a second supported
execution API. Replace their implementation surface with the one runtime below; no legacy job adapter.

#### Pins, owned files and API

Use Linux x86-64, Python3.11 and the [accepted runtime pins](../docs/modal-preflight.md):
`torch==2.14.0`, `torchvision==0.29.0`, `transformers==5.16.1`, `peft==0.20.0`,
`accelerate==1.14.0`; retain the reviewed transitive lock including tokenizers0.23.2,
safetensors0.8.0 and Pillow12.3.0. Development adds torchvision and pins the GPU optional group
and Linux resolution in `pyproject.toml`/`uv.lock`. Modal1.5.5 stays candidate3; no local ML install
is requested by this planning publication. Metadata/source compatibility is not a passed build.
Use native PyTorch SDPA, no third-party FlashAttention, TRL/Trainer, quantization or extra framework.

Development owns only `src/active_ocr/integrations/qwen.py`, `pyproject.toml`, `uv.lock`, new
`tests/test_qwen.py` and new `docs/qwen-runtime.md`. Keep small private recipe/checkpoint records
and helpers in qwen.py using the existing immutable Model base; no new general configuration
system or shared-model edit is presently needed. Imports of torch/Transformers/PEFT are lazy;
ordinary local fixture imports/tests must continue without the GPU group. Testing owns its own
independent test/review files when assigned. No converter, Pipeline, CLI, Modal entrypoint, metric,
selector or source changes in candidate2. Report a concrete scope exception if one becomes necessary.

Proposed `QwenModel(input_root: Path, output_root: Path, real_config: RealOCRConfig,
*, runtime_manifest: Path)` has `backend="qwen3-vl-v1"`, `kind=RunKind.REAL`, the existing
reset-fit policy, recipe/identity metadata and optional ExecutionTelemetry. Constructor validates
records/paths without loading weights. Methods match the current protocol:

```python
load_base(*, experiment_id: str) -> str
fit(examples: tuple[RevealedExample, ...], *, seed: int,
    experiment_id: str, round_number: int) -> str
predict(pages: tuple[SimulationPage, ...], *, experiment_id: str,
        round_number: int, model_id: str,
        purpose: PredictionPurpose = PredictionPurpose.POOL) -> tuple[Prediction, ...]
```

Use `input_root/model` and verified `input_root/images`; write only under output_root. The later
Modal adapter maps safe image keys to worker-local paths; Qwen consumes `image_uri`, original
size/hash and ID, never `source_image`, oracle files or document/GT-derived hints. Reject escaping
paths/symlinks, altered image bytes/dimensions, TEST input, malformed ownership, duplicate pages
and missing assets before loading/forward work. Fit accepts only nonempty unique selected TRAIN
examples at a positive round, cumulative as supplied; it does not call the oracle or select data.
Baseline prediction is round0/VALIDATION/base checkpoint; acquired validation and optional pool
requests are positive-round and match checkpoint ownership. Random smoke/pilot never call pool;
all predictions have confidence/entropy=None. Empty prediction page tuples return empty after
identity validation without generation. Keep runtime methods synchronous for one later dispatcher.

**Worker identity is not local Git discovery.** The local coordinator retains its existing
checkout-based `check_local_identity`; a copied installation without `.git` remains fail-closed.
Run that future CLI from verified checkout source/editable setup. Workers intentionally receive
no `.git` or whole repository. Candidate2 verifies the supplied reviewed runtime manifest's
source SHA, explicit code-file hashes and installed Linux package fingerprint against frozen
`ExpectedIdentity.code_bundle_sha256`/`remote_dependency_sha256`, plus recipe/model/processor
identities, before load or generation. Candidate3's allowlisted build produces this manifest and
binds deployment/build references; local code/dependency fingerprints are not compared to remote
Linux packages. No new identity service, self-reported GPU-as-identity or prior successful GPU run
is required. CPU build/import evidence can establish the expected package/code bundle first.

#### Whole-page processor and bounded sequences

Load only revision `ebb281ec70b05090aa6165b016eac8ec08e71b17` from checked local files, with
`local_files_only=True`, `trust_remote_code=False` where supported, and offline Hub settings.
Construct Qwen3VLProcessor explicitly from Qwen2VLImageProcessor.from_pretrained (the pinned
TorchvisionBackend class), AutoTokenizer.from_pretrained(use_fast=True) and
Qwen3VLVideoProcessor.from_pretrained, all from the local snapshot; supply the exact
`chat_template.json` string. The video component is required by this processor constructor but
video inputs are rejected. This avoids passing one ambiguous backend keyword through all three
component loaders. Assert concrete image/processor classes, fast tokenizer and pinned template/
settings; never accept an automatically substituted PIL processor. The mapping
and resize implementation are inspected at Transformers5.16.1 commit
`93c8b7b485963a10800c91f55304db6be211c2bd`:
[auto mapping](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/models/auto/image_processing_auto.py),
[image processor](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py).

Freeze training/decode IDs `qwen3-vl-page-lora-v1` / `qwen3-vl-page-greedy-v1`, recipe
`qwen3-vl-read-engineering-v1`. For original H,W, set per-page area ceiling
`B=ceil(1024**2 * min(H,W)/max(H,W))`; require `B>=65536`. Pass
`images_kwargs={"size":{"shortest_edge":65536,"longest_edge":B}}` to each processor call.
These keys mean **pixel area**, not lengths. Use its pinned smart_resize: initially round each
dimension to a multiple of32; if rounded area exceeds B, scale by `sqrt(H*W/B)` and floor each
dimension to a multiple of32 (minimum32); if below65536, scale up to that area and ceil to32.
Require the actual result to agree with this calculation, area bounds and longest side<=1024;
otherwise fail. No crop/tiling/padding, external resize, EXIF transform or target-guided shape.
This changes the former unspecified long-side proposal into one tested algorithm, not a new cap.

Use CPU torchvision bicubic/antialias=True preprocessing, rescale1/255, mean/std=(0.5,0.5,0.5),
patch16, temporal_patch2, merge2 from the pinned config. Input must already be admitted RGB with
identity orientation. Spatial rounding slightly changes aspect ratio: coordinate mapping is
`x_processed=x_original*W'/W`, `y_processed=y_original*H'/H`, with no offset. Targets and outputs
remain normalized to the original full page, so inverse x*W/1000,y*H/1000 is unchanged.
[Resize backend](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/image_processing_backends.py).

Assert `image_grid_thw=[1,H'/16,W'/16]`; expanded image placeholders count is
`grid.prod()/4 = H'*W'/1024`. Temporal patches duplicate the still frame; they do not double this
LLM token count. The [pinned processor](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/models/qwen3_vl/processing_qwen3_vl.py)
implements that expansion. Pure dimension arithmetic for the400 staged pages predicts1024-high
outputs: width608/640/672/704 for4/30/206/160 pages respectively, hence608–704 image tokens.
This is an ESTIMATE from source dimensions/source code, not executed processor or VRAM evidence.
Preserve actual grid, prompt/target counts and transform dimensions in execution receipts.

Hard limits: expanded prompt P<=2048, supervised target T (JSON plus terminal token)<=4096,
full training sequence P+T<=6144; generation max_new_tokens=2048 and P+2048<=6144. Count **actual
expanded tokenizer IDs**, including framing/image tokens, not characters. No target/image-token
truncation, dropped pages, automatic lower resolution or selected-page substitution. Preflight
all selected targets before first optimizer work; overflow stops the fit without a checkpoint.
Prediction overflow is an operation error before generation. Coverage of these limits is
unmeasured; T>2048 is disclosed as exceeding the generation budget, not silently shortened.
A cap change needs a versioned reviewed recipe; no adequacy/quality claim follows a completed fit.

#### Literal target, prompt and causal mask

One user message contains the image then the following fixed text (no system/tools/vision IDs):

> Transcribe every text line in reading order, including headings, page numbers and marginalia.
> Return only JSON with exactly this shape: {"regions":[{"text":"...","bbox":[x1,y1,x2,y2]}]}.
> Preserve spelling, punctuation, spacing and empty text. Give one box per line. Coordinates are
> numbers from 0 to 1000 relative to the original full page, with x2>x1 and y2>y1. Do not explain.

Freeze the prompt as one constant with LF joining these four lines. It contains no source text,
line count, region order, boxes, filename or split. Selected training targets alone serialize the
ordered SourceRegions: exact `text`, bbox `[1000*x/W,1000*y/H,1000*(x+w)/W,1000*(y+h)/H]`.
Use fixed key order (regions, then text/bbox), compact JSON, `ensure_ascii=False`, `allow_nan=False`,
finite binary64 values without coordinate rounding/clamping. Escape every literal `<` in the
serialized JSON as `\u003c` so source strings resembling tokenizer control tokens remain ordinary
JSON content; parsing recovers the exact text. This is reversible serialization, not GT rewriting.
Do not supervise IDs, illegibility, source span annotations or region classes.

The [staged chat template](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/chat_template.json)
has no `{% generation %}` block; do not request/trust an automatic
assistant mask. Its generation prefix ends in `<|im_start|>assistant\n`; the complete assistant
message appends JSON then `<|im_end|>\n`. Render the pinned template twice with
`tokenize=False`: user-only/add_generation_prompt=True and user+assistant/False. Process each
with the same image settings, `add_special_tokens=False`, `padding=False`, `truncation=False`,
`return_tensors="pt"`. Assert the full input's first P IDs exactly equal the processed prompt,
including expanded image placeholders; mismatches stop, never infer a mask by decoded lengths.
Verify the remaining token span is exactly the serialized target followed by end ID151645 and
the template's trailing newline. Remove only that trailing framing newline from training inputs;
keep the end token. Labels are -100 at positions[0,P), copied input IDs at[P,P+T). No extra BOS,
manual causal shift, prompt/image/pad loss or double EOS. Batch1 has no padding; tests verify any
padding is attention-masked and labelled -100. Assert control tokens cannot occur inside the
encoded JSON, and supervise at least the nonempty JSON/EOS span even for `regions=[]`.
[Template processing](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/processing_utils.py)
documents the generation-block requirement; the [causal loss](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/loss/loss_utils.py)
shifts labels internally and computes FP32 cross-entropy ignoring -100.

Generation explicitly overrides the staged sampling defaults: greedy, one beam/return sequence,
no sampling, repetition_penalty1, no_repeat_ngram_size0, max_new_tokens2048, EOS151645,
pad151643, no forced tokens/stop strings or length penalty tuning. Build a fresh GenerationConfig
rather than inheriting top-k/top-p/temperature settings; do not retain training labels. Set eval,
inference_mode and use_cache=True. Preserve `mm_token_type_ids` and image/grid tensors from the
processor; let the pinned model calculate multimodal positions, never replace them with flat IDs.
Slice returned token IDs after the exact input length; retain raw IDs/text and termination reason.
Decode without cleanup or skipping unexpected special tokens; remove only the verified terminal
EOS from the response. EOS absent at the limit means truncated even if partial JSON parses.

Strict parser rejects duplicate/extra keys, markdown wrappers/trailing prose, NaN/Infinity,
booleans/coerced coordinate strings, malformed UTF-8 strings and nonfinite/out-of-range or
nonpositive boxes. Empty regions and empty text are valid; keep ordered entries, assign IDs
`line-0001`, etc., map floats to original pixels and set Region.illegible=False. Never repair
JSON, clamp geometry, split blocks using GT or fabricate full-page boxes. Malformed output is
`invalid_output`, token-limit termination is `truncated`, both with empty structured regions and
raw evidence. A plain model has no reliable refusal flag: do not guess one from words; preserve
invalid raw text. `refusal` is used only if an explicit future supported termination signal exists.
Exceptions such as OOM/deadline/identity failure fail the operation, not a fabricated blank page.

#### Reset-fit and exact trainability

Load `Qwen3VLForConditionalGeneration.from_pretrained` from the checked local snapshot with
`dtype=torch.bfloat16`, `attn_implementation="sdpa"`, `device_map={"":"cuda:0"}`; no auto
placement/offload. Verify
BF16/device/library support; never run4B on the laptop/CPU or fall back. Within CUDA forward/
generation, select only PyTorch's built-in `SDPBackend.FLASH_ATTENTION` with `sdpa_kernel` (not
an external flash-attn package); unsupported kernels fail. Pin this backend choice in the recipe.
Use BF16 autocast, FP32 trainable adapters/Adam state, no GradScaler, compile or distributed mode.
Disable TF32; record driver/CUDA/device and attention implementation as observations/recipe checks.
CPU tiny-model tests use FP32/math SDPA explicitly as test evidence, not a production fallback.
[PyTorch2.14 SDPA control](https://github.com/pytorch/pytorch/blob/2b3ec34829036a65cd9d1398ea72a0167dc37470/torch/nn/attention/__init__.py).

Expand exactly `model.language_model.layers.{i}.self_attn.{q_proj,v_proj}` for i=0..35 into72
names; compare named_modules and tensor shapes before PEFT wrapping. Staged safetensor headers
independently confirm36q weights(4096,2560) and36v weights(1024,2560). Configure PEFT LoraConfig:
r16, alpha32, dropout0, bias="none", task_type="CAUSAL_LM", init_lora_weights=True,
use_rslora=False, use_dora=False, modules_to_save=None, full explicit target list; default adapter
only, autocast_adapter_dtype=True. Assert exactly144 trainable A/B tensors, all FP32, with
`36*16*((2560+4096)+(2560+1024))=5,898,240` parameters. Everything else, including vision,
mergers/DeepStack, embeddings/tied lm_head, norms and base language weights, is frozen. No suffix
regex matching vision layers. [Pinned model](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py),
[PEFT0.20 wrapping](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/mapping_func.py),
[LoRA initialization](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/tuners/lora/layer.py).

Every fit discards any cached/adapted model and creates a fresh pinned base, adapter and optimizer;
release old tensors before loading another4B copy. Never resume from the previous round's adapter.
Validate seed is unsigned32bit; reset Python/NumPy/Torch/CUDA RNG from that seed, independent of
UUID, hashes, round number and prior calls. Begin each epoch from sorted selected page IDs and
shuffle using a separate `random.Random(seed+epoch)` for epoch0..2; record order. Three epochs,
pagebatch1, accumulation4, no packing/augmentation. Call gradient_checkpointing_enable with
`gradient_checkpointing_kwargs={"use_reentrant":False}`; use_cache=False during training; test gradient flow with frozen inputs, without unfreezing embeddings.

Use torch.optim.AdamW on only the asserted trainables: lr1e-4, betas(0.9,0.999), eps1e-8,
weight_decay0, foreach=False, fused=False. Constant LR, no warmup/scheduler. Partition each epoch
into groups of up to4 pages. Divide each page's mean supervised-token loss by the group's actual
size, backward, then finite-gradient check, global clip norm1, optimizer.step and zero_grad.
Thus pages are equally weighted within a group; this is not a global token-weighted batch loss.
Flush the last partial group each epoch; updates=`3*ceil(N/4)` (N2=>3,N10=>9,N20=>15).
Require finite loss/gradients and nonzero aggregate update; zero A gradients on the first step
are expected from initially zero B. Compare chunked hashes of frozen parameters before/after fit;
record trainable changes, update/supervised-token counts and finite diagnostics, not raw targets.
No partial checkpoint on failed fit and no within-fit optimizer/RNG recovery in this release.

#### Immutable checkpoint and fresh reload

Maintain two canonical input manifests: base files(config, generation config, safetensors index,
two shards) and processor files(config plus chat template, preprocessor/video-preprocessor,
tokenizer config/JSON, vocab, merges). Sorted relative filenames/bytes/SHA-256 bind each manifest
to repository/revision and schema1; verify against staged public provenance, then against actual
files on load. These digests populate model/processor ExpectedIdentity. No on-demand downloads.

`load_base` verifies/loads the actual base and publishes a base-kind reference manifest; it does
not train. `fit` saves only adapter weights/config via safe serialization with
`save_embedding_layers=False`. Normalize the saved config's base reference to the pinned repository/
revision, not the machine path; reload always receives an explicitly loaded verified base, never
AutoPeftModel or a Hub lookup. Publish only adapter_model.safetensors and adapter_config.json from
the fresh PEFT staging output; exclude autogenerated cards/caches. Verify exact keys/shapes/dtypes,
finite tensor values, hashes and strict config, then publish the manifest last in a new immutable
directory. Repeated paths are reusable only if their complete manifests/bytes match; never overwrite.
[PEFT save/load](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/peft_model.py).

Model ID remains `checkpoint:sha256:<canonical-manifest-hash>`. Manifest schema1 binds kind,
base/processor identities, exact recipe/prompt/target format/limits and code/build/package hashes;
adapter manifests additionally bind owner run/positive round, ordered selected image IDs/hashes,
selected-target digest, seed, actual updates/tokens and adapter file inventory. Use canonical
UTF-8 JSON (sorted keys, compact separators, no NaN); omit paths outside the artifact, IDs for
provider attempts, timings, costs and raw targets. The manifest does not contain its own hash.
Base reference may be shared; an adapter is owned by its recorded run/round. Check the whole
manifest and referenced bytes before model load, then actual loaded adapter keys/values before
forward. Reject missing/extra/corrupt files or recipe/base mismatch; never substitute base weights.
Receipts hold telemetry and verification results separately; journal/call reuse is candidate3.

Independent fresh-process reload test uses the same saved adapter, software/device/backend and
the lexicographically first **selected TRAIN** page (no validation selection). Record one greedy
probe before destruction and after fresh base+`PeftModel.from_pretrained(..., is_trainable=False,
autocast_adapter_dtype=True)` load. Require exact adapter tensor equality and identical generated
IDs, status and parsed regions. Also compare FP32-converted raw full-vocabulary next-token logits
for the first min(8,generated_length) positions, evaluated on the same fixed pre-save generated
prefix in both instances, with `rtol=1e-3, atol=1e-2`; require finite values and record maximum
absolute difference. This is a predeclared acceptance threshold, not a proven hardware guarantee.
If it fails, retain evidence and investigate; do not widen tolerance after seeing results or claim
two separately trained runs must be bit-identical. eval/inference mode and dropout0 apply to both.
The train probe and its outputs are engineering exposure, never unbiased evaluation.

#### Independent acceptance and execution boundary

| Check | Required evidence at exact candidate SHA |
| --- | --- |
| Q1 load/identity | Pure boundary tests reject wrong recipe/package/code/input/manifest before model calls. Local package import needs no ML group; worker succeeds without Git only with verified bundle metadata. Pinned Linux build/import must be reproduced separately; metadata compatibility alone does not pass. |
| Q2 actual processor | On pinned tokenizer/torchvision/processor with synthetic images, verify template bytes, IDs151643/151645/151655, image expansion/grid, per-page area arithmetic/32rounding, transform inverse, prompt+full prefix equality and limits. Golden tall/wide/rounding cases; no4B load. |
| Q3 mask/serialization | Actual tokenization of Unicode, whitespace, quotes/backslashes, empty lines, literal control-token-looking text and zero regions. Decode target JSON back to exact inputs; verify prompt/image/pad=-100, JSON+EOS supervised, trailing framing newline excluded and causal first/last positions by hand-counted CE. Overflow never truncates/drops. |
| Q4 tiny CPU model | Build random tiny Qwen3VL configs using pinned classes (two language/vision layers, small hidden dimensions, compatible DeepStack/mRoPE/head sizes, vocabulary retaining pinned token IDs), use FP32/mathSDPA and synthetic64x64image. Actual forward/backward/PEFT injection, non-reentrant checkpointing and causal loss; explicit test-sized targets, no public4B/runtime fallback. Proves library boundary/gradients only. |
| Q5 LoRA/reset | Verify all72 real names/shapes from headers and later loaded model; exact144tensors/5,898,240FP32trainables on4B. Tiny tests show fresh A/zero B, frozen weights unchanged, same seeded initial state after intervening fits, cumulative inputs only and correct3/9/15updates/partial-group scaling vs manual optimizer reference. |
| Q6 decode/parse | Explicit generation config overrides publisher sampling; exact input slicing/EOS termination, cap-truncated valid-looking JSON, unexpected control tokens, duplicate keys, booleans/nonfinite/out-of-bounds geometry, empty valid output and invalid raw evidence. No GT-based repair, scores or heuristic refusal labels. |
| Q7 checkpoint | Tiny-model safe save/fresh-process reload plus corrupt/missing/extra tensor/file/config/base/path cases fail closed; requesting base after an adapted prediction actually restores base; optimizer/labels/caches absent. IDs hash exact manifests; requested checkpoint is actually loaded. CPU equality/tolerance evidence is distinct from4B/CUDA acceptance. |
| Q8 isolation | Spies prove only selected TRAIN targets enter serialization/fit; prompt/image preparation consumes no GT/order/boxes/counts/document hints. Prediction accepts only page metadata; output/raw references contain no hidden-source material. Invalid ownership/identity/image mutation publishes no result/checkpoint. |
| Q9 future4B smoke | After model+transport review, verify actual CUDA/BF16/SDPA support, mask/token/grid coverage, finite loss/gradients, intended LoRA updates/frozen hashes,3updates on2selected pages, base/postfit2validation predictions, fresh reload probe/tolerance, timing/peakVRAM/cost. Failures stop within the accepted envelope; no OCR-quality or adequate-training verdict inferred. |

Acceptance order is deliberately staged, not circular:

1. Candidate2 publication: run pure helpers/boundary tests, header/shape checks, full existing
   local regression and Ruff without installing ML locally. Independent Testing can pass **only
   this offline scope** and inspect the actual-ML test code. Q2–Q4/Q7 actual processor/tiny-model
   execution, the loaded4B part of Q5 and Q9 stay explicitly pending; skips are not PASS.
   Manager may then release candidate3 implementation without waiting for a GPU or Linux build.
2. First reviewed build: candidate3/Platform constructs the pinned Linux image and runs the
   actual tokenizer/processor/tiny-model/fresh-process tests on CPU before any4B load or GPU call.
   Use only allowlisted synthetic self-check code and pinned small processor assets; never
   dataset XML/labels or full weights in the build. Account for this CPU build/check time inside
   the already approved first USD5 envelope and USD30 gross total. If checks fail or exceed the
   reviewed build allowance, stop; no automatic rebuild or CUDA submission. Independent Testing
   reviews the CPU evidence before the GPU portion is released. No extra platform framework.
3. Bounded GPU portion: after the code/transport review and CPU checks, Q9 acquires loaded4B
   module/trainability, CUDA headroom, decoding coverage and reload-tolerance evidence within
   the same reviewed smoke envelope. These are runtime checks, not prerequisite past successes.

No model/build execution is claimed by this plan. Training adequacy, later uncertainty extraction
and final-test methodology remain held; keep random and score=none.

### Candidate 3 — one Modal adapter, journal and CLI

Development owns new `integrations/modal_model.py`, operation records, only justified SQLite
transaction support, real CLI construction/preflight/reconcile, exports and tests. Platform alone
owns new `entrypoints/modal_app.py`, runtime mounts/resource bounds. Development is sole writer
of `pyproject.toml`/lock after Platform agrees pins. Freeze shared request schemas first; do not
concurrently edit shared files. Model/metric logic stays outside the Modal entrypoint.

One synchronous adapter wraps `load_base/fit/predict` with one sequential GPU Function dispatcher
and two Volumes: one input bundle mounted read-only and one output subtree mounted writable.
Strict `RemotePage` is id, document reference if available, original dimensions,
image_sha256 and safe image key—no `source_image`, host paths or `regions`. Fit arguments contain
only cumulative selected TRAIN examples; predict arguments contain no targets. The uploaded
code/image allowlist must exclude oracle source, original GT/archive, SQLite DB, local memory,
test images and labelled caches. Do not upload artifact_root or repository wholesale. Store only
images and pinned model assets in the input Volume; trained checkpoints and derived outputs
belong in the output Volume. No selected text dataset/targets in persistent trainer caches/logs;
trained weights may retain learned text. Mount only the assigned bundle/run subdirectories,
using provider read-only input enforcement; verify actual mounts at smoke. The two-Volume
correction is required because SDK 1.5.5 rejects one Volume ID at multiple mount paths,
confirmed by installed-source inspection and an offline duplicate-ID probe. See the accepted
[Modal preflight](../docs/modal-preflight.md); no additional worker, queue or service is added.

The common request carries schema, operation, run/round/purpose, ordered image ID+hash list,
exact input checkpoint/base and recipe identities; fit also carries selected-only target digest.
Canonical operation identity **includes purpose, ordered pages and checkpoint**, so baseline,
pool and validation cannot share receipts accidentally. Hash semantic inputs only; attempt IDs,
timeouts/cost limits and volatile telemetry are separate. None of these hashes seed learning.
Result must echo semantic identity and provide complete receipt/artifact checksums and observed
execution identity/telemetry. Rehash consumed image/checkpoint bytes; reject path traversal,
escaping symlinks, wrong dimensions/ownership and incomplete artifacts before publishing a round.

Use existing SQLite kind `model-operation`: atomically reserve owner/request before any paid
submission; competing local callers cannot submit the same operation. Then spawn, persist call
ID, synchronously wait. Reattach to known calls on restart; verify completed receipts before reuse.
Known successful fit survives a later prediction/evaluator/round-commit failure. Unknown submission
outcome (lost spawn acknowledgement/ID) stops for explicit provider/receipt reconciliation, never
blind resubmit. Partial artifacts are not success; committed-round response loss reloads that
round, not another fit. Worker checks completed receipt before work and publishes immutable
attempt outputs plus a completion manifest. Local CAS/Volume do not guarantee exactly-once paid
execution. Test every interruption and preserve cost/attempt evidence.

These API semantics are documented by [FunctionCall](https://modal.com/docs/sdk/py/latest/FunctionCall)
and [Volumes](https://modal.com/docs/guide/volumes); SDK1.5.5 signatures were checked by Platform,
not cloud behavior. Application retries0 does not stop infrastructure crash rescheduling
([retries](https://modal.com/docs/guide/retries)); per-attempt timeout excludes scheduling and
restarts on retry ([timeouts](https://modal.com/docs/guide/timeouts)). Bound total attempt/startup
time and absolute deadline; verify cancellation/terminal state before any subsequent attempt.
Account readiness, deployment and storage/cancel behavior gate cloud work, never candidate-1A tests.

Explicit real CLI creation/resume must reconstruct the frozen recipe and adapter, fail on missing
SDK/auth/checkpoint, and never fall back to fixture. Before submission compare actual immutable
code/packages/model/build against expectations; record telemetry separately. No within-fit
optimizer/RNG recovery in the first release: an approved terminally failed fit restarts from base,
while a completed fit is reused. No new server, queue or generic job framework.

## One future workload and evaluation policy

All rows are proposals requiring the applicable code/data/method/spend release. The table
supersedes prior incompatible counts; resource quotes must be recomputed for these rows.
Order selections by stable page IDs and fixed RNG, never GT/model quality. Exact selected IDs and
hashes must be frozen after source admission; no synthetic document grouping to meet counts.

| Stage | Fixed proposed workload | Meaning / gate |
| --- | --- | --- |
| Offline 1A | Synthetic known-group pages and doubles only; no actual READ or cloud | Proves L1–L9 contracts. |
| Minimal real smoke | Seed824; 2 TRAIN pages, batch2/budget2/1 fit; 2 fixed validation pages; 3-epoch recipe (3 optimizer updates); base and postfit validation (4 page predictions), plus one selected-train probe before/after reload (2 predictions) | Proves real fit/geometry/checkpoint/transport if successful; label exposure recorded. Engineering-only admission is granted; reviewed conversion and smoke release remain required. |
| Random engineering pilot | Seed824; official TRAIN pool, batch10/budget20/2 rounds; first 10 fixed validation pages for diagnostics, then all50 for base/round exploratory tables | 2 fits with9/15 updates under the3-epoch recipe; 150 full-validation predictions, plus diagnostic repeats if required. No pool scores. Tests stable pipeline, not adequate training or independent-document quality. |
| Later adequately calibrated comparison | Only after grouping/design resolution, stable training and score acceptance: random/least-confidence/entropy, seeds824/825/826, batch20/budget60/3 rounds, same frozen approved pool and all50 validation pages | 27 fits, 1800 base+round validation page predictions. With350 train pages and terminal pool scoring retained: two scored methods *3 seeds *(330+310+290)=5580 pool predictions. Requote if pool changes. No execution default until these gates pass. |

Use disjoint recorded TRAIN calibration/probe pages when training-duration tuning is needed;
calibration labels/checkpoints never initialize the later comparison. If excluding calibration
pages changes the350-page pool, freeze the reduced pool and redo counts before approval. Smoke/
pilot outputs may expose development labels, so record total engineering exposure separately
from each run's simulated revealed count. Unknown grouping prevents a claim that such page-level
separation implies document independence. No test archive is used at any stage above.

Three epochs is a starting measurement, not proof of convergence. Before comparison, measure
loss/update/validation curves and target coverage on declared development data under a separate
calibration budget. Research/Planning justify an adequate fixed duration or explicitly limit
claims for undertraining. Freeze E epochs and all other settings across strategies at each budget;
terminal checkpoint, no per-strategy early stopping or tuning during the comparison. Revision
of E means a new experiment recipe. No requirement that uncertainty beats random.

Text evaluation is the slice-1 NFC/count view. For future real results, inspect actual line order/
illegibility semantics and approve the versioned engineering evaluator; reported validation is
used for development and is not an unbiased final test. Full official validation is the common
reporting set once admitted; tiny smoke/diagnostic subsets are clearly labelled and not mixed
into full-set curves. Report raw counts, failures, per-page distributions and paired-seed spread.
For the later comparison, use budgets0/20/40/60 and trapezoidal area under **error** /60 (lower
is better); omit undefined rates and do not integrate across missing budgets. Show attempted/
completed/failed runs; no zero-error imputation, invented human times or extrapolated savings.

### Held modules: layout and uncertainty

No layout scores in 1A. Later proposed `line-iou-total-v1`: compute geometry-only IoU; discard
edges below0.5 **before** matching. Choose one-to-one matching maximizing the sum of admitted
IoUs, allowing unmatched nodes. The objective is total IoU, **not** cardinality-first. Among
exactly equal maximum sums, prefer more matches, then lexicographically smallest sorted
(predicted-ID, reference-ID) pair list; do not use epsilon perturbations that can reverse unequal
sums. Freeze numeric representation/tie implementation with golden competing-edge examples
before release. Report TP/FP/FN, precision/recall/F1 and matched mean IoU; omit each undefined
ratio with counts/defined flags. Never use text similarity or GT matching to repair reading order.
A block covering many lines is one prediction, not several true positives. Standard assignment
implementation/dependency choice belongs to that later exact scope; do not add it to slice1.

Score policy is **none** and strategy **random** for real smoke/pilot. Later use Research's
`generated-text-token-v1` only after extraction acceptance: retain original generated IDs and
lossless serialized JSON **byte** spans; select only ordinary tokens wholly inside a text value's
content span. Exclude quotes/syntax/coordinates/prompt/image/pad/control/EOS and crossing tokens;
include escaped bytes within content. No decode-and-retokenize substitute. Use float32 stable
log-softmax of full-vocabulary raw logits at the model's own prefixes, temperature1. With S the
included positions and V full vocabulary size, confidence=`exp(mean(log p_t[y_t]))`,
entropy=`mean(-sum_v p_t[v]log p_t[v]))/log(V)`. Token-weighted page aggregation, no pool min-max
normalization. Empty S, malformed/refused/truncated output or unverifiable alignment is undefined
and stops scored acquisition; no sentinel or random fallback. Rankers stay unchanged. Test exact
escapes/multibyte/boundary/padding cases, uniform/concentrated logits, batch equivalence and
finite bounds before use. Measure memory/latency and omission/length bias; scores are not
calibrated probabilities of page correctness. These are later scientific gates only.

## Resources, verification and acceptance

Platform's pricing proposal uses one L40S, 2 physical CPU cores and32GiB RAM, approximately
USD2.301264/hour at its 2026-09-14 [official pricing](https://modal.com/pricing) observation.
On 2026-09-16 the user approved a **USD30 gross total** for setup verification/initial pilot,
with the first reviewed smoke at most **USD5**, before credits. This supersedes provisional
USD5/USD20 ceilings; it is not a new quote or permission to run unreviewed code. Exact account,
resource/deadline/accounting checks and measured VRAM/time still gate execution/scaling.
Do not assume credits, actual availability or model fit from installed SDK/weight size.
Two-byte4B weights alone are approximately8GB decimal, before activations/KV/optimizer/workspace.

For the revised smoke, propose at most40 minutes total billed startup+GPU execution, 10-minute
individual execution and5-minute startup limits, one container/input, no app retries, and no
automatic GPU/precision/cap changes. A completed checkpoint may serve multiple calls; count
actual attempts/cold starts/build/storage/idle tails. Reserve worst-case remaining approved cost
before each new call. Absolute deadline/cancel path and restart behavior must be demonstrated
in the authorized smoke; timeout is not an invoice guarantee. Stop on OOM, corrupt identity,
invalid geometry, uncontrolled retries or exhausted limits, preserving evidence. Pilot scaling
waits for measured headroom and a new detailed quote, not linear extrapolation alone.

Expected outputs by stage: reviewed local candidate with test commands; normalized source plus
provenance only after admission; pinned model/processor/recipe/weight manifests; real base/fit/
reload/prediction receipts and raw-output hashes; per-budget JSON/CSV containing kind/identity,
selected counts, edit denominators, failures, steps/tokens/time/VRAM/cost and null human seconds;
then curves and paired analysis. Public summaries include reproducible settings and permitted
small examples, never full oracle GT/private operational records. Preserve original artifacts;
exports never overwrite prior results.

| Readiness verdict | Required evidence; who decides |
| --- | --- |
| Slice1A technical acceptance | Exact code SHA, L1–L9, full regression+Ruff and independent Testing review; Manager accepts warnings. No real-model PASS. |
| Converter acceptance | User engineering-only admission is granted; exact v0.3 mapping/policy, C1–C7 code review and C8 structural smoke still need verification. Unknown groups constrain use; no fabricated identities. |
| Candidate2/3 preflight acceptance | Exact code/recipe, supervised-mask/reset/checkpoint tests, purpose-bound journal failure matrix, upload/mount allowlists, identity checks and independent review. Fakes cannot prove hardware/provider behavior. |
| Paid smoke release | Accepted source/recipe/code, account/access, precise resource/cost/deadline envelope and user spend authorization. Runtime-only evidence is acquired during smoke, not required as a past success to authorize first smoke. |
| Real engineering acceptance | Finite loss/gradients and intended update on CUDA, unchanged frozen weights, full-page geometry evidence, fresh checkpoint reload equivalence with declared tolerance, known-call/receipt recovery and separate-process CLI resume, accounting and independent Testing review. No efficacy threshold. |
| Comparative research release | Grouping or explicitly revised design, adequate-training calibration, exact score/evaluation policy, paired budgets/seeds/frozen memberships and a fresh cost approval. Validation limitations/pretraining contamination disclosed. |
| Final evaluation | Separate explicit frozen-test methodology approval after method/checkpoint selection; test data never train, acquire, tune or select. |

Manager assigns IDs/owned files and approves the exact slice; this plan does not self-release.
No unresolved GPU/corpus/score issue blocks candidate1A. Conversely its successful tests do not
remove later source/method/hardware gates. No production code or job was changed by this revision.

## Change record

- v0.1: phase-A design accepted as planning evidence; preserved at
  `af91ad45592d5815d8e371f8eabf9152e7974292` in Git history.
- v0.2: reconciles accepted Research and Development/Testing/QA preparation; records actual
  archive evidence without independence claims; defines exact local slice1A and gated converter;
  separates baseline step and identity from telemetry; unifies recipe/schedules; represents
  undefined CER/WER independently; holds layout/score implementation with precise later rules.
  Manager accepted this roadmap and initially released only candidate 1A after reviewing exact publication
  `d659c48f38639e54e19ac6aceff1b5adf853355a`; all later gates remain as stated above.
  Corrected local candidate `dee5e1a5ecbbc326921e0f4019792c8f7e5deab2` is now independently
  passed and integrated; source conversion, actual model execution and cloud releases remain pending.

- v0.3: records completed 1A and user admission of the official READ split with unknown groups;
  proposes only the concrete 1B schema/policy, literal PAGE mapping, direct conversion invocation,
  provenance/mutation boundaries and C1–C8 independent checks; corrects the earlier region-index
  contiguity assumption using the two observed gapped pages. Records the approved USD30 total /
  USD5 first reviewed smoke without releasing cloud work. Manager accepted exact specification
  `653fe7dff2018de1e35c91d616c24ed848681a16`; converter implementation and independent C1–C8
  evidence remain pending.

- 2026-09-16 runtime amendment: Manager accepted Platform preflight
  `c2dbd4452bad4c89194e0a10b16b88be2b3a6cdf` and independently reproduced the pinned
  SDK's duplicate-Volume mount rejection. Candidate 3 therefore uses two Volumes with one
  sequential dispatcher. This preserves read-only input enforcement and the same total-byte
  storage estimate; source/methodology, converter scope and user budget are unchanged.
  Candidate dependency pins and USD2.894150 gross smoke estimate remain preflight proposals
  until exact code/build/runtime verification. No resource or paid execution is released here.

- v0.4: proposes the exact candidate2 Qwen API, pinned processor/area-grid limits, explicit
  causal target/EOS mask,72language-only LoRA targets/reset-fit, checkpoint/reload threshold
  and Q1–Q9 independent checks. Worker bundle identity is distinct from local Git discovery.
  Accepted converter, two-Volume topology, source admission and budgets are unchanged.
  No model load, dependency installation/build, production edit or paid execution in Planning.
