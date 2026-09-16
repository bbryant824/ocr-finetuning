# Plan — real OCR with simulated annotation on Modal

Owner: Planning. Version: **0.3**, 2026-09-16. Status: **Converter specification accepted; candidate 1A completed**.
Manager reviewed v0.2 at `d659c48f38639e54e19ac6aceff1b5adf853355a`
and accepts its roadmap and bounded local candidate 1A under the user's engineering authorization.
Candidate 1A at `dee5e1a5ecbbc326921e0f4019792c8f7e5deab2` passed independent Testing;
Manager accepted and integrated it with review evidence `2db380a497ca641d8e1df0cda773074e0d306040`.
See the [exact review and limitations](../docs/verification/local-contract-review.md). Engineering-only
READ source use is now user-approved; Manager accepted the exact converter specification at
`653fe7dff2018de1e35c91d616c24ed848681a16` and releases candidate 1B for implementation/review. Later implementation, methodology and
execution retain their stated gates.
Phase-A v0.1 remains in Git history. Real model execution is still unavailable.

Inspected control for v0.3: `83c026dc6acb4575553fe234a90350435f835ffd`, including accepted
local contracts at `dee5e1a5ecbbc326921e0f4019792c8f7e5deab2`. The v0.2 contract design
below is retained as implementation history; the new scope is candidate 1B only.
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

### Candidate 2 — Qwen load, train, output and checkpoints

Development exclusively owns `integrations/qwen.py`, narrowly needed model records/tests and
agreed GPU lock/dependency edits. Preconditions: accepted full-page output/target and source
mapping, inspected Qwen module names/tokenization, pinned compatible libraries and exact recipe.
No Modal SDK/job/entrypoint or uncertainty implementation here. CPU/synthetic tests precede a
separately authorized CUDA smoke; live evidence comes after candidate 3 supplies the transport.

One future calibration proposal, following Research: LoRA **rank16, alpha32, dropout0**, frozen
base/vision, explicit language-attention q/v projections, page batch1, accumulation4, **3 epochs**,
AdamW lr1e-4, constant schedule, warmup0, weight decay0, clip norm1, no augmentation, bf16 only
if the selected device supports it. Resolve exact full module names and verify frozen tensors;
do not regex-match vision modules or invent a list before architecture inspection. Reset base,
adapter, optimizer and scheduler each cumulative fit. Seed is the paired seed, never UUID/hash.
Flush a partial accumulation group at each epoch end; normalize by its actual microbatch count
so it is not underweighted. Expected updates are `epochs * ceil(revealed_pages / 4)` for this
one-page-example/no-packing recipe. Record actual updates/tokens and test the chosen trainer's
behavior; no silent truncation or fixed-step substitute to satisfy a deadline.

This recipe is calibration, not demonstrated adequate training. Use target/gradient mask tests:
supervise ordered assistant text/geometry and intended termination, ignore prompt/image/pad;
check causal alignment, nonzero supervised tokens, finite loss and finite gradients, intended
trainable tensor changes, frozen tensors unchanged. LoRA components initially having zero
individual gradients are not automatically a failure. Only selected source GT makes targets;
validation and unselected boxes never make crops, prompts or tiling decisions.

Raw schema proposal: strict `{"regions":[{"text":string,"bbox":[x1,y1,x2,y2]}]}`,
ordered lines, coordinates real numbers within [0,1000], x2>x1/y2>y1. No model-generated IDs
required; assign deterministic prediction-local IDs by list position, not source line IDs.
Source illegibility flags remain unchanged in oracle/provenance; this initial model target
supervises literal source text and geometry, not an invented illegibility marker/class. Convert
predictions with `Region.illegible=False` under this schema so an empty string is preserved;
predicting illegibility is a separately versioned task extension, not an implicit text rewrite.
For original width W/height H, map x=x1*W/1000, y=y1*H/1000,
w=(x2-x1)*W/1000, h=(y2-y1)*H/1000. Keep floats; no rounding/clamping/GT-based repair. Training
serialization uses the inverse mapping of the declared polygon-envelope view. The prompt defines
normalized coordinates relative to the **original uncropped page**. Aspect-preserving whole-page
resize needs no content-coordinate offset; exclude cropping/tiling/padding in the first recipe
unless an explicitly versioned inverse transform is tested. Audit processor internals rather
than assume their coordinate interpretation. Stop for invalid geometry, block-as-line output or
unreliable line detection; no invented whole-page boxes or hidden-GT line recognizer fallback.

Decode with one beam, no sampling, no repetition/no-repeat-gram penalties. Initial resource-cap
proposal remains long side1024, max_new_tokens2048, target text cap4096, with total multimodal
sequence/image-token bounds resolved from the pinned processor before dispatch. These limits
may be too small for READ pages; calibrate coverage then freeze, never silently discard/crop
or truncate targets. Preserve raw outputs, finish reasons and invalid/refusal/truncation states.

Model ID `checkpoint:sha256:<manifest_hash>` binds canonical manifest schema, pinned base/
processor/recipe/code, selected-only target digest/ordered IDs, seed, training updates and all
weight/config file hashes. Absolute machine paths and volatile telemetry are not manifest identity.
Use immutable relative paths; incomplete/corrupt checkpoints never load. Fresh process/container
must load the saved checkpoint, verify tensors and reproduce train-probe decoding/logits under
a tolerance declared before running. This compares one saved model, not two independent fits.

### Candidate 3 — one Modal adapter, journal and CLI

Development owns new `integrations/modal_model.py`, operation records, only justified SQLite
transaction support, real CLI construction/preflight/reconcile, exports and tests. Platform alone
owns new `entrypoints/modal_app.py`, runtime mounts/resource bounds. Development is sole writer
of `pyproject.toml`/lock after Platform agrees pins. Freeze shared request schemas first; do not
concurrently edit shared files. Model/metric logic stays outside the Modal entrypoint.

One synchronous adapter wraps `load_base/fit/predict` with one sequential GPU Function dispatcher
and one Volume. Strict `RemotePage` is id, document reference if available, original dimensions,
image_sha256 and safe image key—no `source_image`, host paths or `regions`. Fit arguments contain
only cumulative selected TRAIN examples; predict arguments contain no targets. The uploaded
code/image allowlist must exclude oracle source, original GT/archive, SQLite DB, local memory,
test images and labelled caches. Do not upload artifact_root or repository wholesale. Store only
images, pinned model assets, trained checkpoints and derived outputs on the Volume. No selected
text dataset/targets in persistent trainer caches/logs; trained weights may retain learned text.
Use read-only image/base subpaths and disjoint writable outputs; verify actual mounts at smoke.

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
