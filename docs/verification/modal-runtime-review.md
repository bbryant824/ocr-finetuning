# Independent Modal integration review

## Final verdict — PASS for complete offline integration

Reviewed implementation: `ca3c60c3eb03b51ef442b80e53dade60673e30f2`.
Control: `bfe2789b31cbd6206aa0cb87fd30680869c3b52c`.
Both original regressions pass unchanged. The selector now collects exactly the intended
11 ML cases; collecting them does not execute or validate their ML behavior.
The historical findings and earlier coverage below remain part of this review.

| Final verification | Result |
| --- | --- |
| Full clean exact candidate, staged header-only input, `python -m pytest --tb=short -rs` | **710 passed, 11 explicitly unverified ML skips in 49.26s** |
| Complete independent suite, external test file against a separate clean exact candidate with explicit source imports | **130 passed in 4.72s**; checkout remained clean |
| Ruff, test formatting, diff whitespace | Passed |
| Lock/pyproject comparison with preceding review | Byte-identical; no dependency changes or redundant resolution |
| Actual Modal SDK1.5.5 `prepare_image` construction, actual offline frozen lock export and pinned small processor files | **18 exact individual file copies**, hash-required requirements, CPU-only build-step declaration, output-only build-evidence mount, source inclusion disabled; no build or provider call |

The remaining acceptance checks are now complete:

- Bootstrap checks reject changed code, lock, processor/test bytes, mismatched export,
  dirty tracked state and wrong source revision before image construction. The independent
  negative tests use explicit Git/export fakes; the separate actual SDK construction uses
  a clean exact checkout, real offline export and all eight pinned small processor files.
- Receipt retrieval checks succeed only after the fake eager build produces its identity.
  Missing/noncanonical receipts, corrupt report bytes, wrong source and reported skips fail
  without an automatic rebuild. These are provider fakes, not actual CPU build evidence.
- Input upload planning includes exactly the declared model/image files and excludes extra
  label files. Changed image/model bytes or a model-file symlink reject before any fake batch.
- Real CLI admission uses an **unmocked** clean Git/source and installed-dependency identity.
  Create, local preflight, zero-label baseline, fit, resume, status and export pass with only
  the provider transport replaced. Wrong dependency identity and a tracked source edit reject
  before submission; the temporary edit is restored and the checkout finishes clean.

Together with the earlier byte-complete fit/round recovery, worker, isolation, concurrency,
redaction and deadline checks, no blocking offline integration defect remains. Only the assigned
independent test and review files changed. No author helpers or assertions were substituted.
The actual SDK construction used a network-denying audit hook; an initial scratch inspection
of a private Volume-wrapper attribute was corrected to inspect the forwarded mount options.
This was an inspection-script issue, not a product failure.

**Scope limit:** this PASS does not verify actual Linux CPU test execution, PEFT/model compatibility,
4B training or fresh-process GPU probes, provider mounts/cancellation/terminal status, or billing.
The eleven actual ML checks remain UNVERIFIED, and no model installation/loading, image build,
provider upload/deployment/resource mutation, cloud job or spend occurred. Synthetic OCR outputs
and revealed-page counts establish neither OCR quality nor reduced human annotation time.
Manager acceptance and later execution releases remain separate decisions.

---

## Historical phase B continuation — FAIL: CPU build selects twelve tests but requires eleven

Reviewed implementation: `79fe2c3859fefca501bf4c3efc13d3d23710905d`.
Control: `077bc890bfef618cc565226f9b8e441619c17ef5`.
The original deadline regression is now **PASS**, unchanged. The additional journal-delay
case confirms expiry after the durable SUBMITTING write restores RESERVED with no provider
call/cancellation and preserves the original deadline on another attempt. Earlier failures
and the phase-A verdict remain below as historical evidence.

**Blocking finding (P1, Platform ownership):** `cpu_build_gate` selects tests with
`-k 'actual and not staged_headers'` (`modal_app.py:316`). Pytest matches substrings, so
this selects the eleven intended actual-ML cases **plus**
`test_worker_manifest_no_git_checks_actual_code_and_packages`: **12 cases**.
`_cpu_report` (`modal_app.py:236`) rejects any count other than 11. Consequently, even a
fully successful CPU test run cannot produce the required successful build receipt.
The candidate's fake JUnit tests do not detect this selector/collection mismatch.

Reproduce without running a model or building an image:

```sh
python -m pytest tests/test_qwen.py --collect-only -vv -k 'actual and not staged_headers'
python -m pytest tests/test_modal_independent.py::test_cpu_build_selector_collects_exact_eleven_ml_cases -q
```

The independent regression reads the selector from the actual worker function and applies it
in a fresh collection-only subprocess with plugin discovery disabled. It deliberately fails
on `12 != 11`; no test is xfailed or skipped. Platform should narrow the selector to the
intended ML cases and verify the real collection, keeping the reviewed eleven-case/zero-skip
contract. Testing made no implementation, author-test, lock or dependency changes.

### Continuation evidence

| Check | Observed result |
| --- | --- |
| Full **clean exact candidate**, staged header-only path, `python -m pytest --tb=short -rs` | **681 passed, 11 explicitly unverified ML skips in 47.07s** |
| All independent cases against a separate clean exact candidate checkout, external review-test file and explicit checkout source | **110 passed, 1 failed in 3.63s**; sole failure is CPU selection; execution checkout remained clean |
| Ruff, review-test format, diff whitespace | Passed |
| Lock and pyproject byte equality against previous review | Unchanged; no redundant lock resolution |
| Installed Modal 1.5.5, actual `create_app` constructor with synthetic IDs/canonical settings and a network-denying Python audit hook | One registered dispatch Function; L40S, CPU `(2,2)`, memory `(32768,32768)`, exact RO/RW Volume subpaths, min0/max1/buffer0/retries0, timeout600/startup300, source inclusion disabled |
| Runtime isolation / child watchdog | Exact seven-file copy imports without legacy integrations or Modal/ML; a real local benign Python child is terminated and reaped on timeout |

The SDK experiment constructed objects only: no hydration, image build, deployment or provider
call. The deferred settings-file image remains unhydrated, so no observed final image identity is
claimed. Inspection used SDK1.5.5's deprecated `registered_functions`/`spec` inspection properties;
these are review tooling, not production dependencies. An initial scratch assertion attempted to
read the unhydrated image ID and was removed as invalid inspection, not a product failure.
A local scratch settings path was resolved to remove the OS `/tmp` symlink before the successful
construction; rejecting the unresolved symlink was expected containment behavior.

Additional independent coverage now includes:

- Complete synthetic fit evidence: independently written 144-tensor FP32 safetensor bytes,
  before/after full-vocabulary binary logits and metadata, checkpoint/target/worker closure.
  Distinct process IDs in synthetic metadata are declarations, not an actual 4B probe.
- Two complete cumulative rounds through real Pipeline/ModalModel/SQLite, retaining the same
  base checkpoint for reset-fit. Label counts 1 then 2, previous selected examples retained,
  random/no-pool behavior, final export, idempotent completed-run resume. A committed round's
  lost response is recovered through a reconstructed Pipeline and model.
- Verified fit reuse after downstream result loss: the pending prediction is reconciled;
  round completion performs no second fit or provider submission.
- Hash-consistent invalid evidence rejects: wrong last vocabulary logit, NaN, double `qwen/`
  nested prefix, incorrect actual tensor hash, missing nested logits, extra checkpoint file,
  changed worker identity and foreign attempt. No failed result becomes COMPLETED or auto-retries.
- Deadline cancellation retains the provider call and locked original deadline without
  interpreting acknowledgement as terminal proof or permitting retry.
- Actual worker dispatcher with synthetic input/model bytes and declared environment:
  base completion, exactly two ordered fit stages, metadata-only reload payload, completion
  reuse and shifted-clock rejection. Extra/changed/symlink inputs, changed code/build receipt,
  partial operation and conflicting persisted run clock all reject before a child starts.
- CPU report rejects skip/failure/duplicate/count/exit-code errors; the real collection regression
  exposes the missing link between those report checks and the current worker selector.

The synthetic fit peer's small adapter-config JSON verifies byte closure, **not** PEFT config
compatibility. The full runtime remains unexecuted. Worker tests replace the asset inventory and
package observation with explicitly synthetic ones; they execute the file/identity checks, but do
not prove the staged real input bundle or installed Linux packages. No author helpers are imported.
One test initially used the wrong export filename (`run.json`); it was corrected to the documented
`results.json`, leaving all product assertions intact.

**Remaining before final offline acceptance:** fix and independently retest the real CPU selection;
finish build-image bootstrap construction/allowlist negative checks and CLI admission with an
unmocked clean local identity. Later actual CPU/ML/GPU/provider behavior, cancellation terminal
state and cost settlement remain separate execution gates regardless of offline results. No model
install/load, build, deploy, upload, resource mutation, cloud job or spend occurred. Return this
concrete blocker to Manager/Platform before any CPU-build release.

---

## Historical phase B — FAIL: submission can cross the frozen deadline

Reviewed implementation: `169eb39d359759a9f4f903e720622c968865f0c8`.
Control: `077bc890bfef618cc565226f9b8e441619c17ef5`.
The phase-A result below remains historical and unchanged in scope.

**Blocking finding (P2):** `ModalModel.execute` checks the absolute deadline before
`transport.preflight()`, but does not check again before `spawn`. Preflight performs
provider lookups/hydration and can consume the remaining time. The independent regression
starts with the persisted deadline 3400, enters preflight at 3399, and returns at 3401.
The candidate still submits once, then requests cancellation in `_wait`. The request keeps
its original deadline; the bug is admission after expiry, not deadline extension.

The worker's own deadline check should prevent subsequent model work, but it cannot undo
provider submission/startup, and cancellation acknowledgement is not terminal evidence.
This violates the documented no-submission-after-deadline boundary. Recheck the current
clock after preflight, immediately before submission, and preserve a safe non-submitted
journal state on expiry. Development owns the correction; this review changes no production
code or author tests.

Reproduce using only SQLite, synthetic metadata and a fake transport:

```sh
python -m pytest tests/test_modal_independent.py::test_independent_preflight_expiry_must_not_spawn -q
```

The regression is deliberately **failing**, with no xfail, skip or weakened assertion.
Its assertion is that the fake transport received zero submissions; observed count is one.
This FAIL checkpoint does not claim complete independent integration acceptance.

### Executed evidence

Python 3.11.14, pytest 9.1.1, Ruff 0.16.6; reviewed checkout source used throughout.

| Check | Observed result |
| --- | --- |
| Full clean exact candidate, `QWEN_HEADER_DIR=<staged-model> python -m pytest --tb=short -rs` | **664 passed, 11 skipped in 48.21s**; all skips explicitly unverified processor/tiny-ML checks |
| Independent suite after adding 13 phase-B cases, `python -m pytest tests/test_modal_independent.py --tb=short` | **81 passed, 1 failed in 4.07s**; sole failure is the deadline regression |
| `ruff check .`, review-test format check, `git diff --check` | Passed |
| `uv lock --check --offline --cache-dir <temporary-cache>` | Passed, 96 packages; initial sandbox invocation hit a macOS uv system-configuration panic, outside-sandbox offline retry succeeded |
| Lock comparison against control using stdlib TOML parsing | All 77 existing name/version pairs retained; 19 added, including exact Modal 1.5.5 |
| Installed Modal 1.5.5 source/signature inspection | Local read-only inspection confirms construction parameters, mount options, eager image build and built-in polling `TimeoutError`; no provider call |

The clean regression preceded review edits. Added tests use independent stdlib byte hashing
and JSON serialization, actual SQLite connections and filesystem evidence, and no author
helper imports. The base/prediction fake supplies byte-complete synthetic artifacts; its
Qwen binding uses the production pure binding method. Declared build/package/provider records
are synthetic, not measured execution identities. The Pipeline fixture replaces only the local
machine identity observation, so these tests do not establish clean-source identity admission.

### Independently exercised phase-B boundaries

- Base completion verifies bytes and survives reconstruction with a second SQLite connection,
  even after deadline expiry, without another submission. Remote byte corruption then rejects.
- Lost spawn acknowledgement and lost result read preserve UNKNOWN. Ordinary resume does not
  submit again; explicit observed-call reconciliation verifies completion and reuses it.
- Two actual SQLite connections compete for the same reserved operation while one is paused
  inside preflight. Only one submits; the losing compare-and-swap cannot overwrite completion.
- Actual `Pipeline` plus exact `ModalModel` commits the zero-label baseline. An explicit subset
  uses only that validation page; omission uses both validation pages. Prediction wire payloads
  contain no fixture transcriptions, and validation export membership/count matches.
- Unknown, TRAIN, TEST, duplicate and empty validation subsets reject during real creation.
  A changed run configuration or mutated frozen image blocks resume before further submission.
- The next step sends exactly one selected TRAIN page and its literal transcription. The fake
  deliberately cannot complete fit: no acquisition round commits, and the durable operation
  journal contains no transcription. This checks the selection/submission boundary, not a
  successful fit or OCR quality.

### Inspected implementation and outstanding acceptance

Read the complete new transport/coordinator and worker entrypoint, real CLI, relevant Pipeline,
validation and Qwen changes, deployment documentation and relevant author tests. The implementation
uses one dispatcher, exact seven-file runtime inventory, distinct input/output Volume subpaths,
read-only inputs, one L40S with CPU/memory request-and-limit pairs, zero retries and zero warm/buffer
containers. Source inclusion is disabled; image preparation copies individual allowlisted files.
CPU build selection requires 11 distinct successful cases and zero skips. Fit uses sequential
operation/reload children, with metadata-only reload input and shared absolute deadline.
Worker result publication checks byte closure before immutable completion and Volume commit.
These are code observations, not provider mount, cancellation or GPU evidence.

**Outstanding before a phase-B PASS:** correct and independently retest the deadline finding;
finish independent successful fit/full-round byte closure and recovery after downstream failure
or lost round response; complete adversarial worker/bootstrap and actual SDK construction checks,
and remaining phase-A consumer-limit coverage. Existing author tests for these areas passed in
the full candidate run, but have not been substituted for those independent acceptance checks.
No retry, terminal cancellation, cost settlement, CPU build, full model load, deployment, upload,
resource mutation, GPU job or spend was executed or approved by this review. Actual ML checks and
later provider execution remain separate gates. This evidence supports no OCR/active-learning claim.

---

## Phase A — Independent Modal API review

Verdict: **PASS for phase A: shared records and documented contracts only**, at
`bf0c1d46dd662e38a136f8119cf37ddb9907e994`.
Control: `077bc890bfef618cc565226f9b8e441619c17ef5`.

No blocking API incompatibility was identified. This commit contains schemas, semantic hashes,
structural checks and an implementation contract. It contains **no Modal adapter, journal,
dispatcher, deployment, real CLI or deadline watchdog**. Passing this review does not approve
those missing components or provider execution. Combined integration phase B remains held for
an exact complete candidate and explicit release.

## Executed checks

Python 3.11.14, pytest 9.1.1, Pydantic 2.13.5, Pillow 12.3.0, Ruff 0.16.6. All imports used the
reviewed checkout's source. No optional SDK/ML installation, actual model loading, dataset
experiment, provider call, build, deployment, upload or spend occurred.

| Check | Result |
| --- | --- |
| Full clean candidate, `pytest --tb=short -rs`, with staged Qwen header-only input | **508 passed, 11 unverified ML skips in 40.95s** |
| Independent API suite against a separate clean exact source checkout | **69 passed in 0.28s** |
| Ruff and diff whitespace checks | Passed |

The initial independent 58 cases passed; 11 further result-closure, artifact-path and prediction
purpose cases also passed. Test-file import/format issues were corrected before final lint.
No failing product assertion was relaxed, suppressed or marked xfail. Full regression ran before
review edits; the separate execution checkout remained clean. Dependencies/lock are unchanged
from the previously verified Qwen candidate. No redundant lock resolution or full regression
was run after documentation edits.

Independent tests use their own synthetic records, stdlib canonical JSON and SHA-256 expected
values. They do not import author helpers or mock a nonexistent transport. Synthetic file hashes,
package records, CPU pass counts, image IDs and provider call IDs are **test declarations**,
not measured build/provider evidence. Model asset entries use the existing pinned public inventory;
no model files are opened by the independent suite.

## Verified API boundaries

| Area | Observed evidence |
| --- | --- |
| Semantic identity and redaction | Independently calculated target and operation digests match. Ordered pages, literal target text/geometry, seed, checkpoint, image and policy changes alter identity. Attempts and shifted bounded deadlines do not. Fit RPC JSON retains selected targets; semantic records and repr omit them. This does not authorize logging wire payloads or validation errors. |
| Source and split policy | Required nullable document ID and explicit source policy are enforced. Omitted fields, invented READ document IDs, strict-policy nulls, duplicates, held-out fit labels and out-of-image targets reject. Baseline/positive-round purposes and prediction splits reject incompatible combinations. |
| No oracle fields in prediction | Extra source paths, image URI, provenance, regions and text reject in `RemotePage`; base/predict operations reject examples. Only fit carries selected regions. Actual selection and frozen-snapshot comparison remain coordinator work. |
| Input/build allowlists | Input bundle rejects source JSONL/XML, arbitrary model files, test files and non-addressed images. Exact seven runtime Python paths, pinned small-processor manifest, requirements/test filenames and CPU zero-skip declaration are structurally required. Actual bytes and lock-to-requirements correspondence remain build verification. |
| Bootstrap consistency | Independent code/package digest calculations match receipt properties. Changed image reference, shared Volume ID, code/source/package/build/lock/processor declarations or extra build files reject. Settings bind the run, input bundle and declared deployment; requests using unlisted images reject. |
| Result identity and direct references | Wrong attempt/run/round/model/page order rejects. Prediction raw references must be inventoried and scores absent. Completed fit requires reload evidence; both metadata references and worker/checkpoint references must be present. Duplicate artifacts and mismatched completion key/length/hash reject. Completion digest independently matches exact canonical result bytes. |
| Diagnostic representation | Full-vocabulary FP32 byte count and first-min(8,length) row count, all 144 adapter tensor hashes, distinct declared process IDs and fixed tolerances are required. These fields alone do not prove a second process, tensor equality, finite binary logits or a passing comparison. |
| Deadlines and failure records | Nonpositive or over-2,400-second envelopes reject; a valid shifted envelope is a valid historical record. Failure records reject raw request/targets/traceback, asserted terminal state and cost additions. No clock, cancellation or terminal-status action was executed. |

Fresh interpreter import loads no Modal, Torch, Transformers or PEFT module. The complete
three-file diff was inspected. The records remain in the assigned integration boundary and
reuse existing immutable model types; no new service, queue or general framework was introduced.

## Required consumer checks in phase B

The API intentionally validates only information available in a record. The independent tests
also demonstrate its limits; these are **not completed integration acceptance**:

- `RuntimeSettings.check_invocation` does not compare the current clock or a durable first-call
  deadline. It accepts a shifted bounded envelope with the same operation ID. The later journal,
  immutable run-control record, worker guards and parent watchdog must reject deadline extension
  and expiry across operations/resume, not allocate 40 minutes anew per call.
- An `OperationResult` inventories direct references without opening any bytes. A synthetic fit
  can satisfy these structural fields without a real base/adapter manifest, nested logits or
  process evidence. Consumers must verify canonical completion bytes, whole checkpoint inventory,
  recipe/base/run/round ownership, raw evidence, selected probe identity, nested artifact closure,
  binary logit finiteness/equality/tolerance and actual worker identity before committing.
- `ProbeMetadata.logits` uses generic `ArtifactRef`; the schema accepts both a correct nested key
  and an incorrect extra `qwen/` prefix. The approved root contract below must be enforced by
  producer/consumer code and independently tested in phase B. Generic relative-path validation
  alone is insufficient.
- CPU pass counts and environment/image/Volume/provider identifiers in these records are
  declarations. Actual locked Linux build, zero intended skips, package/code bytes, observed
  provider IDs/mounts and independent CPU acceptance remain later evidence gates.

The bootstrap order in the contract avoids a circular identity dependency: freeze reviewed
build inputs; obtain CPU receipt/package evidence and the actual image ID; freeze expected
identity/deployment naming; create the existing local run UUID; then construct per-run settings
and deploy/verify the actual function. No provider ID is inferred from a declaration. SDK1.5.5
construction and this sequence's operational feasibility were not exercised by this phase.

## Approved cross-role probe contract

The scope approved for subsequent implementation preserves the existing default Qwen behavior:
opt-in `fit(..., external_reload=False)`, a selected-TRAIN-metadata-only `verify_reload` diagnostic
with owned checkpoint and hashed before-metadata/logit paths, and an optional absolute deadline
on construction. The parent owns two sequential fresh interpreters and the watchdog. One fit,
one before probe and one after probe remain the intended counts. These signatures and behavior
are not implemented in this API-only commit.

| Reference | Root | Required key form |
| --- | --- | --- |
| Nested `ProbeMetadata.logits.key` | Qwen output root `/outputs/qwen` | `probes/<checkpoint-sha>/before.f32le` or `after.f32le` |
| `ReloadEvidence.before/after` and outer result artifacts | Run output root `/outputs` | `qwen/probes/<checkpoint-sha>/before.json`, `after.json`, and corresponding binary keys |

Resolve the fixed prefix once; reject mismatched checkpoint namespaces and double prefixes.
A pending checkpoint artifact is not a completed transport fit until fresh-process evidence and
completion bytes verify. Neither a metadata assertion nor this phase A PASS substitutes for
those checks. Original data, labels/splits, Qwen default behavior and budget limits are unchanged.

## Reproduction

Keep the review tests outside a clean checkout of the API candidate:

```sh
review_tests="$PWD/tests/test_modal_independent.py"
review_dir=$(mktemp -d)
git clone --quiet --no-local --no-checkout . "$review_dir/source"
git -C "$review_dir/source" checkout --quiet --detach bf0c1d46dd662e38a136f8119cf37ddb9907e994
PYTHONPATH="$review_dir/source/src" python -m pytest \
  -c "$review_dir/source/pyproject.toml" -o pythonpath= --import-mode=importlib \
  "$review_tests" --tb=short
```

Expected: **69 passed**, without dataset, model or provider access. The 11 actual Qwen ML checks
remain unverified in the full suite. Manager retains acceptance and phase B/build/GPU releases.
