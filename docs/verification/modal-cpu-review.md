# Independent CPU runtime review

## Adapter-export CPU attempt 5 — PASS (actual CPU evidence)

Source `1a03f77519439c0601a83c8c3b4670ca896de337`; control
`a5244059864e711ba615a663752cd3e1cbfbc5ac`. Final author evidence
`b73f8e8060f9821261e9865eaa21e2d1374cd327` and all 32 indexed files verified.
**11 passed, 0 skipped, 0 failed in a 174.973-second build on 2026-09-17.**

Exact Git bytes for the seven runtime files and changed test file match BuildSpec;
canonical receipt/report, source, code-bundle and environment identities agree. Frozen
lock/export, processor manifest, recipe and resource settings match prior acceptance.
Python 3.11.12, 96 sorted unique effective packages and all required pins remain unchanged.
The same eleven case names passed. Their unchanged report hash does not imply unchanged
test bodies: the new test-file hash binds the executed 36-layer injection/export case,
including compact-config rejection, restored full targets and existing tensor/base checks.
That CPU case is now verified at this source; corrected production GPU fit remains unverified.

| Evidence | SHA-256 |
| --- | --- |
| Final index | `0c99a54c29a0a8aac6acc23075d8e36b9b9b2fe22a32ff8bbb59dbf63266ebd5` |
| BuildSpec | `028c1c0444208a92b16f536e46117b61481df60c7cdc71c538dea0db48f142a7` |
| BuildReceipt | `bdb8fd441b19d646d4c85d574ad0b10b5ee5c5f98146268c3172963175473348` |
| CPU report | `274ff01ffbf962724bbd0590a6a32af05cc167fc6dafcd4a983c0a51ab41df39` |
| CPU test file | `9262d9554b990d0df6767f9cac30d78dd621ea31f8b0ac51834bc309bd3756be` |

The log and returned image agree; the unchanged helper performed canonical durable Volume
readback. Image receipt placement follows the successful build step, without separate image
inspection. Final observations show the build App stopped, zero tasks and no containers;
earlier stopped/one-task observations remain retained. Provisional build cost USD0.00484499
and aggregate USD0.60 metered / zero billed after credits do not settle storage or final costs.
Gross USD5/USD30 limits and remaining-work reserves still apply.

Read-only hash/schema/Git comparisons and decimal cost checks passed; no tests, build, ML or
provider calls were repeated. Prior pin/asset and offline code reviews were reused. No CPU
blocker found. GPU checkpoint completion and the full round require a separate release and
actual evidence; failed/truncated outputs remain preserved, with no research-quality claim.

## Corrected CPU attempt 4 — PASS (actual CPU evidence)

Source `7dd40ec6cc444c939bdd09b42963bbf639cecbca`; release/prior offline review
`77b762f91f82f79df08c2e1115aaa23d5ed9bc75`. Final author record
`0d1c8457aa941e7409707be019f76a0bfca5f71c` and its 25-file index inspected.
**11 passed, 0 skipped, 0 failed; 142.588 seconds on 2026-09-17.**

All final indexed files rehash correctly. Canonical spec/receipt/report and exact seven
runtime Git blobs, test blob and lock/export match the new source. Only the dispatcher
runtime file differs from attempt 3; CPU test, recipe, processor identities and resource
settings are unchanged. All eight small processor assets rehash to their pinned values.
Receipt validation confirms Python 3.11.12, 96 unique sorted packages and every required
pin; remote environment digest matches the accepted prior build. The 829-byte report
contains the same eleven intended case names, now executed by this corrected build.

| Evidence | SHA-256 |
| --- | --- |
| Final index | `b625fbaea15eafacb2defa5ceb4c5d5209c15c38b384ad3c8905a8fdd9be6c83` |
| BuildSpec | `c876d946ac6635d8cd9172bafb7c9873cced759ed9fa6d0248177762919f1ecf` |
| BuildReceipt | `cb477825e80734235813a86eeb98309455e814ba3a19a02ba185fc19609cc550` |
| CPU report | `274ff01ffbf962724bbd0590a6a32af05cc167fc6dafcd4a983c0a51ab41df39` |

The log, returned image and successful helper readback agree. The helper differs only in
checkout and attempt number; its canonical durable Volume readback checks remain intact.
Receipt placement inside the image follows the successful reviewed build step, without
separate image-filesystem inspection. The build App stopped; captured inventory shows no
active containers and the older deployment idle. Attempt cost USD0.00357553 and aggregate
USD0.42 metered / zero billed after credits remain provisional. Preserve the higher prior
USD0.45 observation, separate diagnostic costs and unsettled storage under gross USD5/USD30.

Verification used the read-only hash/schema/Git-blob procedure documented below, including
decimal summation of this App's resource rows. No tests, build, provider or ML call was
repeated. No CPU blocker found. This accepts the corrected CPU image only: the incomplete
GPU round and actual 4B/CUDA behavior still require a separate release and measured evidence.

## Input-verification correction — PASS (offline)

Candidate `7dd40ec6cc444c939bdd09b42963bbf639cecbca`, control
`73e8290841ccc3ba5e4cd2053f67da1c58e43b9d`. Review covers the small dispatcher
change and its three added regression cases; prior CPU acceptance applies to its
historical source/image, not a new image built from this candidate.

The parent retains full bundle membership, size, containment and symlink checks.
Request membership and image SHA-256 binding are checked before selecting files to hash.
Base admission reads no input bodies; fit/predict hash each requested image, with same-size
corruption rejected before the child starts. Unrequested image bodies are deferred until use.
No oracle/GT channel, model pin, timeout, research configuration or Qwen implementation changed.

Source tracing confirms all four child entry paths (`load_base`, `fit`, `predict`,
`verify_reload`) call `_verify_runtime` and its exact pinned model/processor inventory
comparison before processor/model use. Consumed images undergo SHA-256 and geometry checks;
checkpoint/adapter integrity and post-operation checks remain. Thus same-size model corruption
moves from parent detection to child detection before use. This conclusion combines inspected
unchanged code with existing integrity regressions; it is not a new real-model corruption run.

**33 existing focused cases passed in 1.26s** at the clean candidate:

```sh
python -m pytest \
  tests/test_modal_app.py::test_base_parent_does_not_read_input_bodies \
  tests/test_modal_app.py::test_requested_image_same_size_corruption_fails_before_child \
  tests/test_modal_app.py::test_base_completion_and_same_attempt_reuse \
  tests/test_modal_app.py::test_identity_failure_before_any_child \
  tests/test_modal_app.py::test_dispatch_resolves_provider_mounts_but_rejects_descendant_symlinks \
  tests/test_modal_independent.py::test_worker_preconditions_never_start_child \
  tests/test_modal_independent.py::test_worker_dispatch_complete_fit_two_stages_and_reuse \
  tests/test_qwen.py::test_symlink_and_image_mutation \
  tests/test_qwen.py::test_fit_failure_before_processor_or_model \
  tests/test_qwen_independent.py::test_real_identity_failure_precedes_optional_model_import \
  tests/test_qwen_independent.py::test_image_mutation_during_generation_publishes_no_receipt \
  tests/test_qwen_independent.py::test_phase2_prediction_drift_stops_publication --tb=short
```

Affected-file Ruff and diff whitespace checks pass. No new regression or production edit
was needed; no full campaign, provider call, build or ML execution was performed in this review.

Both retained diagnostic indexes and their 17 files each rehash correctly. Independent parsing
of event records reproduces 269 image hashes / 331,833,179 bytes in 139.537446 seconds before
the bounded watchdog stopped the first CPU diagnostic. It did not reach directory/worker/
publish/commit stages. The separate model-only diagnostic completed the unchanged verifier
twice in 31.157210 and 28.569850 seconds, matching all pinned file digests: 13 entries,
12 distinct files, 8,887,285,585 bytes per pass (shared config counted twice).
Model-timing index `b2b8dcde4babcadab35a80ecd12a0272c5d85d5b698538fb488853b529b50a36`;
result `e03a2c06995da9fb992e4958c7c6dbeab6b2b3af024e458dffe0ba36bb6fbd32`.

These CPU/4 GiB child diagnostics support removing redundant reads. They neither locate
the original GPU stall conclusively nor measure model loading or corrected GPU performance.
The first GPU call timed out without a worker result; the actual round remains incomplete.
No CPU/code blocker found for a separately released corrected image build. Actual GPU
completion, fresh runtime evidence and provisional cost reconciliation remain later gates.

## Actual CPU attempt 3 — PASS, with provisional accounting warning

Reviewed source: `1a413d7e373a1182c3552bd9c05c9687fef14485`, released under
control `b517d7062cb2770ed7d1072d3b75533e38cf8414`.
**Measured result: 11 passed, 0 skipped, 0 failed on 2026-09-17.**
Independent review of retained execution evidence supports CPU acceptance; no test,
build, provider call or ML execution was repeated. Prior offline acceptance remains applicable.

The final 21-file evidence index rehashes in full, including the execution helper,
build log, canonical BuildSpec/BuildReceipt/report, provider observations and result.
All seven runtime files and the CPU test match exact Git blobs; lock and requirements
bytes match the spec. All eight small processor assets match the pinned manifest.
The receipt validates against the reviewed schema: Python 3.11.12, 96 sorted unique
effective package names, every required model/runtime pin and Modal 1.5.5.
Source, code-bundle and remote-environment hashes agree with the final author record.

| Evidence | SHA-256 |
| --- | --- |
| Final index | `6612fe53db0fcb9c6865880ddebe51e28460f78639b54a31e50049a7b670de51` |
| BuildSpec | `6a73443392d9c38552afbe279884c9ac5c37cf32030e3a729be0730ec73f9288` |
| BuildReceipt | `23bcabdd3fe6c0a89a36c31b09c9e21e24010bec2c8e9b74558e07d4ea7dc069` |
| CPU report (829 bytes) | `274ff01ffbf962724bbd0590a6a32af05cc167fc6dafcd4a983c0a51ab41df39` |

The eleven unique reported cases match the reviewed selector and parameterization:
four processor grid/template sizes; three tokenization/loss-mask targets; tiny forward,
reset, frozen parameters and fresh-process reload; partial accumulation against manual
Adam; generation defaults; and adapter tensor/config validation. The staged-header test
is excluded. The successful gate requires an exited-zero pytest child, eleven distinct
JUnit cases without failure/error/skip elements, Linux/x86_64 and the exact package pins.
Thus these are actual CPU processor/tiny-model checks, not collection or fixture-only evidence.

The retained log records completion of the same final image returned by the helper.
After successful build, the reviewed helper reads canonical report/receipt bytes from the
output Volume twice and checks equality. This supports durable provider readback. The
successful build step writes identical canonical receipt bytes into the image and commits
the Volume; **no separate image-filesystem readback container was run**. Image receipt
placement is supported by inspected executed code, not an independent image inspection.
The overall attempt elapsed 158.76 seconds, including setup and image work.

Read-only verification used `PYTHONPATH=src python` with `hashlib`, JSON parsing,
`git show <source>:<path>` byte comparisons, reviewed BuildSpec/BuildReceipt validation,
pinned processor-file hashing and decimal sums of retained provider observations.
To reproduce the artifact-integrity portion without executing tests or contacting Modal,
set `EVIDENCE_DIR` to the retained attempt directory:

```sh
python - <<'PY'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ["EVIDENCE_DIR"])
for name, expected in json.loads((root / "handoff-index.json").read_bytes()).items():
    raw = (root / name).read_bytes()
    assert len(raw) == expected["bytes"], name
    assert hashlib.sha256(raw).hexdigest() == expected["sha256"], name
PY
```

The post-build inventory records the matching App stopped with zero tasks and no active
containers. Immediate cost observations remain provisional: this App USD0.00409222;
itemized App rows total USD0.01465614; billing-summary App aggregate USD0.01865657
(a USD0.00400043 difference); rounded metered USD0.02, billed zero, credits minus USD0.02.
These separately captured views are not reconciled final billing. Keep the larger observed
aggregate and reporting-lag caveat for gross USD5/USD30 accounting; remaining credits are
unknown. This accounting warning does not contradict CPU execution or quiescence evidence.

No CPU blocker found. Actual 4B/CUDA loading, training/inference, GPU runtime receipts,
full real-round recovery and OCR quality remain unverified and separately gated. No active-
learning improvement or annotation-time result follows from this CPU acceptance.
The [author execution record](modal-cpu-runtime.md) and earlier failures remain preserved.
The historical offline sections below describe their evidence state at the time.

## Inventory correction — PASS (offline)

Code: `1a413d7e373a1182c3552bd9c05c9687fef14485`.
Control: `a4ee183ddb1c33c847b64f860ebb8f1bca122870`.
The three-file correction resolves each normalized distribution name explicitly through
`importlib.metadata.distribution(name)`, then records unique names in sorted order.
It does not choose the highest version, discard packages or rely on enumeration order.
Receipt schema, canonical hashing, complete worker/build inventory equality and exact pins
are unchanged; the next build must record the newly resolved inventory in a fresh receipt.

Retained diagnostic SHA-256
`b0f544826634d40dc7759936dd40a724f30c07db4041214c68ae0c7fb9e37ea0`
verified, along with all nine indexed evidence files. The inspected diagnostic launches a
fresh metadata-only child with the worker's executable/cwd/`-m`/environment pattern.
Whole parent/child captures match: 113 records, 96 normalized names, 17 duplicate groups
(16 with conflicting versions). Every effective model/runtime pin matches; no discovered
name is excluded. Neither capture contains imported ML modules. This establishes metadata
resolution, not provenance of already-imported module objects.

**6 passed in 0.18s**, using the existing real dist-info regression in both path orders,
worker identity rejection, CPU receipt construction for both mount forms and strict receipt
validation:

```sh
python -m pytest \
  tests/test_qwen.py::test_installed_packages_records_effective_search_path_versions \
  tests/test_qwen.py::test_worker_manifest_no_git_checks_actual_code_and_packages \
  tests/test_modal_app.py::test_cpu_build_writes_measured_receipt_and_commits_only_output \
  tests/test_modal_model.py::test_cpu_receipt_cannot_pass_with_skips_or_different_build --tb=short
```

Affected-file Ruff and diff checks pass; pins/lock, shared receipt and worker boundaries are
byte-unchanged. No blocker, new test, broad campaign, ML execution or cloud action occurred.
The prior eleven actual CPU passes apply to the earlier candidate only. This new candidate
has not run remotely; phase-B receipt/image acceptance and a build retry remain separately gated.

## Phase A — PASS for the two offline runtime corrections

Code: `7ea96c3b9b10612a0751ff78a668cbe076c18ef9`.
Control: `5fb5c94c13cfcee92d0195b795d411f135fe0dd7`.
This scoped verdict reuses the [accepted integration review](modal-runtime-review.md).
It is **not** a successful CPU build or permission to retry one.

The normalization diagnosis matches the exact lock-pinned Transformers 5.16.1 wheel
(SHA-256 `2f2d5b98a5ad3718713653734298fa620754ed683702a635ebb587df3ed29c7e`).
`Qwen2VLImageProcessor` inherits `TorchvisionBackend`; initialization calls
`BaseImageProcessor._set_attributes`, which calls `_standardize_kwargs` before assigning
attributes. That method converts mean/std lists to tuples. The pinned processor asset has
three `0.5` values for both. The correction changes only the expected representation;
bicubic, normalization values, patch/token checks, pins and recipe remain unchanged.
The existing loader regression accepts those tuples and rejects changed values; diagnostics
name only the controlled setting key. Source inspection imported no ML runtime.

The retained filesystem diagnostic index and all ten indexed files rehash correctly.
Its durable record hash is
`a67b225ed8b42f202bc59df41b7776a6910eb80fb04ed1025c9010b51e5c50bd`.
The diagnostic records mount-symlink rejection, hardlink `EPERM`, no-replace rename `EINVAL`,
and successful fsync/commit. These observations justify replacing the former publication path;
they do not demonstrate that the new implementation has run on the provider.

The fix resolves only the fixed provider mount roots. Descendant containment/symlink checks
remain active. Publication uses exclusive directory reservation, a complete fsynced temporary
file and rename; same-byte reuse succeeds, conflicts and busy/stale reservations fail closed.
Existing tests exercise competing cooperative publication before rename, atomic visibility,
fsync failure cleanup, trusted mount resolution and descendant rejection. This is bounded
single-dispatcher ownership, not a distributed lock or protection from non-cooperating writers.
Persistence failures now disclose stage/type/errno without raw exception text or secrets.

Verification at the clean exact candidate: **162 passed, 24 deselected in 5.04s**:

```sh
python -m pytest tests/test_modal_app.py \
  tests/test_qwen.py::test_load_processor_accepts_canonical_tuple_normalization_and_rejects_drift \
  tests/test_modal_independent.py \
  -k 'not full_round and not unmocked_clean and not real_baseline and not bad_subset and not frozen_image and not receipt_retrieval and not bootstrap_allowlist' \
  --tb=short
```

Affected-file Ruff, diff whitespace and unchanged lock/pyproject checks passed.
No uncovered defect required a new test or production change. The prior full integration
campaign was not rerun; author full-suite results are not claimed as independent execution.

The [first actual CPU attempt](modal-cpu-runtime.md) remains **FAIL**: its retained report
has eleven setup errors and no passing ML test bodies. A corrected CPU attempt, durable
receipt/report and observed image identity require a separate release and phase-B review.
No provider call, cloud job, ML installation/loading or spend occurred in this review.
