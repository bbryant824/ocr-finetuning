# Independent CPU runtime review

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
