# Local contract independent review

Verdict: **FAIL**, limited to candidate
`3c86330b82121bd86c7e6a9d926103d8d62b23f0` and the accepted plan's candidate 1A.
Parent/control: `cce1f14c9bb107b2a1b9dc4c0f8c94c942f37122`.
This review uses synthetic pages and adapters only. It establishes no real model, corpus,
CUDA, checkpoint-byte, or provider result.

## Findings

1. **P1: failed output can make the committed run unreadable.** In
   `Pipeline._validate_simulation_predictions`, finite geometry validation runs only for
   `status=ok`. `Box(x=inf, y=0, width=1, height=1)` is constructible, and an
   `invalid_output`, `truncated`, or `refusal` prediction containing it passes the boundary.
   Both the baseline and acquired-round step commit successfully. Pydantic serializes the
   infinity as JSON null; the next `get_simulation`, resume, or export raises a validation
   error at the persisted box coordinate. The acquired-round case also makes the previously
   usable baseline inaccessible through the normal API. Failure counts do not prevent this.

   Require persisted structured predictions to round-trip for every status. Either reject
   nonfinite structured geometry before commit or discard malformed structured regions on
   failed predictions while preserving status and raw evidence separately. Full valid-layout
   requirements may still differ by status. Do not merely suppress the reload exception.
   Regression: `test_failed_nonfinite_geometry_cannot_publish_unreadable_state`, six cases.

2. **P2: legacy scoring accepts validation-purpose predictions as pool output.** A positive
   round `purpose=validation` result with matching run/model/page coverage passes `poll_job`;
   it is saved and then returned by `_predictions_for` for acquisition. The new shared
   `Prediction` type makes this result legal, but the unchanged legacy boundary checks only
   run/round/model. Baseline-purpose rejection through the round-zero invariant does not
   cover this positive-round case. This violates the plan's separation of prediction purposes.

   Require pool purpose before legacy scoring publishes anything; consider the same filter
   when reading stored acquisition predictions. Manager must explicitly assign the small
   legacy-boundary adjustment because that function was outside Development's original scope.
   Regression: `test_legacy_scoring_rejects_validation_purpose_before_publication`.

No production or author-test files were changed by this review. The regressions remain failing
assertions, not xfails or tests that declare the defects acceptable.

## Evidence

Python 3.11.14, Pydantic 2.13.5, pytest 9.1.1, Ruff 0.16.6; no dependencies installed.

- Author suite independently executed on the clean exact candidate: **155 passed in 4.26s**.
  `ruff check .` and candidate `git diff --check` passed. No type checker is configured.
- Independent suite against a separate clean exact-candidate source snapshot:
  **7 failed, 15 passed in 4.90s**. Six failures reproduce finding 1 across both step types
  and all three failed-output statuses; one reproduces finding 2.
- Actual SQLite `AFTER UPDATE` trigger aborts and separate-connection baseline/round races
  passed. Retry round-trips; one competing writer commits and the other gets a stale-state error.
- Historical compatibility passed using a database actually written by parent application
  code, including Unicode source/metric keys, float metrics, pool scores, and old validation
  records without purpose. The candidate's `exclude_unset`/`RootModel` expected value matched
  the historical JSON; resume retained old history and added a new validation-purpose round.
- Unpatched local identity checks passed for clean operation, actual untracked-file drift,
  an additional discovered distribution, completed resume, and fresh Python processes.
  The temporary changes were removed. No identity function was mocked in the independent suite.
- Zero paths, hidden text/geometry perturbation, reversed manifest order, matched fixture
  sampling, cumulative 2+1 page budget, Unicode/micro/undefined counts, nonfinite evaluator
  rejection, varying valid telemetry, separate-process baseline/round resume and export passed.
  Missing real adapters and all author ownership/geometry/failure probes also passed in the
  author suite. Normal telemetry rejects nonfinite timing/cost values; evaluator NaN and
  either infinity reject atomically. Those guards do not protect failed region geometry.

| Plan check | Independent assessment |
| --- | --- |
| L1 compatibility | Original suite and actual prior-writer resume pass; legacy purpose boundary fails. |
| L2 baseline | Creation/no model work, baseline-only first step, zero budget/round/train paths pass. |
| L3 atomicity | Injected call/evaluator failures and actual storage rollback/races pass; malformed failed outputs can publish unreadable state. |
| L4 ownership | Simulation purpose/coverage/successful geometry checks pass; findings 1 and 2 remain. |
| L5 sampling | Paired fixture batches, acquired-round seed offsets and cumulative unique budgets pass. |
| L6 identity | Real local code/dependency checks and author recipe/adapter mutation probes pass; telemetry excluded as intended. |
| L7 text counts | Golden Unicode, blank/whitespace, micro counts, failed hypotheses and independent undefined denominators pass. |
| L8 isolation | Only selected TRAIN targets reach fit; validation metadata reaches predict; hidden/validation perturbations do not change fit IDs or selections. |
| L9 resume/export | Normal and historical processes/export pass; finding 1 blocks failed-output resume/export. |

The implementation retains shallow modules, one coordinator, one adapter protocol and the
existing store. The two commit helpers and historical serialization workaround address concrete
requirements. No additional orchestration framework is warranted for the fixes. Test doubles
exercise metadata/ownership contracts, not actual model pin verification or learning.

## Reproduction

Use an environment with the project's local development dependencies. Keep the review tests
outside the clean candidate checkout so real Git identity checks remain meaningful. From the
checkout containing this review, run these commands, substituting that environment's Python
and Ruff executables if necessary:

```sh
review_dir=$(mktemp -d)
review_tests="$PWD/tests/test_contract_independent.py"
git clone --quiet --no-local --no-checkout . "$review_dir/source"
git -C "$review_dir/source" checkout --quiet --detach 3c86330b82121bd86c7e6a9d926103d8d62b23f0
PYTHONPATH="$review_dir/source/src" python -m pytest \
  -c "$review_dir/source/pyproject.toml" -o pythonpath= --import-mode=importlib \
  "$review_tests" --tb=short
```

Expected candidate result: seven failures, fifteen passes. Use `-k failed_nonfinite` or
`-k legacy_scoring` for the minimal defect families. The historical-writer check reads the
parent commit through `git archive`; retain that commit in the source clone. Tests temporarily
write an untracked identity marker only inside the disposable checkout and remove it in `finally`.
They run sequentially; do not use parallel pytest workers with that shared identity marker.

After Development fixes the findings, repeat affected regressions against the new exact candidate,
then the relevant full suite and Ruff. Acceptance and any scope change remain Manager decisions.
