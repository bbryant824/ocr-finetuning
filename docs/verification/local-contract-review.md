# Local contract independent review

Verdict: **PASS**, limited to corrected candidate
`dee5e1a5ecbbc326921e0f4019792c8f7e5deab2` and the accepted plan's candidate 1A/L1–L9.
Control: `dec843196a0edba537a05ee26345d814ab525b45`.
This review uses synthetic pages and adapters only. It establishes no real model, corpus,
CUDA, checkpoint-byte, or provider result. Manager owns acceptance and subsequent releases.

## Findings resolved

The previous **FAIL** applies to `3c86330b82121bd86c7e6a9d926103d8d62b23f0`.
Its original review and failing regressions remain in commit
`5d0ac216da821bdc92ac143a0eb4ee4d1f7491f3`; this review does not retroactively pass that code.

- **P1, unreadable failed predictions: resolved.** The simulation boundary rejects nonfinite
  coordinates for every status, before publishing a baseline or acquired round. Previously,
  failed-status infinity became JSON null and prevented reload. All four statuses now refuse
  malformed structured payloads atomically. Finite failed records retain partial regions, status,
  raw-artifact reference and finish reason, and are scored as empty hypotheses with failure counts.
  Successful contract predictions retain their additional bounds/unique-region checks.
- **P2, legacy prediction purpose: resolved.** The legacy scoring boundary verifies pool purpose
  across the complete batch before publishing any prediction or experiment state. Acquisition
  reads also exclude non-pool records. A wrong-purpose result last in a batch leaves all state
  unchanged. Missing purpose in historical JSON still means pool. Missing genuine pool scores
  fail acquisition without advancing; validation scores are not silently used as replacements.

The correction is small: two purpose guards and one status-independent finite-coordinate guard
in the existing coordinator, associated tests, and four documentation lines. No new store,
service, abstraction or production changes by Testing. The ordinary merge parent preserves the
failed candidate and the Manager's explicit narrow correction authorization.

## Executed evidence

Python 3.11.14, Pydantic 2.13.5, pytest 9.1.1, Ruff 0.16.6; no dependencies installed.
All execution below used an unchanged clean checkout at the exact corrected candidate.

| Check | Observed result |
| --- | --- |
| Full candidate suite, `python -m pytest` | 197 passed in 4.93s |
| Candidate `ruff check .` and corrective `git diff --check` | Passed |
| Original independent suite, unchanged from the prior review | 22 passed in 4.37s |
| Added independent correction cases, `-k correction` | 18 passed, 22 deselected in 7.17s |

The original test file's SHA-256 before additions was
`213d19d5b15456929133b08de6ef2c56fcf07577351f3c04225033b7244a4d1d`.
Its bytes remain an unchanged prefix; additions do not weaken, replace or xfail old regressions.
The combined independent file has 40 cases, verified in the two invocations above. No configured
type checker exists. Author tests were inspected and executed, never edited by Testing.

Additional cases send raw dictionaries with NaN, positive infinity and negative infinity at
x/y/width/height, for every status at both baseline and acquired-round steps: 96 rejection
attempts. Each leaves the prior run readable; export equals prior state and a valid retry works.
Six finite failed-output cases prove preservation and JSON round-trip at both step types.
Three mixed legacy-batch cases check wrong-purpose-last rejection, explicit pool compatibility
and omitted-purpose compatibility. A stored-purpose case verifies the actual acquisition API
rejects missing genuine scores, then accepts historical purpose-omitted JSON.

| Plan check | Assessment at corrected candidate |
| --- | --- |
| L1 compatibility | Full original/legacy suite and actual prior-version writer resume pass; purpose guard regression resolved. |
| L2 baseline | Creation makes no model call; first step is baseline-only; zero budget/round/train paths pass. |
| L3 atomicity | Original call/evaluator failure tests, actual SQLite trigger rollback and separate-connection baseline/round races pass; malformed output preserves readable state. |
| L4 ownership | Simulation purpose/coverage/layout checks and corrected legacy purpose/all-status finite boundaries pass. |
| L5 sampling | Matched fixture batches, acquired-round seed offsets and cumulative unique 2+1 budget pass. |
| L6 identity | Unpatched code/dependency checks, actual untracked-file/distribution drift and author recipe/adapter mutation probes pass; varying valid telemetry remains excluded. |
| L7 text counts | Unicode, blank/whitespace, micro edits, failed hypotheses and independently undefined CER/WER denominators pass. |
| L8 isolation | Selected TRAIN targets only reach fit; validation metadata reaches predict; hidden text/geometry and validation perturbations preserve selections and model IDs. |
| L9 resume/export | Fresh-process baseline/round/completed resume, actual prior-writer Unicode/float JSON CAS, and normal/failed-output export pass. |

Historical CAS verification runs parent application code to write its actual database bytes,
rather than reconstructing legacy JSON using the new model. Old validation purpose remains pool
in stored history; new validation records have explicit purpose. Independent identity tests do
not patch the identity function. They use the actual candidate source SHA and installed versions,
then check real filesystem/distribution changes and restore them. Original rollback/concurrency
cases use actual SQLite triggers and separate connections. Local CAS still does not guarantee
exactly-once remote side effects.

## Reproduction

Use an environment with the project's local development dependencies. Keep the review tests
outside the clean candidate checkout so real Git identity checks remain meaningful. From the
checkout containing this review, substitute the environment's Python executable if necessary:

```sh
review_dir=$(mktemp -d)
review_tests="$PWD/tests/test_contract_independent.py"
git clone --quiet --no-local --no-checkout . "$review_dir/source"
git -C "$review_dir/source" checkout --quiet --detach dee5e1a5ecbbc326921e0f4019792c8f7e5deab2
PYTHONPATH="$review_dir/source/src" python -m pytest \
  -c "$review_dir/source/pyproject.toml" -o pythonpath= --import-mode=importlib \
  "$review_tests" --tb=short
```

Expected result: all 40 independent cases pass. To reproduce the original 22 alone, extract
`tests/test_contract_independent.py` from commit `5d0ac216da821bdc92ac143a0eb4ee4d1f7491f3`
outside the source checkout; its hash is above. The historical-writer case needs parent commit
`cce1f14c9bb107b2a1b9dc4c0f8c94c942f37122` available to `git archive`.
Tests temporarily create an untracked identity marker in the disposable checkout, remove it in
`finally`, and run sequentially. Do not use parallel pytest workers with that shared marker.

## Limits

No remaining blocking finding was reproduced in the released local-contract scope. Immutable
model/checkpoint values are declared synthetic adapter metadata, not verified weight bytes.
Fixture resume retains its documented unchanged-environment assumption. Source admission,
actual training/loss/gradients, CUDA, remote jobs/recovery/cost, layout matching, uncertainty and
final evaluation remain outside this verdict. A later implementation change needs its own exact
candidate and proportionate independent review; these tests are not OCR or scientific evidence.
