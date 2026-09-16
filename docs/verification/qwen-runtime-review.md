# Independent Qwen runtime review

Current verdict: **PASS for the reviewed offline scope** at corrected candidate
`31bb47224c03b0db8f7f9705028e29e1d13e2c0e`.
Control: `58c34c9e9b8460e9b19af0a23d5c8af1f78789aa`; the accepted specification and production
in that control are unchanged from the earlier review. Both original failures below are resolved.
Actual Linux processor/tiny-model checks and 4B/CUDA execution remain **unverified** and separately
gated. This is not model-runtime, OCR-quality or experiment acceptance.

## Phase 2 correction and independent evidence

Reviewed the complete correction to model code, author tests and runtime documentation. The
original 26 independent tests and phase 1 review were unchanged by Development. All 26 now pass
without changing assertions or using xfail. This review adds 15 affected-scope cases to the same
independent test file; no production, author-test or dependency changes were made by Testing.

The correction retains the verified adapter file inventory and tensor hashes across image
preparation and loading. `_load` compares the loaded state with that retained identity, rather
than deriving a replacement expectation from current disk contents. Prediction checks the
requested manifest/files before and after loading, before each generation, and after the final
runtime/image/telemetry checks before publishing its receipt. Fit retains its staged inventory
through reload and publication. A failed binding clears the previous identity; loading base
clears the adapter binding. The added helper reuses existing inventory validation and introduces
no new service or general identity framework.

The JSON correction is deliberately local: only a recursion-depth exception from `strict_json`
inside `parse_regions` becomes the existing invalid-output path. Tokenizer recursion, including
failure during the second content decode, remains an operation error. Raw generated text and
EOS termination remain available for the invalid page record.

| Check at the corrected candidate | Independent result |
| --- | --- |
| Full clean candidate `pytest --tb=short -rs` with staged header-only input | **467 passed, 11 unverified skips in 40.99s** |
| Expanded independent suite against a separate clean exact source checkout | **41 passed in 0.47s** |
| Ruff, correction/review whitespace and ownership checks | Passed |

The full regression ran before any new review edits. The separate execution checkout remained
clean. Environment versions are unchanged from phase 1. Dependency pins/lock are byte-identical
to the previously checked candidate, so no redundant lock resolution or installation was needed.

The new independent cases cover:

- Adapter-file and manifest drift during loading, generation and final observation; no prediction
  receipt is published, and load-time drift prevents forward work.
- The actual `_bind_adapter` and `_load` methods with explicit package/tensor spies: unchanged
  loading succeeds, then base loading clears the binding; replacement loaded into memory is
  rejected even when original disk bytes are restored; drift after tensor capture also rejects.
  The tensor validator is called only when binding, not to replace expected hashes during load.
- Mutation during binding clears retained identity rather than leaving a stale successful one.
- The actual `fit` control flow with training/reload spies and real staging/inventory/publication:
  unchanged bytes publish a checkpoint whose files equal the retained inventory; drift after
  reload or during receipt creation cannot publish a completed checkpoint. Private staging is
  cleaned in all three cases.
- Tokenizer recursion during either raw or content decoding still propagates. The unchanged
  nested-JSON regression now returns `INVALID_OUTPUT`, empty regions and raw evidence.

These spies test ordering and integrity, not actual tensor decoding, optimization, loaded-model
numerics or CUDA. The optional actual-adapter test was inspected: it now writes a different
finite, correctly shaped safetensors payload and expects inventory rejection before CUDA, but
remains one of the 11 unexecuted ML cases. The staged-build and 4B/CUDA gates described below
remain required. Synchronous checks do not provide a filesystem write lock against a malicious
concurrent writer; the planned single dispatcher and input/output ownership still apply.

No blocking defect remains in this affected offline scope. Manager retains acceptance,
integration and subsequent release authority. No ML install/load, dataset experiment, build,
cloud action or spend occurred during either review phase.

To reproduce all 41 independent checks, keep the review file outside the clean source checkout:

```sh
review_tests="$PWD/tests/test_qwen_independent.py"
review_dir=$(mktemp -d)
git clone --quiet --no-local --no-checkout . "$review_dir/source"
git -C "$review_dir/source" checkout --quiet --detach 31bb47224c03b0db8f7f9705028e29e1d13e2c0e
PYTHONPATH="$review_dir/source/src" python -m pytest \
  -c "$review_dir/source/pyproject.toml" -o pythonpath= --import-mode=importlib \
  "$review_tests" --tb=short
```

## Preserved phase 1 review

Historical phase 1 verdict: **FAIL for the offline candidate**
`c4d9e09770f1c31c75a6078898dd342da63f5469`, including implementation commit
`17e8167084ae5b437e32c3ab071a01a98b109bb6`.
Control/accepted specification: `d8ee28cf6ebc02cbe6fe949a96ffb3c0c100f3e9` and
[candidate 2, Q1–Q9](../../plans/PLAN-003-real-ocr-modal.md#candidate-2--concrete-qwen-runtime-training-and-checkpoints).

At that candidate, two independent regressions remained. No production or author-test changes were made. The
existing passing suite does not override these findings. Linux processor/tiny-model execution
and the later 4B/CUDA checks remain **unverified**, not failed or passed by this offline review.

## Historical findings (corrected in phase 2)

### P2: checkpoint bytes are not bound across preparation and loading

In `QwenModel.predict`, `_checkpoint` verifies the requested manifest and files before image
preparation. The later `_load(directory)` checks adapter configuration/tensor shapes/finite
values again, but derives its expected tensor hashes from the directory's *current* contents.
It does not compare those bytes with the requested manifest inventory. Final checks repeat
worker/base/processor identity and image validation, without rechecking this checkpoint.

The independent test
`test_adapter_mutation_between_validation_and_load_is_rejected` changes the adapter file from
an image-preparation callback, after the real initial checkpoint/ownership/file validation.
Prediction reaches the load boundary and publishes a receipt/result under the original model
ID instead of rejecting changed bytes. Expected refusal fails with `DID NOT RAISE ValueError`.

This is a filesystem/call-order reproduction using synthetic adapter bytes and explicit ML
spies: tensor validation, actual loading and generation are not executed. The code consequence
for a replacement adapter that still has valid configuration/shapes/finite values is an
inference from the inspected `_load` path: it can be accepted against its own new hashes and
attributed to the earlier checkpoint. The reproduction does not claim a corrupted safetensors
file would pass the real tensor parser, nor that mutation occurs in an ordinary unchanged run.

Development should bind loading to the already verified inventory/tensor identity, detect
checkpoint drift before forward and before publishing results, and preserve the existing
content-addressed files. Keep this bounded to the model boundary; no new service is needed.
Relevant candidate code: `qwen.py` `_load` (1024–1065), `predict` (1132–1140, 1180–1197).

### P2: nested invalid JSON aborts the operation without a page failure record

`decode_result` catches `ValueError`, `UnicodeError` and `OverflowError` around generated
response parsing. Python's JSON decoder instead raises `RecursionError` for sufficiently
nested input. For the response below, the exception escapes, so no `INVALID_OUTPUT` result
with empty regions and raw evidence is produced:

```python
raw = '{"regions":' + '[' * 1000 + ']' * 1000 + '}'
```

`test_nested_invalid_json_is_page_failure_not_operation_error` reproduces this with an EOS
terminated character-tokenizer test double: 2,013 IDs, within the 2,048-token cap. The payload
is deliberately incompatible with the output schema. This proves parser exception handling,
not its likelihood under the pinned tokenizer/model. A real model emitting the same decoded
text reaches the same JSON decoder.

Development should classify bounded parser-depth failure as invalid generated output while
retaining raw evidence. Do not turn unrelated infrastructure failures into blank successful
pages. Relevant candidate code: `qwen.py` `decode_result` (394–402).

## Historical phase 1 executed evidence

Python 3.11.14, pytest 9.1.1, Pydantic 2.13.5, Pillow 12.3.0, Ruff 0.16.6. Source imports were
explicitly from the reviewed checkout. No ML installation, model loading, build, cloud call,
download of model assets or paid execution occurred.

| Check | Observed result |
| --- | --- |
| Clean exact candidate, full `pytest --tb=short -rs`, with the staged header directory supplied | **430 passed, 11 skipped in 40.70s** |
| Independent suite against a separate clean exact-candidate checkout | **24 passed, 2 failed in 3.55s** |
| Exact-candidate Ruff and diff whitespace checks | Passed |
| Independent test Ruff | Passed |
| `uv lock --check --offline` with the existing cache | Passed; 77 packages resolved, no installation |

The first independent run had 21 passes and the same two failures. Three further checks covered
training-loop partial groups, original-size reload geometry and interrupted publication; all
passed. Neither failure is suppressed, relaxed or marked xfail. Only owned review tests/docs
were added; the full candidate suite was run before those untracked files could affect its
clean-source identity checks. No redundant full regression was run after documentation edits.

Independent checks construct their own pages, selected targets, malformed outputs, checkpoint
records and spies. They do not import author helpers. Identity-negative, actual file/hash,
strict Pipeline geometry, parser and serialization checks execute production code. Generation,
tensor and optimizer spies test control flow only; they are not model numerics or GPU evidence.

| Acceptance area | Evidence and limits |
| --- | --- |
| Q1 identity and lazy import | Fresh interpreter imports no Torch/Transformers/PEFT/torchvision/safetensors. An invalid actual worker allowlist fails before processor/model calls. Existing author fixture verifies synthetic successful package/code identity and mutation; no genuine Linux worker pass is claimed. Lock pins/Linux wheel metadata were inspected, not installed. |
| Q2–Q3 processor, serialization, masks | Independent literal decomposed Unicode, CR/LF/tab/NUL, quotes/backslashes, empty text and control-token-looking text round-trip. Source IDs/illegibility/document hints are absent. Exact 2,048+4,096 mask limits, causal first/last positions, EOS and overflow/prefix refusal pass. Actual processor/tokenizer/grid/loss execution remains skipped. |
| Q4 actual tiny model | Optional two-layer FP32/math-SDPA Qwen+PEFT tests were inspected: actual forward/backward, hand-selected causal losses, frozen hashes, seeded reset and fresh-interpreter reload are present. None executed here. |
| Q5 reset/trainability | Actual header-only author test passes for all 72 real q/v shapes, 144 planned A/B tensors and 5,898,240 parameters. Independent training-loop spy verifies three epochs of six pages use denominators 4 then 2 inside the backward context and six optimizer steps, plus the declared AdamW options. Actual autograd, Adam values, 4B trainability and reset remain runtime gates. |
| Q6 decoding/geometry | Independent duplicate/nonfinite/surrogate/trailing-output rejection retains raw evidence. 912 right/bottom-edge cases, including both previously reported cancellation examples, pass unchanged strict Pipeline bounds without rounding or clamping. Nested JSON fails as described above. |
| Q7 checkpoints/reload | Real manifest/inventory checks reject missing/extra/modified/symlink files, wrong owner/round and interrupted manifest-last publication. Requested adapter then base reaches the respective load paths under spies. The actual reload-probe function maps a known normalized box using original 101×1,237 dimensions under token/logit spies, confirming the final geometry correction. Between-check/load mutation fails. Actual safe tensors/fresh-process model equality remain gated. |
| Q8 isolation | Held-out fit examples stop before preparation; both selected targets are preflighted before any base load. Prediction preparation receives image pixels only and receipts contain no source/document fields. During-generation image mutation prevents receipt publication. No oracle is called. These are interface/call-flow checks, not proof against a malicious adapter. |
| Q9 4B/CUDA smoke | Not executed or released by this review. |

## Optional runtime checks and simplicity

The 11 skips are four actual processor geometry cases, three actual tokenization/mask cases,
one tiny-model/reset/fresh-process case, one partial-accumulation numerical reference, one
actual generation-config case and one full adapter tensor/config validation case. Their fixture
requires pinned small processor assets and exact package versions. Missing inputs explicitly
skip as UNVERIFIED; the later build must inspect skip counts rather than accept pytest's exit
code alone. CPU tests intentionally use tiny dimensions/rank, math SDPA and a two-token forced
termination reload probe; they do not test the production CUDA/4B recipe.

Inspection found the intended reset path drops prior model state, resets RNG, creates fresh
base/LoRA/optimizer, preflights all selected targets, checks frozen hashes and adapter changes,
and performs a fresh-model same-process reload before publication. Actual execution of those
library boundaries is still required; spies cannot certify it. The independent fresh-process
4B reload and fixed logit tolerance remain later smoke checks.

The complete five-file implementation diff was reviewed. One fixed synchronous model module
contains the recipe, parser, identity checks, processor/training helpers and checkpoint I/O;
no new service, registry, generic framework or Pipeline/selector/evaluator change was added.
Its size mostly reflects the accepted contract. The findings call for local corrections,
not another abstraction layer. Original source labels/splits and prior converter evidence are
unchanged. No document-independence, OCR quality, training adequacy or active-learning result
is established.

## Reproduce the historical failures

Keep the review test file outside the clean candidate checkout and use the existing development
environment. This command needs no real dataset, model assets or optional ML packages:

```sh
review_tests="$PWD/tests/test_qwen_independent.py"
review_dir=$(mktemp -d)
git clone --quiet --no-local --no-checkout . "$review_dir/source"
git -C "$review_dir/source" checkout --quiet --detach c4d9e09770f1c31c75a6078898dd342da63f5469
PYTHONPATH="$review_dir/source/src" python -m pytest \
  -c "$review_dir/source/pyproject.toml" -o pythonpath= --import-mode=importlib \
  "$review_tests" -k "not phase2" --tb=short
```

Expected for the original 26 tests at that candidate: **24 passes, 2 failures** named above.
The phase 2 tests are excluded because they exercise the later correction. These failed results
remain historical evidence; the corrected candidate verdict and remaining gates appear above.
