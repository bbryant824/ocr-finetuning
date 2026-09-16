# Independent Qwen runtime review

Verdict: **FAIL for the offline candidate**
`c4d9e09770f1c31c75a6078898dd342da63f5469`, including implementation commit
`17e8167084ae5b437e32c3ab071a01a98b109bb6`.
Control/accepted specification: `d8ee28cf6ebc02cbe6fe949a96ffb3c0c100f3e9` and
[candidate 2, Q1–Q9](../../plans/PLAN-003-real-ocr-modal.md#candidate-2--concrete-qwen-runtime-training-and-checkpoints).

Two independent regressions remain. No production or author-test changes were made. The
existing passing suite does not override these findings. Linux processor/tiny-model execution
and the later 4B/CUDA checks remain **unverified**, not failed or passed by this offline review.

## Findings

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

## Executed evidence

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

## Reproduce the independent suite

Keep the review test file outside the clean candidate checkout and use the existing development
environment. This command needs no real dataset, model assets or optional ML packages:

```sh
review_tests="$PWD/tests/test_qwen_independent.py"
review_dir=$(mktemp -d)
git clone --quiet --no-local --no-checkout . "$review_dir/source"
git -C "$review_dir/source" checkout --quiet --detach c4d9e09770f1c31c75a6078898dd342da63f5469
PYTHONPATH="$review_dir/source/src" python -m pytest \
  -c "$review_dir/source/pyproject.toml" -o pythonpath= --import-mode=importlib \
  "$review_tests" --tb=short
```

Expected at this candidate: **24 passes, 2 failures** named above. Development owns corrections;
Testing will review a separately released corrected SHA. Manager retains acceptance and all
later CPU-build/CUDA releases.
