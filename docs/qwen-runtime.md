# Pinned Qwen runtime

`integrations/qwen.py` implements the candidate-2 boundary in the accepted
[real OCR plan](../plans/PLAN-003-real-ocr-modal.md). This is implementation and offline
contract evidence, **not executed OCR, a successful Linux build, or CUDA evidence**.
Pipeline REAL enablement, transport, worker deployment, operation journaling and CLI
construction remain candidate 3. There is no automatic fixture or CPU fallback.

## API and inputs

One synchronous `QwenModel(input_root, output_root, real_config, runtime_manifest=...)`
implements `load_base`, cumulative reset `fit`, and `predict`. Both roots must already exist,
be disjoint and have no symlink components. Inputs live under `input_root/model` and
`input_root/images`; image URIs may be absolute worker paths or relative to the images root.
The later transport owns conversion from safe remote image keys to this representation.
Only outputs under the separate writable root are created. One dispatcher serializes calls.

Construction validates records and paths without importing ML packages or loading weights.
Each operation verifies worker identity and pinned asset bytes before model work. Fit accepts
only nonempty, unique selected TRAIN examples, a uint32 seed and positive round. Prediction
uses only image URI, ID, dimensions, hash and split. It never reads `source_image`, document
identity, XML, oracle targets or region counts. Page bytes, RGB mode, identity EXIF orientation,
size, containment and split are checked before preparation and again before publication.

Round 0 requires `BASELINE_VALIDATION`, VALIDATION pages and a base reference. Positive
rounds require that run/round's adapter; `VALIDATION` takes validation pages and `POOL` takes
TRAIN pages. TEST is never accepted. Empty predictions return empty only after identity,
checkpoint and purpose validation. Missing assets, invalid identity, OOM or unsupported CUDA
raise operation errors; they cannot become blank successful predictions.

## Frozen worker identity, without Git

The coordinator's existing local Git/package check is unchanged. A worker verifies an
explicit schema-1 JSON document whose **parent is the code bundle root**:

```json
{
  "schema_version": 1,
  "source_sha": "<40 lowercase hex characters>",
  "files": [
    {"filename": "active_ocr/__init__.py", "bytes": 123, "sha256": "<64 hex characters>"}
  ],
  "environment": {
    "python": "3.11.14",
    "packages": [["accelerate", "1.14.0"], ["other-installed-distribution", "version"]]
  },
  "build_spec_sha256": null,
  "deployment_reference": null
}
```

This is a shape example, not usable identity data. `files` is a complete reviewed code
allowlist, sorted by normalized relative filename. It must contain the actually executing
Qwen, models and package initializer files. All listed bytes are rehashed. The record cannot
point outside its parent, use symlinks, duplicate entries or substitute a different package
copy. The future builder supplies any additional dispatcher files in the same allowlist.

`code_bundle_sha256` is the SHA-256 of canonical JSON
`{"source_sha": ..., "files": [...]}`. Canonical JSON means sorted keys, compact separators,
UTF-8, no NaN and no ASCII escaping. `remote_dependency_sha256` hashes the complete
`environment` object. `installed_packages()` obtains Python's version and one effective
version per discovered distribution name, lowercased with runs of `[-_.]` normalized to `-`
and sorted by name. Each version comes from `importlib.metadata.distribution(name)`, using
Python's metadata search-path precedence. Provider bootstrap paths can expose additional,
shadowed copies with equal or conflicting versions; these are not selected by highest
version, enumeration order or package exclusions. A measured Modal parent/fresh-child
diagnostic found identical effective inventories despite 17 duplicate names across image
and bootstrap paths. Build and worker checks still compare the complete effective inventory
and enforce exact pins; this is metadata identity, not proof of imported module provenance.
There is no comparison of remote packages with the coordinator's local
`dependency_sha256`. Code SHA, build and deployment declarations must match ExpectedIdentity;
null build/deployment values are permitted by that existing record, not evidence of a build.
Candidate 3 must freeze actual reviewed build/deployment values before execution release.

The runtime additionally requires Linux x86-64, Python 3.11 and these exact versions:

| Package | Version |
| --- | --- |
| torch / torchvision | 2.14.0 / 0.29.0 |
| transformers / peft | 5.16.1 / 0.20.0 |
| accelerate | 1.14.0 |
| tokenizers / safetensors / Pillow | 0.23.2 / 0.8.0 / 12.3.0 |

The optional GPU group and universal lock pin the requested direct versions; uv's Linux
x86-64 required environment and Linux wheels are checked. Lock resolution installs nothing
and does not establish joint runtime compatibility. Imports remain lazy for local fixtures.

## Processor, labels and decoding

The only accepted repository is `Qwen/Qwen3-VL-4B-Instruct` at
`ebb281ec70b05090aa6165b016eac8ec08e71b17`, for model and processor. Base and processor
file inventories are separate canonical manifests, sorted by filename and containing byte
counts/hashes, schema 1, repository and revision. Exact entries are embedded from the
[public staged provenance](../experiments/assets/read2016-qwen3vl4b.json). No weights are
vendored. Every consumed asset is hashed; imports use offline/local-only mode and no remote code.

Processor construction explicitly combines `Qwen2VLImageProcessor`'s torchvision backend,
fast tokenizer, required video processor component and pinned chat template. Video inputs
are never supplied. CPU bicubic/antialiased preprocessing uses patch16, temporal2, merge2,
rescale1/255 and mean/std0.5. For original H,W the maximum pixel **area** is
`ceil(1024² * min(H,W)/max(H,W))`, minimum area65536. The pinned 32-grid resize arithmetic is
checked against the actual image grid and expanded image-token count. Infeasible geometry
fails. There is no crop, tiling, target-guided sizing or external image resize.

The four-line prompt is frozen verbatim in `PROMPT`. Selected targets preserve ordered literal
source strings, whitespace and empty lines; boxes map original endpoints to 0..1000 without
rounding. Compact target JSON uses only `regions`, `text`, `bbox` and escapes every `<` as the
reversible JSON escape `\u003c`. No source line ID, illegibility guess or annotation metadata
enters the target. A zero-region target still supervises its JSON and EOS.

User-only and user+assistant templates are independently processed. Actual expanded prompt
IDs must prefix the full sequence exactly, followed by exactly target IDs, EOS151645 and
one template newline. Only that newline is removed. Prompt/image/padding labels are -100;
JSON+EOS labels equal the input IDs. The model performs the causal shift. All selected pages
are preflighted before any optimizer step; overflow aborts, never truncates or drops a page.
Limits are P2048, T4096 including EOS, P+T6144; targets exceeding the generation budget are
recorded explicitly. These limits have not established coverage or training adequacy.

A new `GenerationConfig` supplies greedy one-beam decoding, max2048 new tokens, EOS151645,
PAD151643, repetition penalty1 and no forced tokens/stop strings. The staged publisher's
sampling defaults are not inherited. Generation keeps image/grid and multimodal token-type
inputs, uses eval/inference mode, cache and native Flash SDPA. It slices after the actual input
length, retains raw IDs/text and removes only a verified terminal EOS for strict parsing.
At the cap without EOS, even valid-looking JSON is `TRUNCATED`. Duplicate/extra keys, wrappers,
trailing text, special tokens, nonfinite/string/bool coordinates, malformed Unicode or invalid
boxes yield `INVALID_OUTPUT` with empty regions and raw evidence. A recursion-depth failure
from parsing generated JSON has the same page-failure treatment; recursion errors from the
tokenizer or other runtime components still propagate as operation errors. Empty valid regions/text
remain valid; there is no word-based refusal detector. Scores are always absent.

Normalized endpoints are mapped individually into original pixels, then extents are computed
by subtracting those endpoints. This avoids the right-edge cancellation demonstrated in the
regression tests without relaxing the Pipeline's strict original-image bounds or rounding
coordinates. Output IDs are ordered `line-0001`, etc., with `illegible=False`.

## Fit, checkpoints and evidence

Each fit releases prior model state, resets Python/NumPy/Torch/CUDA RNGs to the supplied seed,
loads a fresh verified BF16 base on cuda:0 with native SDPA, and injects a new LoRA adapter and
optimizer. There is no CPU/offload/quantized fallback or warm-start. Exact72 language q/v names
and q(4096,2560)/v(1024,2560) shapes are checked before PEFT. Rank16/alpha32/dropout0/default
adapter yields exactly144 trainable FP32 A/B tensors and5,898,240parameters; all else is frozen.

Three epochs start from sorted selected IDs and shuffle with `random.Random(seed+epoch)`.
Batch size1/accumulation4 uses page-mean supervised-token loss divided by the actual group
size, including the final partial group. AdamW uses lr1e-4, betas(.9,.999), eps1e-8,
weight_decay0, foreach/fusedFalse, norm clipping1, no schedule and no GradScaler. Non-reentrant
checkpointing keeps embeddings frozen. Forward and checkpoint-recomputed backward remain
inside the native Flash-only SDPA/BF16 context; unsupported kernels fail. Updates are
3*ceil(N/4), including3/9/15 at N2/10/20. Losses/gradients must be finite, some adapter values
must change, and chunked before/after hashes of every frozen parameter must agree.

Checkpoints live in `checkpoints/<manifest SHA256>/` and have IDs
`checkpoint:sha256:<manifest SHA256>`. Schema1 manifests bind recipe/prompt/limits, base and
processor inventory, code/packages/build expectations, and kind. Adapter manifests additionally
bind run/round, ordered selected image IDs/hashes, selected-target digest, seed, epoch order,
actual token/update counts and adapter file inventory. No target strings, machine paths,
optimizer state, provider attempts, timing or costs enter the checkpoint manifest.

Only `adapter_config.json` and `adapter_model.safetensors` are copied from PEFT staging.
The base reference is normalized to the pinned repository/revision. Exact configuration,
keys, shapes, FP32 dtype and finite values are checked before base loading, and loaded adapter
values are compared with the verified file before forward. The verified adapter file inventory
and tensor hashes are retained as one operation's identity, including through image preparation.
Loading uses those retained hashes rather than accepting the current file as a new expectation.
The requested manifest/inventory is rechecked immediately before and after loading, before each
generation, and after final runtime/image/telemetry checks before receipt publication. A transient
replacement loaded into memory is rejected against the retained tensor hashes even if original
file bytes are restored. Fit likewise retains its staged adapter inventory through reload and
checkpoint publication. These are synchronous integrity checks, not a filesystem write lock;
provider read-only inputs and single-dispatcher output ownership remain transport requirements.
Fresh reload always receives an
explicit verified base; there is no AutoPeft/Hub lookup or base substitution. Directory creation
is exclusive, files are fsynced, and the manifest is written last. Existing content is reusable
only when every byte and the complete inventory match; incomplete directories remain errors.
The enclosing Volume/journal transaction semantics are candidate 3, not promised here.

Fit makes a greedy engineering probe on the lexicographically first selected TRAIN page,
saves and validates the adapter, drops the model, reloads fresh base+adapter and compares exact
adapter hashes, generated IDs, status/regions and raw FP32 full-vocabulary next-token logits
for the first min(8,length) positions. Tolerances are fixed at rtol1e-3/atol1e-2; failure cannot
widen them or publish a completed checkpoint. This production check uses fresh models **in the
same process** and labels its receipt accordingly. The separate optional CPU test repeats the
probe in a fresh interpreter; the independent fresh-process4B/CUDA check remains a smoke gate.

Prediction raw evidence and fit probe/telemetry receipts are content-addressed separately under
`receipts/`. Telemetry records observed GPU, CUDA, driver when available, elapsed time and
memory; peak allocation is the process high-water mark, not billed cost. No provider billing is
invented. Runtime receipts and learned adapters stay outside Git. Any engineering probe exposure
is not unbiased validation. No within-fit optimizer/RNG resume is implemented.

## Verification and remaining gates

From a **clean candidate commit**, with the project's existing local dev environment:

```sh
python -m pytest tests/test_qwen.py
python -m pytest
ruff check --no-cache .
git diff --check
uv lock --check
```

Set `QWEN_HEADER_DIR` to the existing staged model directory to run the read-only header test.
It reads only the safetensors header bytes, verifies all72 shapes and the adapter count, and
never loads tensors. Pure tests cover strict JSON/geometry, serialization/loss-span arithmetic,
paths/images, identity failures, package/code mutation, checkpoint bytes/ownership, purposes,
partial-group orders and counts, and fixture-import independence. Checkpoint-mutation tests
cover preparation, loading, generation and final observation; loader spies also exercise a
replacement restored on disk after load. The gated actual adapter test writes a different finite,
correctly shaped safetensors payload and requires inventory rejection before any CUDA load. The worker identity success
case uses an explicitly synthetic file/package/platform fixture; it is not a Linux runtime pass.
The full existing suite deliberately rejects a dirty Git identity, so dirty-tree failures are
not evidence against the clean candidate and must be rerun after publication.

After the reviewed Linux build is released, install from the frozen GPU/dev lock and provide
**only the small pinned processor files**, not dataset files or4Bweights:

```sh
QWEN_PROCESSOR_DIR=/reviewed/processor-assets \
  python -m pytest tests/test_qwen.py -k 'test_actual_ and not staged_headers' -rs
```

The small code allowlist for those synthetic tests is `active_ocr/__init__.py`,
`active_ocr/integrations/__init__.py`, `active_ocr/models.py`,
`active_ocr/integrations/qwen.py`, and `tests/test_qwen.py`. Paths/import layout must preserve
`active_ocr`; the ordinary regression suite needs the full source separately. The optional tests
verify actual pinned grids/tokenization/masks, hand-counted causal CE, random two-layer FP32
Qwen+PEFT forward/backward, frozen weights, seeded reset, partial-group Adam reference, safe
adapter tensor/config failures and an independent subprocess reload. Test-only rank2 dimensions,
math SDPA and a two-token forced termination probe are explicit test reductions, never runtime
fallbacks. Production decode defaults are asserted separately. The tiny test's input is a
synthetic64x64RGB image; the recipe's minimum area expands it to256x256.

Absent assets/dependencies mean **UNVERIFIED skips**, never PASS. Even a successful CPU run
would not prove loaded4B trainability, CUDA/BF16 Flash kernels, memory headroom, source token
coverage, reload tolerance on GPU, cost, OCR quality or active-learning benefit. Those require
the separately reviewed and authorized smoke. No ML installation, build,4Bload, dataset/model
experiment or cloud call was performed for this offline candidate.

Pinned API sources inspected during implementation:
[processor](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/models/qwen3_vl/processing_qwen3_vl.py),
[torchvision image processor](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py),
[causal loss](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/loss/loss_utils.py),
[Qwen configuration](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/models/qwen3_vl/configuration_qwen3_vl.py),
[PEFT config serialization](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/config.py),
[PEFT save/load](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/peft_model.py).

## Remote fresh-process diagnostic

`QwenModel(..., deadline_unix_seconds=None)` preserves the local default and optionally checks an
absolute wall-clock deadline before model loads, training pages and prediction/probe generation.
The remote parent must also terminate a child blocked in a long CUDA operation at that deadline.

`fit(..., external_reload=False)` retains the original default. With `True`, it performs one fit
and the before probe, saves its verified adapter/checkpoint and full probe evidence, then returns
without a same-process after probe. That checkpoint is a pending operation artifact. The parent
exits the training interpreter and launches a fresh interpreter for `verify_reload`, which takes
only the owned checkpoint, first selected TRAIN page metadata, and hashed before-evidence paths.
It runs one after probe and compares the exact generated IDs/status/regions/tensor hashes and
full-vocabulary logits using the fixed **before-generated** prefix. No labels are passed to the
second interpreter. One fit/two probes are preserved; no third probe is implied.

Evidence is immutable `probes/<checkpoint-sha>/before.json`, `before.f32le`, `after.json` and
`after.f32le` under the Qwen output root. Binary logits are finite little-endian contiguous FP32,
`[min(8,generated_length),151936]`, with exact byte/hash references. The nested `logits.key` is
relative to this Qwen root. The coordinator's outer references add `qwen/` relative to the run root;
neither component infers host paths or adds that prefix twice. `verify_reload` returns the maximum
absolute difference and actual training/reload PIDs only after successful verification. Comparison
uses the existing `torch.allclose(before, after, rtol=1e-3, atol=1e-2)` convention. Failed diagnostics
retain partial evidence and cannot produce a successful transport completion or round commit.
The local coordinator validates this evidence with standard-library binary readers, without ML
packages. See [Modal recovery](modal-runtime.md) for the completion and execution gates.

## Pinned processor normalization representation

Transformers 5.16.1 initializes the Torchvision backend through
`BaseImageProcessor._set_attributes` and `_standardize_kwargs`; the latter converts
`image_mean` and `image_std` JSON lists to tuples. The pinned assets specify three `0.5`
values for each. The loader checks those canonical tuples, preserving normalization values,
bicubic resampling and every other processor check. A mismatch reports the failing setting
name without emitting input text or images. The former list comparison rejected valid setup
before all eleven intended CPU checks. The offline regression reproduces this boundary with
ML dependency stubs; successful actual processor/ML execution still requires the reviewed CPU
build. Source inspection used the exact lock-pinned Transformers wheel, SHA256
`2f2d5b98a5ad3718713653734298fa620754ed683702a635ebb587df3ed29c7e`.
