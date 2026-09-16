# Modal bootstrap and dispatcher

`active_ocr.entrypoints.modal_app` implements the Platform half of the
[shared runtime contract](modal-runtime.md). Importing it performs no provider or
model operation. The shared coordinator/CLI, dependency lock, Qwen amendment and
lazy integration imports must be combined and independently reviewed before use.
Author tests use explicit synthetic SDK/process/evidence fakes; their pass counts,
provider IDs and checkpoints are not cloud or model results.

## Operator sequence and evidence

The following is a recipe for a **separately released execution**, not permission
to execute while reviewing code. Retain the approved first-smoke USD5 gross and
USD30 total ceilings, [cost assumptions](modal-preflight.md), exact code/review
commits, source/model hashes and engineering-only unknown-group policy. Stop on
any missing, failed or inconsistent evidence. Do not add retries or increase the
workload to diagnose a failed first smoke.

1. Use the clean reviewed combined checkout and its existing local run store.
   Export the requirements outside the tracked tree; preserve hashes. `uv` must
   already be available on the operator host. No project installation or unlocked
   resolution is allowed:

   ```sh
   uv export --offline --frozen --format requirements-txt \
     --no-emit-project --no-header --no-annotate \
     --extra dev --extra gpu --extra modal \
     --output-file /approved/staging/requirements-linux.txt
   ```

   Construct a `BuildSpec` from the actual `git rev-parse HEAD`, sorted
   `RUNTIME_CODE_FILES` under `src/`, `uv.lock`, the exported requirements and
   `tests/test_qwen.py`, using `qwen.inventory`/`file_hash` and the pinned
   `asset_manifest(PROCESSOR_FILES)` digest. Save canonical JSON with exclusive
   creation. The seven files include the real entrypoint and shared/Qwen code;
   do not hash placeholders or an earlier partial candidate. The prepared image
   helper independently checks clean tracked state, HEAD, every file and the
   offline frozen export's noncomment lines before constructing an image.
2. Under the explicit resource release, resolve/create the two reviewed Volume
   names in the selected environment; hydrate and record their actual distinct
   IDs. Select the named App for the build and eventual deployment. Do not infer
   provider IDs from names or allocate another experiment UUID. The input bundle
   contains only individually verified pinned model files and content-addressed
   page images. The build contains only small processor assets and synthetic
   tests, never the 4B weight shards or real images/labels.
3. Call the explicit eager helper with the hydrated output Volume and initialized
   App. This call builds an image and must not be used as an offline preview:

   ```python
   from pathlib import Path
   from active_ocr.entrypoints.modal_app import build_image

   image_id, receipt, report = build_image(
       spec,
       app=observed_app,
       checkout=Path("/reviewed/checkout"),
       requirements=Path("/approved/staging/requirements-linux.txt"),
       processor_root=Path("/verified/small-processor-assets"),
       output_volume=observed_output_volume,
   )
   ```

   The helper uses Debian/Python3.11, hash-required requirements and individual
   `copy=True` file additions. The CPU `run_function` has GPU=None, cpu2,
   memory32768MiB, timeout900s and `include_source=False`. Its only Volume mount
   is the output Volume `/builds/<build-spec-sha>` at `/build-evidence`.
   The SDK CPU/memory settings are soft requests; the estimate is not a bill cap.
4. The CPU gate runs exactly `tests/test_qwen.py -k 'test_actual_ and not staged_headers'`
   with the pinned small processor directory, disabled plugin auto-discovery,
   suppressed raw stdout/stderr and a temporary pytest directory. Require eleven
   distinct passes, zero skips/failures and a successful process exit. A subprocess
   watchdog terminates/reaps its process group after850s; the provider build step
   has its separate900s bound. No model weights are loaded by these intended
   synthetic tests. Temporary test checkpoints and JUnit details are cleaned up.

   On failure, preserve a bounded `failures/<uuid>.json` diagnostic in build
   evidence and a safe fallback log: actual return code/reported test names and
   statuses, short redacted failure messages. No raw environment or full transcript
   is exported. Missing JUnit/return code remains unknown. A failure cannot produce
   a successful `BuildReceipt`. Persistence failure is not provider termination.
5. A successful step writes hashed `cpu-report.json` and canonical
   `build-receipt.json`, commits the output Volume, and stores identical receipt
   bytes in `/opt/ocr/build-receipt.json` in the image. Environment packages and
   Python patch are observed there, with Linux/x86-64 and direct pin checks.
   `build_image` reads and rehashes the durable evidence after `Image.build(app)`
   returns and only then returns the observed `image.object_id`. Missing cached
   evidence blocks use; there is no forced rebuild fallback. Retain the observed
   image ID separately from the receipt to avoid a circular hash. Obtain
   independent CPU acceptance before the GPU deployment/run stages.
6. Development's coordinator freezes the receipt's code/package fingerprints,
   image/build identity, recipe and declared deployment reference, then creates
   the local run through the existing Pipeline. Use that actual UUID to construct
   `RuntimeSettings`. Save the canonical settings bytes outside the tracked
   checkout. The settings contain no fit targets or oracle paths.
7. Construct the single Function, then deploy only after checking the named App's
   current provider state and the operator's saved deployment record:

   ```python
   from active_ocr.entrypoints.modal_app import create_app

   app = create_app(settings, settings_file=canonical_settings_path)
   # Only after the independent gate, collision check and explicit release:
   app.deploy(environment_name=settings.environment_name)
   ```

   `create_app` constructs; it does not deploy or decide replacement safety.
   Refuse an existing deployment whose recorded settings differ. An existing App
   used only as build context is not proof of a deployed matching Function. If
   the state/record is unavailable, stop for explicit provider reconciliation;
   do not replace the deployment speculatively. These operator checks are part
   of the execution gate, not a claimed SDK universal comparison API.

   The module-level `dispatch` imports the exact copied code. The one canonical
   settings file is attached at `/opt/ocr/runtime-settings.json` with
   `add_local_file(copy=False)`. In SDK1.5.5 this is a deferred file mount preserving
   the base image ID, not a baked image change. Keep the file unchanged and present
   until deployment resolves the mount. Check the actual deployed image, Function/
   App IDs, Volume IDs/subpaths/options and settings digest, recording observed
   evidence; a successful constructor or a declared reference proves none of these.
   The coordinator consumes a separate observed deployment record, not invented IDs.

The eager image/build-step and module-level Function construction were exercised
**offline** against installed SDK1.5.5 with synthetic handles/files; no hydration,
build, deployment, upload or model operation was called. Relevant official APIs:
[Image](https://modal.com/docs/sdk/py/latest/Image),
[Volume](https://modal.com/docs/sdk/py/latest/Volume),
[App](https://modal.com/docs/sdk/py/latest/App).

## Runtime behavior

One synchronous dispatcher requests L40S only, CPU request/limit2 and memory
request/limit32768MiB; max1/min0/buffer0 containers, scaledown2s, retries0,
execution600s and startup300s. Input Volume `/bundles/<bundle-sha>` is mounted
read-only at `/inputs`; the distinct output Volume `/runs/<actual-run-uuid>` is
writable at `/outputs`. Only the exact input inventory is admitted; symlinks,
changed bytes, extra oracle/GT files and invalid page headers reject. Code/build
receipt and actual packages are rechecked before model work. The worker manifest
is constructed from observed build evidence and copied into operation artifacts.

`run-control.json` atomically binds the settings digest and original submission/
deadline values. Subsequent calls cannot extend or shift the window. The worker
checks the current clock and the coordinator owns the persisted40minute first-
submission deadline, aggregate accounting and cancellation. Within each invocation,
the parent gives children the earlier of that deadline and590s from dispatcher
entry, leaving a termination margin before the600s provider limit. This cap is
shared by both fit stages. Startup, queue time, crashes and billing lag retain the
limits and caveats in the preflight; timeout/retries0 are not a dollar cap.

The parent imports no Torch and constructs no model. Each operation runs in a
fresh Python subprocess. A fit uses exactly one `external_reload=True` fit/before
probe, waits for that interpreter to exit, then runs exactly one `verify_reload`
after probe in another interpreter in the same GPU invocation. Selected labels
travel only through the first child's stdin pipe. The second child receives only
selected TRAIN page metadata, checkpoint identity and hashed before references.
Raw child stdout/stderr are suppressed; its separate result pipe emits derived
results only. Failure/timeout kills the child process group and reaps the child;
uncertain provider termination is still a coordinator/operator reconciliation task.

Before completing a fit, verify checkpoint ownership/files, selected target digest,
actual distinct process IDs, all144 adapter tensor hashes against FP32 safetensor
bytes, generated IDs/status/regions and finite full-vocabulary logits. The fixed
comparison is `abs(after-before) <= .01 + .001*abs(after)`. The nested logit key
is Qwen-root-relative `probes/<checkpoint-sha>/...`; outer references add `qwen/`
exactly once. No top-k reduction, extra paid probe or same-process substitute.
Prediction completion verifies ordered ownership, original-image geometry, raw
receipt content addresses and consistency with the structured output.

A fresh execution UUID owns an attempt directory. The started worker record is
committed before model work. Canonical `complete.json` is published atomically
last and then committed. A matching existing completion is reused only after all
referenced bytes and worker identity pass again. A different attempt, partial or
failed execution, multiple completions or corrupted evidence requires explicit
reconciliation; none triggers automatic retraining. Bounded failure receipts
contain no request/targets/traceback or terminal/cost assertion. Partial artifacts
may be persisted for diagnosis but are never accepted as completed model operations.
The system does not promise exactly-once paid execution.

## Local verification

Use the checkout's existing development environment, without GPU installation:

```sh
python -m pytest tests/test_modal_app.py tests/test_modal_model.py tests/test_qwen.py
ruff check src/active_ocr/entrypoints/modal_app.py tests/test_modal_app.py
python -m pytest
```

Absent actual-ML dependencies/assets are unverified skips, not passing CPU/model
checks. Combined independent review, zero-skip Linux CPU evidence and the bounded
real4B/GPU smoke remain separate gates. Preserve failures and stop at the first
smoke failure; no authentication, budget, source or methodology approval is inferred
from these tests or this guide.
