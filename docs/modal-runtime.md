# Modal runtime and recovery

The candidate implementation connects the existing Pipeline/SQLite to one synchronous Modal
adapter. Its wire API was frozen before the disjoint worker implementation. Development owns
`integrations/modal_model.py`, local coordination/CLI/config/dependencies and Qwen; Platform owns
`entrypoints/modal_app.py`, its tests and deployment instructions. Incompatible shared API changes
require an acknowledged review. Imports remain free of Modal/ML execution.

This is offline implementation evidence, not permission to build or execute. Actual Linux CPU
checks, deployment identities/mounts, 4B CUDA training/probes, provider recovery and costs remain
separate review/execution gates. See [deployment](modal-deployment.md) for the bootstrap helpers.

## Shared call

One deployed synchronous Function named `dispatch` takes a JSON-compatible dictionary and
returns a JSON-compatible dictionary:

```python
# Platform boundary; one operation per synchronous dispatcher call.
invocation = Invocation.model_validate(payload)
settings.check_invocation(invocation)
# Check immutable inputs, deadline/control, completion reuse; dispatch one operation.
return DispatchResponse(result=result, completion=completion).model_dump(mode="json")
```

The discriminated `request.operation` is `load_base`, `fit` or `predict`. `RunContext` carries
the existing run UUID, explicit `source_policy`, input bundle digest and full frozen
`RealOCRConfig`. Base has round0/baseline purpose and no checkpoint/pages/labels. Fit has a
positive round, base input checkpoint, uint32 seed and cumulative selected TRAIN examples.
The worker must inspect the input checkpoint's manifest and require **base kind**, then call
reset-fit; a previous round adapter cannot become a warm start. Predict has exact input
checkpoint, ordered image metadata, purpose and round. Baseline requires round0/validation;
positive validation and pool use their respective splits and owned adapter checkpoint.
Empty prediction lists are permitted but retain identity/purpose validation.

Only `RemoteExample.regions` carries labels. Fit serializes these transiently for RPC, never
into journals, printed exceptions, provider logs, output receipts or a cached trainer dataset.
Do not log `model_dump()` or the incoming dictionary. `repr` hides examples, but this is not
a substitute for the explicit no-payload-logging rule. Standard source regions preserve literal
text/order; finite original-pixel bounds and unique line/page IDs are validated.

`RemotePage` is only ID, required nullable document ID, split, original dimensions, image hash
and `image_key="images/<image SHA256>"`. It rejects source paths, region payloads, provenance
and GT fields. The worker reconstructs the existing `SimulationPage` with:

- `image_uri = "/inputs/" + image_key`, after containment/symlink/byte/header checks;
- `source_policy = invocation.request.context.source_policy` (mandatory, never inferred);
- `source_image = image_key`, a nonlabel compatibility placeholder never consumed by Qwen;
- the exact remote ID/document/split/dimensions/hash.

This explicitly admits nullable READ IDs only under the frozen engineering policy. It carries
no source XML, oracle manifest or provenance. Prediction receives no examples. Coordinator
construction must compare remote page metadata against its frozen snapshot before dispatch;
the worker checks membership/hash against the approved bundle and Qwen checks image bytes.
The boundary is protection against accidental leakage, not a malicious authorized coordinator.

`operation_id(request)` hashes schema1 and `request.semantic_record()` as canonical UTF-8 JSON
(sorted keys, compact separators, no NaN). This binds operation, run, recipe/expected identities,
source policy, bundle, round, purpose, input checkpoint, ordered full page metadata and fit seed.
Fit replaces examples with ordered pages plus a digest of ordered page IDs/source regions;
no raw text appears in the semantic record. Attempts, deadlines, provider IDs and telemetry
are excluded from this hash and never seed training. Base input identity is in the pinned
recipe manifests, because its checkpoint has not yet been materialized.

## Bootstrap without a circular run or provider ID

The following is the proposed later operator sequence. It must be implemented, reviewed and
separately released before any command makes a provider call. No new run-ID API is needed.

1. Freeze the complete reviewed source commit in a clean execution checkout, local dependency
   fingerprint, input image/model allowlist and `BuildSpec`. Select unique reviewed deployment
   and two Volume **names** now. Names are declarations, not invented provider object IDs.
   `BuildSpec` hashes the exact seven runtime Python files, frozen `uv.lock`, the exported
   `requirements-linux.txt`, one synthetic test file and pinned small-processor manifest.
   Export requirements from the lock with project installation excluded and dev/gpu/modal extras;
   verify the export/lock relationship before accepting its recorded bytes. The later dependency
   change adds `modal==1.5.5` as its own optional `modal` extra; laptop installs never imply `gpu`.
2. Under an explicit execution release, create/resolve the two approved Volumes, verify their
   actual hydrated IDs differ, and persist observed IDs. Use one named App as the build context
   and eventual deployment. Eager `Image.build(app)` can produce the image before the GPU
   Function or local run exists. This is an SDK build primitive, not a second worker/service.
3. Build Linux Python3.11 from the frozen requirements with hashes. Use individual
   `add_local_file(..., copy=True)` calls for the code/test/small-processor allowlists; no project
   install, `add_local_dir` of the repo, or implicit source collection. `run_function` receives
   the canonical build-spec digest explicitly (its cache does not track every transitive helper).
   Use `include_source=False` because the reviewed code is already copied. The CPU build step
   has no GPU, cpu2, memory32768MiB and timeout900s. This is a soft-resource/build allowance,
   not a provider dollar cap. No4Bweights or real dataset images/labels enter the build.
4. That step runs `tests/test_qwen.py -k 'actual and not staged_headers'` with the pinned small
   processor assets. Require the exact11selected tests,11passes,0skips,0failures; a pytest exit0
   with skips fails the gate. Export a hashed report and `BuildReceipt` through **only** output
   Volume `/builds/<build-spec-sha>` mounted at `/build-evidence`. Commit it and retain an
   identical build receipt inside the image under `/opt/ocr/build-receipt.json`. The CPU report
   has its own hash; the receipt does not contain its own hash or an unknown future image ID.
   There is no input Volume mount in this step. Failure stops; no automatic rebuild or GPU call.
5. Once `image.build(app)` returns, record the **observed** `image.object_id`, fetch/read/rehash
   the CPU receipt/report, compare source/files/build spec and packages, and obtain independent
   CPU acceptance. Keep the image ID as a separate coordinator observation. A cached build is
   usable only when its immutable receipt/report are still available and match; missing evidence
   is a blocker, not permission to rebuild or synthesize a receipt. Package inventory is measured
   inside this image, not guessed from the laptop. Base-image resolution is retained through the
   observed immutable image ID; the minor-version base selector itself is not a patch pin.
6. Compute `code_bundle_sha256` and `remote_dependency_sha256` using `BuildReceipt`'s properties,
   which match the Qwen worker manifest algorithms. Form the declared deployment reference
   `modal:<environment>/<deployment-name>/dispatch@<observed-image-id>`. It names the intended
   immutable deployment; it does **not** claim an already deployed Function or provider ID.
   Freeze these values, model/processor manifests, local code/dependencies and build-spec digest
   into existing `ExpectedIdentity`/`RealOCRConfig` and the simulation config. Call the existing
   `Pipeline.create_simulation`; it generates and atomically stores the actual UUID. It starts
   no model operation. There is no build/package/run-ID dependency cycle.
7. With that UUID, create `RuntimeSettings` and deploy the single `dispatch` Function using
   `Image.from_id(observed_image_id)`, the selected deployment name and the per-run mounts below.
   Refuse to replace an existing deployment with different settings. Capture actual Function/App
   IDs after deployment; do not fabricate them in preflight. Verify observed name/image/mount
   configuration against settings before the first call. Deployment settings are explicit
   nonlabel data, not source-code edits that would invalidate the frozen code commit.
8. Persist first-submission time and absolute deadline before the first GPU spawn, together
   with the settings digest. Every invocation/restart carries those same values; they must be
   echoed in an immutable `/outputs/run-control.json`. Worker compares them on every call and
   refuses extension. `Invocation` permits at most2400seconds between the two values. This is
   the run-wide envelope, not2400seconds per operation. Validate remaining cost/storage/build
   allowance independently; the user budget is not encoded as a fabricated bill.

The SDK's eager [Image build](https://modal.com/docs/sdk/py/latest/Image#build),
[CPU build functions](https://modal.com/docs/guide/images#run-a-python-function-during-your-build-with-run_function)
and [Volume mounting](https://modal.com/docs/sdk/py/latest/Volume#with_mount_options) support
this proposed sequence. Bundled Modal1.5.5 reference signatures were inspected offline; the
image guide was also read from the official site. No provider/build behavior was exercised.
Platform must verify exact construction against SDK1.5.5 in its scoped implementation tests.

## Files, mounts and completed results

`InputBundle` is a canonical sorted inventory containing exactly the approved model/processor
files under `model/` plus content-addressed image bytes under `images/`. No dataset source
manifest, XML, archive, DB, labels, tests, runtime code or local memory is allowed there.
Copy/upload individual verified files only; duplicates can reuse identical content addresses.
The inventory travels in deployment settings; it is not an oracle manifest on the input Volume.

| Path | Binding | Access |
| --- | --- | --- |
| `/inputs` | input Volume `/bundles/<bundle-sha256>` | provider read-only |
| `/outputs` | distinct output Volume `/runs/<run-uuid>` | writable single dispatcher |
| `/opt/ocr` | explicit runtime code allowlist and build receipt in the immutable image | verified code |

Qwen uses `input_root=/inputs`, `output_root=/outputs/qwen`. This shared run-level Qwen output
root is necessary so later operations can load previously completed checkpoint IDs without
copying, symlinking or changing Qwen's path contract. Its checkpoints and raw receipts are
content addressed and immutable; an incomplete checkpoint remains an error. Attempt receipts
reference these artifacts rather than duplicating weights. Do not mount the entire output
Volume or a different run. Missing remote subdirectories are not evidence of a valid upload.

Construct `/opt/ocr/worker.json` from the independently obtained build receipt and deployed
settings: code files/source, actual package inventory, build-spec digest and deployment
reference. Verify packages again in the running worker. The manifest parent is the code root,
so Qwen validates its actually executing package files; never construct evidence merely by
copying an incoming request's expectations. Copy the verified manifest to the attempt evidence
inventory for local verification. GPU/device/driver/cost observations stay separate telemetry.

Each submitted `Invocation` has a client-generated attempt UUID. Each actual worker execution
gets a new execution UUID so infrastructure rescheduling cannot overwrite earlier output.
Observed `provider_call_id` is obtained from the provider context, not the operation hash.
Attempt prefix is:
`operations/<operation-id>/attempts/<attempt-id>/executions/<execution-id>/`.
The worker writes canonical `OperationResult` bytes as `complete.json` **last**, commits the
Volume, then returns `DispatchResponse(result, completion-ref)`. The completion hash covers
exact result bytes without including itself. Each `ArtifactRef` is relative to the assigned
run output subtree, with bytes and SHA256. The result inventories its checkpoint manifest,
raw prediction evidence, observed worker manifest and any probe artifacts. Prediction evidence
paths in the wire `Prediction` are relative keys; the coordinator resolves only verified keys.

Before local completion, re-read completion/artifacts through `Volume.read_file`, verify hashes,
canonical JSON, operation/attempt/run/round/purpose/page order, checkpoint owner/recipe/base and
its complete referenced inventory, all raw references and actual worker code/package identities.
Validate original-image geometry with the existing strict Pipeline rules. No result/status is
trusted merely because SDK deserialization succeeded. `check_request` and schema checks are
necessary structural checks, not substitutes for reading bytes and validating manifest closure.
For fits also verify the fresh-process probe evidence below before marking completion.

Failures may write `FailureReceipt` at the same prefix's `failure.json`, with only a bounded
error code and actual ownership IDs. It contains no request, targets or raw traceback. Preserve
partial artifacts; absence of `complete.json` is never success. A failure receipt does not
establish provider terminal/cancellation state. No ledger or cost value is invented by the worker.

## Reservation, waiting, recovery and cancellation

Use existing SQLite `model-operation` records and `compare_and_swap`; no additional database
or scheduler. The future transport persists only the semantic record/digest and attempt state.
Fit payloads are rebuilt from already revealed local TRAIN examples when genuinely needed.

| Observed state | Allowed next action |
| --- | --- |
| No operation record | CAS a reservation with unique owner token; losers reload and do not submit |
| Reserved, not submitted | Persist attempt ID, run deadline and SUBMITTING state before `spawn` |
| Spawn returns a call ID | Persist ID before waiting; use `FunctionCall.from_id(id)` on restart |
| Spawn acknowledgment lost / crash while SUBMITTING | UNKNOWN; stop and reconcile, never blind resubmit |
| Known call pending | Reattach and wait within remaining envelope; client wait timeout is not remote cancellation |
| Completed result/receipt verified | CAS COMPLETED and reuse without another fit, even after later evaluation/round-commit failure |
| Worker exception / partial output | Retain failure and IDs; confirm provider termination and reconcile before any separately approved retry |
| Deadline or operator cancellation | Persist CANCEL_REQUESTED, cancel known call with `terminate_containers=True`, then confirm terminal state |
| Cancellation uncertain / provider identity absent | UNKNOWN; no retry, keep spend unresolved for explicit reconciliation |

Worker startup first searches completed receipt(s) for this semantic operation/attempt and
verifies bytes before reuse. A completion from a different attempt is a reconciliation conflict,
not permission to rewrite its producer IDs. Duplicate/rescheduled execution may still incur
cost; neither CAS nor Volume publication promises exactly-once paid execution. A verified
completed fit checkpoint survives a failed validation or round commit. If the round commit
succeeded but the response was lost, reload the existing round rather than applying it twice.

Cancellation confirmation requires a provider terminal result/cancellation observation plus
bounded container/account reconciliation; the best-effort call graph or `cancel()` acknowledgment
alone is insufficient. Do not claim a nonexistent SDK universal terminal-status API. A still
pending `get(timeout=0)` leaves the attempt unresolved. Preserve explicit operator/provider
reconciliation evidence when an ID cannot be recovered. Subsequent retries require terminal
confirmation, budget checks and Manager release; first smoke stops at the first failure.
[FunctionCall wait/cancel](https://modal.com/docs/sdk/py/latest/FunctionCall) documents the
client wait and cancellation methods; their operational success remains an execution gate.

## Fresh-process reload contract

The approved narrow Qwen amendment preserves the default same-process behavior and adds an
opt-in remote path. One fit and two probes remain the workload; no extra diagnostic RPC exists.

1. `fit(..., external_reload: bool = False) -> str`. In the opt-in remote path, do the
   existing fit once and **one** pre-save greedy probe on the lexicographically first selected
   TRAIN page. Validate/save its adapter, preserve before-probe IDs/status/regions/all144tensor
   hashes and full FP32 logits, and publish its immutable checkpoint as a *pending operation
   artifact*. Skip the current same-process after probe in this mode. It is not a completed
   transport fit until the next step succeeds. Other calls retain the accepted same-process path.
2. The model-specific diagnostic has this signature:
   `verify_reload(page: SimulationPage, *, experiment_id: str, round_number: int, model_id: str,
   before_metadata: Path, before_metadata_sha256: str, before_logits: Path,
   before_logits_sha256: str) -> dict`. Both paths must remain under this Qwen output root;
   the expected hashes come from the parent's verified first-child outputs. It accepts only
   the owned checkpoint and selected TRAIN page metadata, never targets. It verifies identities/bytes, loads
   a fresh base+adapter, performs exactly **one** after greedy probe, compares exact tensor hashes,
   IDs/status/regions and the fixed before-prefix full-vocabulary logits at the same
   rtol1e-3/atol1e-2. It writes after evidence or fails without a completion receipt. Before/after
   metadata carry actual process IDs; no targets are needed by this second stage.
3. The optional absolute `deadline_unix_seconds` into `QwenModel` construction, preserving
   the current no-deadline default for existing local checks. Check it before loading, each
   training page and each prediction/probe generation; expiration is an operation failure.
   The parent also terminates a child that is inside a long-running CUDA call at the deadline.
   The guard changes neither RNG nor recipe; a long CUDA call still needs the parent watchdog.
4. Platform runs these as two sequential fresh Python interpreters **inside the same existing
   GPU Function invocation**, with the parent never loading CUDA/model state. Same GPU/container,
   software/image and backend; no extra Modal Function, service, fit or paid probe. Pass fit input
   through an ephemeral pipe only. Child output is derived evidence, never the selected-label
   payload; suppress unreviewed library logs/raw exceptions from escaping provider logs. Parent
   owns absolute deadline enforcement, child termination and output-Volume commit. Fresh child
   process exit precedes the second child; actual distinct PIDs are recorded conservatively.

Derived files are under `qwen/probes/<checkpoint-sha>/before.{json,f32le}` and
`after.{json,f32le}`. `ProbeMetadata` binds model/page/image/original dimensions/process,
IDs/status/regions/tensor hashes and the binary logit reference. Logits are contiguous finite
FP32 little-endian, shape `[min(8, generated_length),151936]`, no pickle and no top-k reduction.
The after logits use the **before generated prefix** even if independently generated IDs
already mismatch. The coordinator can compare bytes/numbers with stdlib `array('f')`/byte order
handling, without installing Torch locally. Per element require
`abs(after-before) <= .01 + .001*abs(after)` (the existing
`torch.allclose(before, after)` reference convention); retain maximum absolute difference and fail on
nonfinite values/shape drift. `ReloadEvidence` records references to both metadata files,
actual different process IDs, fixed tolerances and exact-match verdict. Structural fields are
not proof by themselves; the coordinator verifies all referenced bytes and values.

This substitutes the already accepted two probes with one before/one fresh-process after;
count stays two, fit stays one. Parent/child startup time is charged inside the same envelope.
Whether it fits the budget and600second attempt limit is still unmeasured. A partial trained
checkpoint after a probe failure stays quarantined/uncommitted; do not call it fit completion,
retry the fit automatically, or reinterpret the later run as an unbiased evaluation.

The RPC has no separate paid diagnostic operation. A successful fit requires verified
`ReloadEvidence`; a published checkpoint alone cannot complete a local model operation.
`verify_reload` returns `max_absolute_difference`, `train_process_id`, `reload_process_id`.
Its nested `ProbeMetadata.logits.key` is strictly Qwen-root-relative (`probes/<sha>/before.f32le`
or `after.f32le`). Outer operation references are run-root-relative (`qwen/probes/...`).
Consumer checks translate this fixed prefix once and reject mismatched/doubled namespaces.

## Local CLI and recovery

Install the coordinator with the optional `modal` extra from the reviewed lock, independently of
`gpu`. Run commands from the clean, exact execution checkout with explicit source imports.
The configuration and runtime files are inputs from the reviewed bootstrap, never guessed IDs:

```sh
PYTHONPATH=src python -m active_ocr.entrypoints.cli real create \
  /prepared/pages.jsonl /runs/smoke /reviewed/simulation-config.json
# Save canonical RuntimeSettings using the returned run UUID and the actual CPU build receipt.
# Save DeploymentObservation {settings_sha256, function_id, app_id} after approved deployment.
PYTHONPATH=src python -m active_ocr.entrypoints.cli real preflight \
  /runs/smoke RUN_UUID /reviewed/runtime-settings.json
PYTHONPATH=src python -m active_ocr.entrypoints.cli real preflight \
  /runs/smoke RUN_UUID /reviewed/runtime-settings.json \
  --deployment /reviewed/deployment.json --provider
PYTHONPATH=src python -m active_ocr.entrypoints.cli real run \
  /runs/smoke RUN_UUID /reviewed/runtime-settings.json /reviewed/deployment.json --one-round
PYTHONPATH=src python -m active_ocr.entrypoints.cli real resume \
  /runs/smoke RUN_UUID /reviewed/runtime-settings.json /reviewed/deployment.json
PYTHONPATH=src python -m active_ocr.entrypoints.cli real status /runs/smoke RUN_UUID
PYTHONPATH=src python -m active_ocr.entrypoints.cli real export /runs/smoke RUN_UUID /new/export
```

Local preflight checks source integrity, exact clean local identity, validation membership and
runtime/run/bundle agreement; it does not import Modal. `--provider` hydrates existing resources
and compares Function and Volume IDs without creating them. The App ID is retained deployment
provenance; the client rechecks the stable Function ID, not an invented SDK universal App-status
API. Deployment/resource/image verification and CPU acceptance precede this CLI's run gate.
`real run/resume` can submit work and therefore requires the separate execution release.

`model-control` in the existing SQLite table freezes settings digest and first-submission wall
clock/deadline. `model-operation` stores semantic requests with fit target digests, submission UUID,
state, observed call ID and verified response. Selected target payloads are never journaled.
A CAS reservation and SUBMITTING transition precede spawn; a known call ID is saved before wait.
Only verified COMPLETED records are reusable. A per-run active-operation field prevents another
operation starting while a previous submission is unresolved. Completed fits remain reusable
through a failed prediction, evaluation or round CAS; successful round-response loss resumes
from the committed snapshot. No extra model call or round is inferred from missing responses.

After submission ambiguity, inspect provider evidence and explicitly attach its observed call:

```sh
PYTHONPATH=src python -m active_ocr.entrypoints.cli real reconcile \
  /runs/smoke RUN_UUID /reviewed/runtime-settings.json /reviewed/deployment.json OP_SHA \
  --call-id OBSERVED_FUNCTION_CALL_ID
```

Reconciliation polls that call once (`timeout=0`) or accepts a saved `DispatchResponse` file with
`--completion`; all ownership, canonical bytes, artifact hashes, checkpoint and probe checks still
apply. A fit request is reconstructed only from its journaled selected page IDs and verified local
oracle, then its semantic digest must match. Producer operation/attempt/call IDs are never rewritten.
`--cancel` persists CANCEL_REQUESTED before requesting `terminate_containers=True`. Neither the
acknowledgement nor an absent response is terminal proof. Deadline expiry follows the same path;
no CLI option extends the deadline or blindly retries. Failed/cancelled/unknown work retains the run
lock until verified completion or a reviewed operator/provider decision. This candidate has no
in-place retry command: a failure stops the first smoke, and any later attempt needs terminal,
budget and release review. Provider/container/account reconciliation remains an operator gate.

`input_upload_files` produces a verified explicit allowlist of frozen images and pinned model
files. `upload_input_bundle` uploads only those paths under a new bundle hash with overwrite
disabled; it is an explicit provider action for a later authorized bootstrap. No directory import,
XML/oracle/ground-truth dataset upload, host path in an RPC or selected-target file is used.
Runtime source and CPU test/processor allowlists remain separate.

JSON exports retain the existing run schema and add a separate `validation.json` membership/count
record; CSV includes `validation_page_ids` and `validation_page_count`. The optional config subset
is nonempty/unique and must contain frozen validation pages at create/resume. Omission retains all
validation pages. The exact ordered subset reaches both prediction and the isolated evaluator.

## Offline verification and limits

Author tests use actual SQLite/CAS/filesystem state and synthetic provider/adapter bytes. They
exercise ambiguous submission, concurrent reservation, known-call reattach, terminal failure,
cancellation ambiguity, absolute deadline preservation, corrupt/foreign receipts, full FP32 probe
comparison, adapter tensor-byte closure, fit reuse after downstream failure, real Pipeline round
commit recovery, image-only upload plans, CLI and validation subsets. Fake transport is not proof
of Modal recovery, mounts, measured spend, GPU behavior or OCR quality. The 11 gated actual-ML
checks remain unverified until the separately reviewed Linux build runs without skips.
