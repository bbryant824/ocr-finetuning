# Modal runtime and smoke preflight

Recommendation checked 2026-09-16 against main
`83c026dc6acb4575553fe234a90350435f835ffd`, its `uv.lock`, Modal SDK 1.5.5,
and [the accepted engineering plan](../plans/PLAN-003-real-ocr-modal.md).
This is a runtime specification for review, not evidence of a successful build,
model load, training run or bill. Development owns dependencies/model/transport;
Platform owns the eventual entrypoint only after explicit file assignment.

## Candidate dependencies and model boundary

Use Linux x86-64, Python 3.11, with these exact package constraints. Existing
local verification uses Python 3.11.14. Modal `Image.debian_slim(python_version="3.11")`
selects a minor line, not an immutable patch/base image: record the resolved
Python patch, Modal image ID/base digest and installed dependency inventory before
execution. Reject unreviewed resolution changes; do not copy a macOS environment.

| Dependency | Candidate pin | Evidence / gap |
| --- | --- | --- |
| torch | 2.14.0 | Existing lock; CPython 3.11 Linux wheel exists |
| transformers | 5.16.1 | Existing lock; built-in Qwen3-VL implementation |
| peft | 0.20.0 | Existing lock; metadata accepts the proposed stack |
| accelerate | 1.14.0 | Existing lock; metadata accepts the proposed stack |
| torchvision | 0.29.0 | **Missing from lock**; package requires torch==2.14.0 |
| modal | 1.5.5 | Inspected local SDK; **missing from project lock** |
| huggingface-hub / tokenizers | 1.30.0 / 0.23.2 | Existing lock |
| safetensors / pillow / numpy | 0.8.0 / 12.3.0 / 2.4.6 | Existing lock; NumPy Python<3.12 branch |

Release metadata: [torch](https://pypi.org/pypi/torch/2.14.0/json),
[transformers](https://pypi.org/pypi/transformers/5.16.1/json),
[PEFT](https://pypi.org/pypi/peft/0.20.0/json),
[Accelerate](https://pypi.org/pypi/accelerate/1.14.0/json),
[torchvision](https://pypi.org/pypi/torchvision/0.29.0/json),
[Modal](https://pypi.org/pypi/modal/1.5.5/json).
Metadata compatibility does not prove joint runtime compatibility. Development
must resolve and commit a reproducible Linux dependency set; the current lock
alone is incomplete for this runtime.

The torch wheel requests CUDA toolkit 13.0.3 and cu13 dependencies. Modal documents
driver 580.95.05 / CUDA Driver API 13.0 and supports pip-installed CUDA dependencies.
This supports the candidate choice; verify actual driver, `torch.version.cuda`,
CUDA availability and BF16 support in the released smoke. Do not install a second
system CUDA toolkit, FlashAttention or a quantization stack for this candidate.
Use PyTorch SDPA; stop on unsupported hardware or OOM without GPU/model/precision
fallback. [Modal CUDA guide](https://modal.com/docs/guide/cuda).

Load only the staged snapshot of `Qwen/Qwen3-VL-4B-Instruct` revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`, with verified bytes,
`local_files_only=True` and `trust_remote_code=False`. Use BF16 on one device.
The Transformers tag maps Qwen3-VL to Qwen2VLImageProcessor; its torchvision
backend imports torchvision directly. Pin that backend explicitly and test it;
do not let installed optional packages silently select different preprocessing.
[Mapping](https://raw.githubusercontent.com/huggingface/transformers/v5.16.1/src/transformers/models/auto/image_processing_auto.py),
[processor](https://raw.githubusercontent.com/huggingface/transformers/v5.16.1/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py).

The model source and staged 36-layer text config support the explicit LoRA target
list `model.language_model.layers.{0..35}.self_attn.{q_proj,v_proj}`: 72 modules.
Enumerate and assert the actual matches and trainable parameters; freeze base
and vision weights. Retain the plan's rank16/alpha32/dropout0, batch1/accumulation4,
three epochs and optimizer settings. Do not rely on a broad suffix regex.
Image-token count comes from `image_grid_thw.prod()/merge_size**2`; preserve actual
grid, prompt/target counts and mask evidence. A 4096 target-text cap is not a total
multimodal sequence bound. Validate the full sequence and fail on overflow rather
than truncate labels. [Model source](https://raw.githubusercontent.com/huggingface/transformers/v5.16.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py),
[processor token expansion](https://raw.githubusercontent.com/huggingface/transformers/v5.16.1/src/transformers/models/qwen3_vl/processing_qwen3_vl.py).

## One dispatcher; Volume constraint needs plan amendment

Recommend one sequential Function with `gpu="L40S"`, `cpu=(2, 2)`,
`memory=(32768, 32768)`, `max_containers=1`, `min_containers=0`,
`buffer_containers=0`, `scaledown_window=2`, `retries=0`, `timeout=600`,
`startup_timeout=300`. No input concurrency decorator, GPU list, region premium,
nonpreemptible premium or enlarged ephemeral disk. Serialize client submissions;
container count alone does not bound cumulative jobs. CPU throttling and RAM OOM
limits constrain this quote; exceeding memory is a failure, not an automatic
resource increase. [Resource semantics](https://modal.com/docs/guide/resources).

**Blocking SDK fact:** 1.5.5 `validate_volumes_by_object_id` rejects the same
Volume ID mounted at multiple paths, irrespective of `sub_path` or read-only
options. Separate handles do not solve this. The plan's one-Volume requirement
cannot implement separate provider-enforced read-only input and writable output
mounts as proposed. This is established by installed SDK source, without launching
anything. Reproduce by inspecting `modal/_utils/mount_utils.py` in the pinned
[Modal 1.5.5 distribution](https://pypi.org/pypi/modal/1.5.5/json): both
`validate_volumes` and the post-hydration `validate_volumes_by_object_id` enforce
the restriction. An offline duplicate-ID probe of the latter raises `InvalidError`;
it uses stand-in records and makes no remote request.

**Recommendation requiring Manager/Planning acceptance:** keep one dispatcher and
use two named Volumes: consolidate images/base files in one read-only input
Volume; use another for writable run outputs. Storage is billed by total bytes,
so the same 12 GiB estimate below applies. Do not silently weaken immutability by
mounting a single shared root writable. No resource is created by this document.

Proposed amended mounts:

| Container path | Volume / subdirectory | Access |
| --- | --- | --- |
| `/inputs` | input Volume, `/bundles/<bundle-sha256>` | read-only |
| `/outputs` | output Volume, `/runs/<run-uuid>` | writable, unique run |

The input bundle contains only `images/` and `model/`; identify their independent
manifest/revision hashes in the run. It contains no source truth.

SDK 1.5.5 supports `with_mount_options(sub_path=..., read_only=True)` and
`Volume.from_name(..., create_if_missing=False, environment_name=...)`. Check
the remote subtree inventory/hashes before starting: absent subdirectories can
be created at container startup, so their existence cannot prove a valid upload.
Do not mount the Volume root. Validate resolved paths against the assigned mount;
write immutable operation-specific checkpoints/receipts, commit completed output
and reload before a separate consumer reads it. Verify actual read-only enforcement
and checkpoint visibility during the released integration check.
[Volume API](https://modal.com/docs/sdk/py/latest/Volume),
[Volume consistency](https://modal.com/docs/guide/volumes).

Upload only approved page images and pinned model/processor files. Build from an
explicit code/dependency allowlist; exclude source XML/oracle, archives, local DB,
labels in tests, `.agent-local`, `.local`, credentials and the repository root.
Fit receives only selected training targets in its request; prediction receives
images/IDs/purpose and no truth. Do not persist targets in caches or logs. Keep
validation truth solely with the local evaluator. Set offline model-loading
configuration and use local paths; avoid unintended Hub downloads during work.

Persist the local operation reservation before `spawn`, then its returned call ID
before waiting. Reattach known calls with `FunctionCall.from_id`; an unknown
submission outcome requires reconciliation, never a fresh blind submission.
`get(timeout=...)` limits client waiting, not remote execution. On deadline/failure,
cancel the known call with `cancel(terminate_containers=True)`, record the response
and confirm terminal call/container state before any further work. Uncertain
cancellation is unresolved spend; retain identifiers and stop. A receipt is not
an exactly-once guarantee. [FunctionCall API](https://modal.com/docs/sdk/py/latest/FunctionCall).

## Gross smoke estimate and limits

Retain seed824, two selected training pages, one three-epoch fit with three
optimizer updates, two fixed validation pages at base and post-fit (four page
predictions), and one selected-train page before/after checkpoint reload (two
more predictions). No pool scoring or subsequent acquisition round. Model fit,
latency and OCR quality remain unknown.

The 2026-09-16 read-only account rate query reports L40S $1.95/hour, physical CPU
$0.04730/core-hour, RAM $0.008/GiB-hour and Volume $0.09/GiB-month. Thus
`1.95 + 2*0.04730 + 32*0.008 = $2.30060/hour`. Published per-second rates imply
$2.301264/hour, a rounding difference; round up to $2.302/hour below.
[Published pricing](https://modal.com/pricing). Refresh rates before execution.

| Allowance (ESTIMATE, before credits) | Gross USD |
| --- | ---: |
| 40 minutes aggregate GPU startup + execution + restarts | 1.534667 |
| 5 additional minutes aggregate idle/cancellation tail | 0.191833 |
| One CPU-only build: 15 minutes, assumed 2 CPU / 32 GiB | 0.087650 |
| Up to 12 GiB Volume retained for one full month | 1.080000 |
| Subtotal | **2.894150** |
| Remaining contingency inside the $5 smoke ceiling | **2.105850** |

The build resources/time and storage size are assumptions to verify, not configured
provider caps. Charge setup builds, failed work, uploads that invoke compute,
actual egress and all retention against the same gross ledger. Do not assume free
credits/storage; available credit balance is unknown. No automatic rebuild/retry.
Check actual build cost and stored bytes before GPU submission; reduce available
runtime or stop if the remaining ceiling cannot cover the reviewed envelope.
Storage continues after execution; reserve its retention cost within the $30 total
and review retention before the quoted month ends without deleting results silently.

Set a persisted absolute deadline no later than 40 minutes after first GPU
submission, shared by every operation/restart, with worker checks before loading,
each batch and generation. Count queue time conservatively within this wall-clock
deadline. Client monitoring must cancel at it; also track actual aggregate
container time and stop at 40 minutes if reached earlier. Do not reset the deadline
on resume. Per-attempt 600s execution/300s startup limits are separate, exclude
scheduling, and may overrun slightly. `retries=0` does not disable infrastructure
crash rescheduling. These controls bound intended work, **not an instantaneous
provider-enforced dollar cap**; cancellation lag, client loss and delayed billing
remain risks. Stop after the first observed failure and reconcile before review.
[Timeouts](https://modal.com/docs/guide/timeouts),
[crash retries](https://modal.com/docs/guide/retries).

## Release and evidence

Before any resource/build/upload/job: exact reviewed converter/model/transport/
entrypoint commits and dependency lock; accepted source mapping and Volume
amendment; explicit code
execution release within the approved $5 smoke / $30 total; verified account,
mount allowlist and cancellation path. Preserve official READ train/validation
membership and unknown document grouping; this is engineering verification only.

Only then reserve a real run and record code/review SHA, plan/config hashes,
seed, asset hashes, split/group policy, selected/validation IDs, package and image
identity, device/driver/precision, operation/call/container IDs, timestamps/deadline,
attempts, resource settings and gross cost reservations. Capture peak CPU/RAM/VRAM,
tokens/steps, trainable modules, selected-only/mask evidence, losses, prediction
failure flags, checkpoint hashes/reload comparison, cancellation/terminal evidence,
metered/billed costs and reconciliation time. Never log raw GT or credentials.
Use measured smoke results to quote a later pilot against the remaining total.
