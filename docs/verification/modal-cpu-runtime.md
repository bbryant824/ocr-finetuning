# Actual Modal CPU builds

## Attempt 1

**Result: FAIL, 2026-09-17.** The locked Linux build reached the CPU test gate,
but all eleven intended tests failed during processor fixture setup with
`ValueError: processor settings changed`. No successful CPU report, BuildReceipt
or accepted runtime image was produced. This is not a GPU or OCR result.

## Frozen inputs and execution

- Source: `ca3c60c3eb03b51ef442b80e53dade60673e30f2`.
- Independent offline review: `877bd11e03be296abaef2a0974a9e01049ab0397`;
  see [integration review](modal-runtime-review.md).
- BuildSpec SHA-256:
  `b9ced5a273885200dd1e29a06a158e5e0593adf2b4233b911bae520b441747b6`.
- Frozen hashed requirements SHA-256:
  `80ec96e25655ae315b3ef5f791a5956b350672cde452025f9617bee730add660`.
- Exact seven runtime files, one synthetic test file, lock/requirements and
  11,499,725 bytes of verified pinned processor assets; no real pages, ground
  truth or 4B weights. The separate local Python3.11.14/Modal1.5.5 client
  environment contained no GPU/ML packages.
- One explicit `build_image` invocation followed the
  [deployment recipe](../modal-deployment.md), with Debian/Python3.11 and
  hash-required dependency installation. CPU gate requested CPU2, RAM32768MiB,
  GPU=None, timeout900s and child watchdog850s; no automatic retry.
- Operator attempt: 01:04:55.015322–01:08:18.630811 UTC, 203.50 seconds including
  resource setup, dependency installation, image layers and failure handling.

## Observed failure

Dependency installation and allowlisted image layers completed. The gate ran
`tests/test_qwen.py -k 'test_actual_ and not staged_headers'`. Its retained JUnit
summary reported eleven cases, eleven setup errors, zero passes, zero skips,
and pytest return code1. All errors had the same processor-settings message.
The exception originates in `qwen.load_processor`'s settings comparison;
the failing property/value was not included in the bounded diagnostic, so its
specific cause is not established here. Test bodies did not establish tiny-model
behavior, training, fresh-process reload or generation correctness.

The fallback build log retained the bounded diagnostic. Its extracted canonical
JSON SHA-256 is
`a306dc0b64ffb2452d77f3ef2459d62313b5d338fd739a1f6af8f578890693aa`.
The output Volume listing showed only the empty build-spec directory: no durable
failure file, CPU report or receipt. Diagnostic persistence therefore also needs
investigation; no cause is inferred from its absence. Raw logs, actual provider
IDs, timestamps, listings and indexed hashes remain in private execution evidence.

After the exception, provider inventory confirmed the build App stopped and no
running containers. The two approved Volumes remain preserved; no GPU Function
was deployed and no 4B weights, real pages or learning round were involved.

## Cost and next gate

The immediate provider report attributed USD0.00220927 to this App
(CPU0.00213731, memory0.00007196). This is a provisional observation: aggregate
monthly metered/billed totals still showed zero, so reporting lag and additional
unsettled charges are not excluded. Retain the full USD0.087650 CPU-build reserve
inside the existing USD5 first-smoke/USD30 total budget until reconciled; do not
treat the observed fraction of a cent as a final bill.

Stop at this failed attempt. Development must diagnose the processor comparison;
Platform must resolve the missing durable failure diagnostic within an assigned
correction. Preserve these failed outputs, independently review any correction,
then obtain a bounded retry release. All eleven actual CPU checks must pass with
zero skips before a GPU round can be released.

## Attempt 2 — eleven CPU passes, receipt failure

**Build result: FAIL, 2026-09-17. CPU checks: 11 passed, 0 skipped, 0 failed.**
This single corrected attempt used clean source
`7ea96c3b9b10612a0751ff78a668cbe076c18ef9`, independently reviewed at
`4753dd0600047d6fc03f6c469b1ecf28c7ff6e68`, unchanged dependency lock/export,
existing processor assets and the same named resources. No separate diagnostic,
GPU work or automatic retry was performed.

New BuildSpec SHA-256:
`ac5f9a5c32fa819432182c51c5266dce4b0d9113ef8e9806afd7b5aca8dff702`.
The attempt took149.47seconds including setup/image layers and failure handling,
ending01:38:13.076701UTC. The output Volume durably contains the canonical
829-byte `cpu-report.json`, SHA-256
`274ff01ffbf962724bbd0590a6a32af05cc167fc6dafcd4a983c0a51ab41df39`.
Readback verified eleven distinct intended test names and the exact zero-skip
pass counts. These are real synthetic processor/tiny-model/fresh-process CPU
checks, not a4B/CUDA result. The corrected mount/publication path now produced
durable output.

After these passes, `BuildReceipt` construction rejected the observed package
inventory with `build package inventory must be sorted with unique names`.
The retained exception does not identify the duplicate names/versions; those
details remain to be diagnosed. No BuildReceipt or accepted runtime image exists,
so the runtime acceptance gate remains closed despite the passing CPU report.
Provider inventory confirmed the App stopped and no containers remained.

Immediate billing observations remain provisional: this App's per-resource
report showed USD0.00393008, while the workspace ephemeral-App breakdown had
reached USD0.01130515 across work so far. The summary showed USD0.01 metered,
USD0 billed and a USD0.01 credit adjustment; reporting intervals/lag differ.
Continue gross accounting inside the existing USD5/USD30 ceilings, without
assuming further free credits. Preserve both attempts and correct/review only
the reproduced inventory-receipt blocker before any further build release.

## Attempt 3 — successful CPU build and durable receipt

**Measured result: build succeeded, 11 passed, 0 skipped, 0 failed, 2026-09-17.**
This single released attempt used clean source
`1a413d7e373a1182c3552bd9c05c9687fef14485`, independently reviewed at
`19c48c3ed655dec1150471e12ac37a6017a49b98`. It reused the frozen lock/export,
client environment, processor assets, cached dependencies and existing Volumes.
No GPU, real4B weights, additional diagnostic or local test campaign ran.

The build ran02:03:09.044837–02:05:47.808315UTC,158.76seconds including setup
and image work. Its canonical receipt records Python3.11.12 and96 unique effective
packages; required direct pins passed. The helper re-read both durable artifacts,
checked canonical bytes, report size/hash and source/code/spec identities, and
returned the actual completed image ID. The reviewed successful step writes the
same canonical receipt into the image; no separate image-inspection container
was launched. Actual provider IDs and full package inventory remain in private
indexed evidence for independent review.

| Identity | SHA-256 |
| --- | --- |
| BuildSpec | `6a73443392d9c38552afbe279884c9ac5c37cf32030e3a729be0730ec73f9288` |
| BuildReceipt | `23bcabdd3fe6c0a89a36c31b09c9e21e24010bec2c8e9b74558e07d4ea7dc069` |
| CPU report | `274ff01ffbf962724bbd0590a6a32af05cc167fc6dafcd4a983c0a51ab41df39` |
| Code bundle | `dbcf09900b9f828f05a7b87ff23c260182e6bfd0a31331646ae735f25f174412` |
| Remote environment | `1ca1ef0d869fd49340cbe1cfdb8ade6f3e016cfa900443a11f60c6084a98cf14` |

Provider inventory confirmed the App stopped and no active containers. The latest
provisional workspace App-usage breakdown was USD0.01865657 across work so far;
the rounded summary showed USD0.02 metered,0 billed and a0.02 credit adjustment.
Keep gross USD5/USD30 accounting and reporting-lag caveats; remaining credits
are unknown. Preserve all prior failures. This completes author CPU evidence;
independent CPU acceptance still precedes any GPU deployment or real OCR round.

## Attempt 4 — corrected dispatcher CPU build

**Measured result: build succeeded, 11 passed, 0 skipped, 0 failed, 2026-09-17.**
One released build used clean source `7dd40ec6cc444c939bdd09b42963bbf639cecbca`,
independently reviewed at `77b762f91f82f79df08c2e1115aaa23d5ed9bc75`. Only the
dispatcher differs from attempt 3's seven runtime files; Qwen, CPU test bodies,
lock, hashed requirements and processor assets are unchanged. Existing cache,
client and Volumes were reused. No local test campaign, separate diagnostic,
real page/model-weight upload, GPU allocation or deployment replacement occurred.

The existing `build_image` helper ran once with CPU2 / RAM32768MiB / GPU=None,
900-second timeout and 850-second child watchdog. Operator elapsed time was
142.588 seconds, 04:01:11.562126–04:03:34.152830 UTC. The actual image returned was
`im-82wRAkCIoH38W1LnO3G4KT`. The durable canonical report and receipt were read
back and checked against source, code, spec, byte count and hashes. The receipt
records Python3.11.12 and 96 effective package names; the environment digest
matches attempt 3. No separate image-inspection container was launched.

| Identity | SHA-256 |
| --- | --- |
| BuildSpec | `c876d946ac6635d8cd9172bafb7c9873cced759ed9fa6d0248177762919f1ecf` |
| BuildReceipt | `cb477825e80734235813a86eeb98309455e814ba3a19a02ba185fc19609cc550` |
| CPU report | `274ff01ffbf962724bbd0590a6a32af05cc167fc6dafcd4a983c0a51ab41df39` |
| Code bundle | `2a641c51600b848bb93557afa4d8b193e46108e1c158400bcef6cf33727d62ed` |
| Remote environment | `1ca1ef0d869fd49340cbe1cfdb8ade6f3e016cfa900443a11f60c6084a98cf14` |

The build App stopped and provider inventory listed zero containers. The previous
dispatcher remained deployed with its original Function ID and zero warm containers;
its failed run/control was not changed. All prior attempts remain preserved.
The immediate attempt-specific resource report attributed USD0.00357553
(CPU0.00097151, memory0.00260402). Latest aggregate accounting reported
USD0.42 metered / USD0 billed after credits,
provisional; retain the earlier USD0.45 observation and separate diagnostic costs
rather than interpreting lower later totals as refunds. Storage and final billing
remain unsettled. Gross firstUSD5/totalUSD30 bounds remain in force.

This is author evidence for the corrected CPU image, not independent acceptance
or 4B/CUDA execution. Evidence-only review must precede any GPU release. Exact
invocation, inputs, returned identities, logs and accounting remain indexed in
private build records; the standard command recipe is unchanged in the
[deployment guide](../modal-deployment.md).

## Attempt 5 — corrected adapter export CPU build

**Measured result: 11 passed, 0 skipped, 0 failed, 2026-09-17.** The single
standard build used clean source `1a03f77519439c0601a83c8c3b4670ca896de337`,
with independent focused review `c5d1d87d1c1c3276ba8cbe649d1583aef612c4c2`.
Only Qwen and the test file changed among build inputs; pins, requirements and
processor assets remained unchanged. The existing adapter case now exercises
36-layer injection/export and strict configuration validation; the gate still has
eleven cases. Cache, client and Volumes were reused without a local test campaign.

The reviewed helper ran once, CPU 2 / RAM 32 GiB / no GPU, timeout 900 seconds
and child watchdog 850 seconds. It took 174.973 seconds, from 05:08:39.505 to
05:11:34.455 UTC, and returned image `im-kTX4rgt7qqc1JzkmhM1vxb`.
Canonical report/receipt were read back from the durable Volume and checked
against exact source, code, spec, sizes and hashes. Python 3.11.12 and all 96
effective packages match the prior accepted environment.

| Identity | SHA-256 |
| --- | --- |
| BuildSpec | `028c1c0444208a92b16f536e46117b61481df60c7cdc71c538dea0db48f142a7` |
| BuildReceipt | `bdb8fd441b19d646d4c85d574ad0b10b5ee5c5f98146268c3172963175473348` |
| CPU report | `274ff01ffbf962724bbd0590a6a32af05cc167fc6dafcd4a983c0a51ab41df39` |
| CPU test file | `9262d9554b990d0df6767f9cac30d78dd621ea31f8b0ac51834bc309bd3756be` |
| Code bundle | `5e8aff34f3281e03b122999f21b9735efcf329f9354dc5a8bfdf1737d2307051` |

The App stopped and final inventories showed zero tasks/containers. An earlier
lagged task-count observation is retained. Existing deployment, old run state
and indexed evidence remained unchanged. Provisional build cost: USD 0.00484499
(CPU 0.00148693, memory 0.00335806); aggregate USD 0.60 metered / USD 0 billed
after credits. Storage/settled costs remain unknown. No GPU or deployment
replacement occurred. This image awaits independent evidence review before a
new execution release; the earlier failed GPU fit remains preserved.
