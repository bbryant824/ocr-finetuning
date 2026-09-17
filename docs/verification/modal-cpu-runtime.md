# First actual Modal CPU build

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
