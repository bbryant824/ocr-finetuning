# READ2016 converter independent review

Verdict: **PASS for C1–C7 only**, at exact candidate
`8db583326e55b307df487d57d4c03be1d6e12689`.
Parent: `fad7cae5baf5d7e30ecf9a772152363ab9ca6902`.
Review control: `b9d5cc2c59c1be73c3d48745205da273443470a0`.

No blocking defect reproduced in the released synthetic converter/policy scope.
**C8 remains unexecuted and requires a separate release:** no actual full READ dataset was
converted or frozen during this review. This is not final converter acceptance, verified
document independence, real OCR training, model execution, or scientific performance evidence.

## Scope and implementation assessment

Reviewed the complete six-file candidate diff and accepted candidate-1B specification.
The changes stay within the source policy, single fixed converter, oracle integrity/projection,
export disclosures, author tests and documentation. Legacy `Page`, importer, storage, selectors,
evaluation, dependencies, CLI and model/provider integrations are unchanged.

Unknown grouping requires explicit caller/config/row/snapshot agreement and the narrow READ
policy; it is not inferred from null IDs. Strict defaults and image duplicate checks remain.
Source provenance is oracle-side and omitted explicitly from fit/predict page projections.
Source XML supplies literal text/order and a declared polygon envelope, with no semantic repair.
The publication boundary is the final manifest, following exclusive destination reservation,
copy validation and provenance checks. Recovery behavior does not promise hardware durability
or remote exactly-once execution.

The roughly 540-line converter includes the fixed schema, parser, inventory/path checks and
publication logic required by the plan. Helpers are reused by freeze/reload validation; no
registry, extra service or generalized ingestion framework was introduced. Repeated source
hashing/parsing is intentional drift detection; actual full-source runtime remains C8 evidence.
No production or author-test edits were made by Testing.

## Executed checks

Python 3.11.14, pytest 9.1.1, Pydantic 2.13.5, Pillow 12.3.0, Ruff 0.16.6.
The configured local environment was invoked directly with exact checkout source imports;
no dependency installation, model download or cloud action occurred.

| Command/scope | Result |
| --- | --- |
| Exact clean candidate `python -m pytest --tb=short` | 320 passed in 34.51s |
| Exact candidate `ruff check .` and corrective diff whitespace check | Passed |
| Independent `tests/test_read2016_independent.py` against a separate clean exact source checkout | 38 passed in 9.23s |

The independent suite constructs its own synthetic XML/JPEGs, 804-file staging inventory,
400-page publication and fault probes. It does not import author test helpers or patch Git
identity, archive pins, source policy checks or expected counts. Its synthetic positive path
uses the lower-level publication helper, explicitly **not** proof of public API acceptance of
the actual archive. Public API negative cases verify that arbitrary archives and implicit
admission cannot bypass the pin. Actual positive conversion belongs to C8.

An initial independent probe hit Python CSV's default 128 KiB reader field cap when reading
hundreds of serialized pool predictions. The test now uses a bounded 2 MiB field limit and
restores it afterward. This was a reader configuration issue, not malformed CSV or a production
fix. No failing application case was suppressed or marked xfail.

| Acceptance | Evidence and result |
| --- | --- |
| C1 strict compatibility | Actual previous application commit `dee5e1a5ecbbc326921e0f4019792c8f7e5deab2` wrote a database without source-policy fields; candidate resumed its 2+1 budget through real JSON/CAS and exported strict known/false disclosures. Independent legacy importer null/missing-ID rejection and strict Page checks pass. Existing cross-split/image and all 1A regressions pass in the full suite. |
| C2 explicit admission | Independent default rejection and explicit 350/50 unknown-document admission pass. Null without policy, mixed policy, fake known ID, omitted required document, TEST, inconsistent provenance/mapping and snapshot metadata refuse. Author suite additionally covers wrong archive/version/coverage and same-split duplicate READ images. |
| C3 literal text/order | Independently authored reversed XML nodes/geometric order with gapped region indices preserve indexed order, decomposed Unicode, CR/LF character entities, whitespace, empty strings and abbreviated/struck/partial-unclear text. No region summary becomes a line target; no inferred illegibility. Ambiguous references/order/text and unsupported flags refuse. All 804 synthetic originals are byte-identical after publication. |
| C4 geometry/orientation | Hand-computed nonrectangular envelopes match maximum-minus-minimum bounds, including a margin at original dimensions. Missing baselines/structure remain represented by original XML and counts. Degenerate/nonfinite/out-of-bounds inputs refuse; image-mode/orientation/size/format/truncation probes reject without transformations. |
| C5 provenance/mutation | Independent manifest truth/policy, provenance, XML/doc.xml/image and snapshot metadata changes reject. Corruption of actual saved image/manifest bytes during freeze rejects. During-fit source mutation prevents round publication and rejects completed resume. Identical clean code/source produces byte-identical manifest/provenance across roots. |
| C6 publication/filesystem | Independent public pin/default/collision refusal, archive traversal/absolute/hard-link/FIFO probes, two concurrent publishers, handled interruption with unrelated-content preservation, and lost publication response pass. Winner/completed manifests remain readable. Author suite additionally covers duplicate/missing/extra members, symlink targets, staged symlinks and copied-byte mutation. |
| C7 isolation/export | Independent spies see only selected TRAIN target lines; model page keys contain no provenance/XML/regions/counts. Validation labels reach only the evaluator; validation reveal rejects. Fresh process reload/export preserves null IDs and exact JSON; CSV discloses READ policy, unknown grouping, engineering-only, and fixture kind. |
| C8 actual structural smoke | **Not run.** Requires separately released actual 350/50-page conversion, literal source comparison, original hashes, full freeze/reopen and counts/hash evidence. |

Historical JSON compatibility is tested with bytes written by the old application, not a new
model dump edited to look old. Synthetic freeze, mutation and concurrency probes exercise actual
files/SQLite, not only mocks. Fault injection is limited to explicit I/O interruption or mutation;
validation and identity remain active. The exact source clone was clean after execution.

## Reproduction

Use a local environment with the project's development dependencies. Keep the independent test
outside the clean source checkout so untracked review files do not change its Git identity:

```sh
review_dir=$(mktemp -d)
review_tests="$PWD/tests/test_read2016_independent.py"
git clone --quiet --no-local --no-checkout . "$review_dir/source"
git -C "$review_dir/source" checkout --quiet --detach 8db583326e55b307df487d57d4c03be1d6e12689
PYTHONPATH="$review_dir/source/src" python -m pytest \
  -c "$review_dir/source/pyproject.toml" -o pythonpath= --import-mode=importlib \
  "$review_tests" --tb=short
```

Expected result: 38 passes. Retain the historical commit above for the `git archive` compatibility
probe. No actual READ assets are required for this command; no production pin is relaxed.

Manager may accept this C1–C7 evidence and separately release C8. Local hashes detect accidental
drift, not malicious replacement of all evidence. Grouping remains unknown and original splits
are engineering-only; final test remains excluded. Converter tests do not establish model
loadability, trained updates, prediction quality, provider readiness or annotation-time savings.
