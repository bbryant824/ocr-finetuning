# READ2016 converter independent review

> Historical verification: statements apply to the recorded candidate and date.
> See [the project guide](../PROJECT_GUIDE.md) for current status and completed GPU evidence.

Verdict: **PASS for C1–C8**, at exact candidate
`8db583326e55b307df487d57d4c03be1d6e12689`.
Parent: `fad7cae5baf5d7e30ecf9a772152363ab9ca6902`.
Review controls: C1–C7 `b9d5cc2c59c1be73c3d48745205da273443470a0`;
C8 `1c5f0222f32dfcfc3b8937f969dc80e3f9480a2c` (unchanged implementation).

No blocking defect reproduced. After the C1–C7 review, the separately authorized C8 converted
the pinned actual archive, independently compared every page/line/original file, and froze and
reopened all 400 pages. This is engineering verification; it does not establish document
independence, real OCR training, model execution, or scientific performance.

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
hashing/parsing is intentional drift detection; measured full-source runtime appears below.
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
admission cannot bypass the pin. The separate actual positive conversion is recorded under C8.

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
| C8 actual structural smoke | **Passed.** Unpatched public conversion of the pinned archive; independent comparison of all 350/50 pages, 8,367/1,043 lines and 804 originals; explicit-policy full freeze and fresh-process reopen. See measurements below. |

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

## C8 actual-source verification

Executed on 2026-09-16 with the same Python/Pydantic/Pillow versions listed above and a clean
checkout of the exact candidate. No source identity, archive pin or validation was patched.
Both output directories were new; about 19 GiB was free before execution. Inputs remained
read-only. Prepared data and the verification store/snapshot were preserved for later use.

The independent checker streamed every regular archive member and compared its size/hash to
the extracted inventory. All **804 files / 499,592,094 bytes** matched, including images,
XML, `doc.xml` and `list`; the archive's 400 image symlinks were not materialized. Copied files
matched both byte-for-byte and by SHA-256. The original archive and extracted inventory were
hashed again after completion and remained unchanged.

A separate ElementTree traversal of original XML sorted region references and line indices,
decoded literal Unicode without normalization, and calculated each polygon's min/max envelope.
It compared the resulting ordered line IDs, text, boxes and false illegibility flags directly
to every manifest row, without calling the production parser for expected values. Provenance
mapping/counts also matched. All 400 original images decoded as single-frame RGB JPEGs with
the declared dimensions and no EXIF orientation. No transformation was applied.

| Source split | Pages | Lines | Empty strings | Missing baselines |
| --- | ---: | ---: | ---: | ---: |
| Training | 350 | 8,367 | 1 | 2 |
| Validation | 50 | 1,043 | 3 | 0 |

All 400 document IDs remain null under `read2016-official-unknown-engineering-v1`; no TEST
page exists. Full explicit-policy freeze preserved the manifest and every image hash.
Independently serialized expected labels matched the snapshot ground-truth hash. Every model
page projection contained only the nine declared image/identity/policy fields, with no labels
or provenance. A fresh Python process parsed the persisted snapshot and constructed
`LocalOracle` successfully. The initialized SQLite store contained **zero records**: no model,
learning run, baseline, fitting, acquisition or evaluation was invoked.

| Measured phase | Seconds |
| --- | ---: |
| Independent original archive/extracted inventory | 1.80 |
| Public conversion | 10.56 |
| Independent complete output comparison | 7.13 |
| Full freeze and projection/image checks | 16.04 |
| Fresh-process reopen | 7.72 |
| Original/code/no-run postchecks | 0.94 |
| Complete check | 44.18 |

Prepared regular files occupied 501,429,177 bytes; verification regular files before the final
receipt occupied 495,351,680 bytes. These are one local structural execution's measurements,
not throughput guarantees or training estimates. The local receipt retains the command,
checker source/hash, original inventory, environment, timing and paths.

| Identity | SHA-256 |
| --- | --- |
| Original archive (493,223,531 bytes) | `f4748c58af757e06804e638a6e84c2150daab19f50d55e10aac3115e6bfc1756` |
| Published and frozen manifest | `ac0a02a218a2142c6ee92ae964e7828ddfc65545dd6ca12f695815bb4493bdd6` |
| Provenance | `93b0628b5f3e7551b985ef679e755ad1ae16fc4fa7380ac8f66f9b57dfbbf68f` |
| Independently derived and frozen ground truth | `7636a6bbb33979326acf1904e6ea7cc5c67bdabaabc22077d0c36aa5fa1bdbc5` |
| Persisted snapshot JSON | `7666ab67bfce90d6fe03604454cc97366e1d6a45d5075cd302f36c2ae5997173` |

The snapshot JSON hash identifies this execution and includes local absolute artifact paths;
it is not expected to match at another location. Manifest/provenance/ground-truth identities
are independent of the output root for the same source and exact converter code SHA.

The public positive path can be reproduced in a clean exact-candidate checkout using new
`PREPARED` and `VERIFY` locations and an `ASSETS` directory holding the pinned archive and its
verified extracted originals. Set `PYTHONPATH` to that checkout's `src` and use the same Python
for both commands. The independent comparison above is an additional check of the outputs.

```sh
python - <<'PY'
import os
from pathlib import Path
from active_ocr.integrations.public_dataset import convert_read2016
from active_ocr.integrations.simulation import LocalOracle
from active_ocr.models import SourcePolicy
from active_ocr.pipeline import Pipeline

assets, prepared, verify = (Path(os.environ[k]) for k in ('ASSETS', 'PREPARED', 'VERIFY'))
assert not prepared.exists() and not verify.exists()
manifest = convert_read2016(assets / 'Train-And-Val-ICFHR-2016.tgz',
                           assets / 'extracted', prepared, source_policy=SourcePolicy.READ2016)
pipeline = Pipeline.for_simulation(verify / 'store')
snapshot = LocalOracle.freeze(manifest, pipeline.store, source_policy=SourcePolicy.READ2016)
with (verify / 'snapshot.json').open('x') as stream:
    stream.write(snapshot.model_dump_json(indent=2) + '\n')
PY
python - <<'PY'
import os
from pathlib import Path
from active_ocr.integrations.simulation import LocalOracle
from active_ocr.models import DatasetSnapshot, SourcePolicy

snapshot = DatasetSnapshot.model_validate_json((Path(os.environ['VERIFY']) / 'snapshot.json').read_bytes())
assert snapshot.source_policy is SourcePolicy.READ2016 and len(snapshot.pages) == 400
LocalOracle(snapshot)
PY
```

Only this review document changed after C8; the 320-test regression and 38 independent cases
were not redundantly repeated against unchanged source. Manager retains acceptance/integration
ownership. Local hashes detect accidental
drift, not malicious replacement of all evidence. Grouping remains unknown and original splits
are engineering-only; final test remains excluded. Converter tests do not establish model
loadability, trained updates, prediction quality, provider readiness or annotation-time savings.
