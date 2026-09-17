# First historical page OCR pilot

> Historical research rationale and proposals. See [the project guide](../docs/PROJECT_GUIDE.md)
> for the implemented pipeline and measured run; this note does not supersede its current status.

Research recommendation, 2026-09-14. Evidence: primary literature, publisher/Hub
metadata and repository inspection. No real inference, fine-tuning, GPU measurement,
dataset-archive inspection or active-learning result was produced for this note.
All pilot settings below are **proposals for review**, not a frozen final methodology.

## Selection recommendation

**First pair: READ2016 / Bozen, release 1.2.0 (Zenodo 1297399), with
Qwen/Qwen3-VL-4B-Instruct at `ebb281ec70b05090aa6165b016eac8ec08e71b17`.**
Stage only the separately packaged training/validation archive and this model's
required files. Start with full-page inference and a small random-acquisition
adaptation smoke test. This keeps the configured model, supports the intended
page-to-lines task, and limits corpus download to about 493 MB. It is a feasibility
choice, not evidence that Qwen recognizes this handwriting well.

**Later replication pair: Bentham R0 (Zenodo 44519), with the same pinned Qwen.**
Keep the model fixed initially to study transfer of the procedure to English
historical handwriting. Bentham costs more to stage, and its full-page/test
coverage needs verification. It is not a clean independent-pretraining benchmark.

**Smaller model comparator: microsoft/Florence-2-base-ft at
`f6c1a25888ffc1d945ee8a1a77ac833c7303d46e`.** It offers image-to-text-region
generation with much smaller weights. Its historical German recognition, line
granularity and dense-page capacity are unverified. Do not silently substitute it
for Qwen if a smoke test fails. TrOCR by itself is a line recognizer, not a
substitute for either full-page system.

Advance from download to a real comparison only after document grouping, source
labels, geometry, training masks, checkpoint reload and evaluator behavior pass
review. Download permission is not permission to launch paid compute or open the
final test set.

## Sources and search scope

Searched and accessed 2026-09-14. Queries included `Bentham 433 PAGE XML Zenodo`,
`HTR Dataset ICFHR 2016 1.2.0`, `READ 2016 license PAGE`,
`Florence-2-base-ft OCR_WITH_REGION`, and official Qwen3-VL documentation.
Followed dataset-author DOI links, publisher metadata and model-owner repositories.
Search aggregators were leads only. Inclusion: downloadable historical pages with
line transcription/geometry, identifiable releases and stated rights. Exclusion:
line-only mirrors without page provenance, baseline-only annotations without text,
unverified model leaderboards and datasets needing a new commercial agreement.
This is a bounded feasibility review, not an exhaustive literature survey.

| Primary source | Version / read scope | Evidence used and limits |
| --- | --- | --- |
| Sánchez, Romero, Toselli, Villegas and Vidal, *A Set of Benchmarks for Handwritten Text Recognition on Historical Documents*, Pattern Recognition 94 (2019), 122–134, [DOI](https://doi.org/10.1016/j.patcog.2019.05.025), [author manuscript](https://riunet.upv.es/bitstreams/2dfbe9a2-820f-4cc0-8525-fe4c11a0786f/download) | Peer-reviewed article; abstract, §§3–7, dataset tables, particularly manuscript pp. 11–12 and 18–20 | Competition partitions and line/page distinction; historical benchmark results do not measure our pretrained full-page adapters. |
| Toselli et al., [READ2016 1.2.0](https://zenodo.org/records/1297399), with Lorenzo Quirós's revision notes | Publisher description, file table and [JSON export](https://zenodo.org/records/1297399/export/json); publication date 2018-02-01, record created 2018-06-25 | Annotated PAGE release, revised regions, exact downloadable objects; archive members not inspected. |
| Sánchez, [Bentham Dataset R0](https://zenodo.org/records/44519) | Publisher record, [JSON export](https://zenodo.org/records/44519/export/json), and [README](https://zenodo.org/records/44519/files/README.txt?download=1), released 2016-01-08 | Page images, PAGE labels, packaging and unpacked-size guidance; no archive audit. |
| Qwen Team, [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631v2) | Technical-report preprint, v2; abstract, §§3.2.3–3.2.4 and §5.4; PDF cover dated 2025-12-01 | OCR/document training and normalized grounding; does not establish performance or absence of pretraining overlap on these corpora. |
| Qwen, [pinned model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/README.md) and [official repository](https://github.com/QwenLM/Qwen3-VL) | Card, quickstart, relevant README sections; public Hub API metadata read 2026-09-14 | Model identity, license, inference support; historical instructions to install unreleased Transformers are stale. |
| Microsoft, [Florence-2-base-ft](https://huggingface.co/microsoft/Florence-2-base-ft), [pinned config](https://huggingface.co/microsoft/Florence-2-base-ft/blob/f6c1a25888ffc1d945ee8a1a77ac833c7303d46e/config.json) | Model summary, OCR task output, configuration and Hub API | Small encoder-decoder producing region quadrilaterals. No local handwriting evidence. |
| Microsoft/Hugging Face, [TrOCR handwritten model card](https://huggingface.co/microsoft/trocr-base-handwritten), [Microsoft release](https://github.com/microsoft/unilm/tree/master/trocr) | Intended-use/card and release README sections; card explicitly authored by Hugging Face | IAM-adapted single-line recognition requires a separate image-derived detector for page use. |
| Hugging Face, [generation outputs](https://huggingface.co/docs/transformers/en/internal/generation_utils) | GenerateDecoderOnlyOutput and GenerateEncoderDecoderOutput documentation | Distinguishes processed generation scores from raw logits; pin and verify the installed implementation. |
| Guo, Pleiss, Sun and Weinberger, [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html), ICML 2017 | Proceedings abstract only | Calibration is a separate empirical property; this classification study does not validate OCR acquisition scores. |

Public JSON GETs verified model revisions and LFS object sizes; publisher JSON and
DataCite rights records independently identified both dataset licenses as CC BY 4.0:
[READ rights metadata](https://api.datacite.org/dois/10.5281/zenodo.1297399),
[Bentham rights metadata](https://api.datacite.org/dois/10.5281/zenodo.44519).
Some Zenodo web/API requests timed out or were rate-limited; publisher export
endpoints succeeded. Download links are verified as published endpoints, not as
completed bulk downloads. Preserve their notices/attribution when staging.

## Corpus comparison and exact download proposal

| Property | READ2016 / Bozen | Bentham R0 |
| --- | --- | --- |
| Historical source | Ratsprotokolle council minutes; source collection spans 1470–1805. Description calls language Early Modern German; catalog language field says Middle Low German—retain this discrepancy. | Bentham Papers, chiefly English, multiple hands; do not infer that every page is English. |
| Label form | Page images with line-level PAGE XML; 1.2.0 revises region types/polygons and removes spurious regions. | Images plus layout and line transcription in PAGE XML, separate archives. |
| Published benchmark partition | 350 training / 50 development / 50 test pages. Publisher's “400 training” wording describes the combined train/development material. | ICFHR2014 benchmark: 350 training / 50 development / 33 test pages, 433 total. |
| Coverage limitation | Historical competition test PAGE files lacked transcripts; later record includes a test archive. Its precise label contents remain uninspected and irrelevant to initial development. | Historical benchmark supplied test images at line level. Do not assume R0 archive delivers a complete page-level final-test set merely from the benchmark counts. |
| Document grouping | Writer count unknown; an official page partition is not proof of disjoint volumes, meeting records or writers. | Page/folio IDs do not alone establish independent manuscripts or writer separation. |
| Choice | First: smaller download, separate train/validation packaging, explicit revised release. | Later: useful language/layout transfer; larger staging and test-page audit. |

Counts above describe the published benchmarks, not a measured inventory of either
download. Verify unique page images, XML/image correspondence and actual partition
lists after staging; publish any discrepancy before creating a run.

Exact publisher file metadata (decimal bytes; MD5 is the publisher checksum, not
a locally verified download hash):

| Object | Bytes | Publisher MD5 |
| --- | ---: | --- |
| [READ 1.2.0 train/validation](https://zenodo.org/records/1297399/files/Train-And-Val-ICFHR-2016.tgz?download=1) | 493,223,531 | `654f2d2c62055f1847f65ba83bd5d744` |
| [READ separate test](https://zenodo.org/records/1297399/files/Test-ICFHR-2016.tgz?download=1), deferred | 58,636,060 | `894063b8abbe2f671dbb07a144738300` |
| [Bentham R0 images](https://zenodo.org/records/44519/files/BenthamDatasetR0-Images.tbz?download=1) | 2,019,386,713 | `35a00f5e94728191cbc188963d533495` |
| [Bentham R0 GT](https://zenodo.org/records/44519/files/BenthamDatasetR0-GT.tbz?download=1) | 1,450,288,537 | `df4e1cf620c68ea8af7301fd51ed42fd` |

**Version decision:** use 1297399, not a moving `latest` URL. The older
[1164045 release](https://zenodo.org/records/1164045) has a different train/validation
MD5 (`3e7f116ab365098426005fa35b889832`). The revised annotations are not identical
content. Starting with 1.2.0 avoids changing labels after the first experiment.
Do not mix images/labels across releases without explicit equality checks.
Bentham R0 is also distinct from the larger 2015/R1 competition release; the
[TC11 R1 page](https://tc11.cvc.uab.es/datasets/HTR%20Competition%202015_1) carries
a different license statement and points to a different DOI. Do not copy its
license or statistics onto R0.

Bentham's README estimates approximately 2 GB unpacked images plus 3 GB GT; with
archives retained this is roughly 8.5 GB before normalized copies. READ unpacked
size is not yet measured. For staging reserve space for compressed source,
extraction, normalized images, model cache and checkpoints; a weight-only total
is not the required free disk or GPU memory.

Before fitting, preserve original XML/image bytes, archive checksums, locally
computed SHA-256s, manifest version and license notices. Resolve `document_id`
from archival provenance, including recto/verso or other views of the same unit.
Inspect cross-partition grouping and near duplicates, not only exact image hashes.
**Do not assign each page its own invented document ID to bypass leakage checks.**
If a real document crosses official splits, retain the original data and report
the conflict: a separately versioned group-disjoint split needs approval, or use
another corpus. If grouping cannot be established, permit only an explicitly
limited engineering smoke; no document-independent comparison claim.

## Model versions, scope and resource implications

| Model | Immutable Hub revision / license | Published weights | Role / limitation |
| --- | --- | ---: | --- |
| Qwen/Qwen3-VL-4B-Instruct | `ebb281ec70b05090aa6165b016eac8ec08e71b17`; Apache-2.0 | 8,875,719,344 bytes, two safetensors shards | First full-image adapter. JSON line geometry/order is a proposed target requiring verification, not a guaranteed native line detector. |
| microsoft/Florence-2-base-ft | `f6c1a25888ffc1d945ee8a1a77ac833c7303d46e`; MIT | 463,221,266 bytes, one safetensors file | About 0.23B parameters; `<OCR_WITH_REGION>` yields text plus quadrilaterals. The pinned config has 1,024 positional embeddings; dense-page target length is a feasibility gate. |

Metadata sources: [Qwen public API](https://huggingface.co/api/models/Qwen/Qwen3-VL-4B-Instruct?blobs=true)
and [Florence public API](https://huggingface.co/api/models/microsoft/Florence-2-base-ft?blobs=true).
Both were public and ungated when checked. Pin the processor/tokenizer and any
custom code to the same repository revision. Download safetensors rather than
both safetensors and duplicate `.bin` weights. Include config, tokenizer, chat
template and required implementation files; do not download video examples.

| Pinned file | Bytes | Publisher LFS SHA-256 |
| --- | ---: | --- |
| [Qwen shard 1](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/resolve/ebb281ec70b05090aa6165b016eac8ec08e71b17/model-00001-of-00002.safetensors) | 4,967,229,296 | `30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9` |
| [Qwen shard 2](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/resolve/ebb281ec70b05090aa6165b016eac8ec08e71b17/model-00002-of-00002.safetensors) | 3,908,490,048 | `046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6` |
| [Florence weights](https://huggingface.co/microsoft/Florence-2-base-ft/resolve/f6c1a25888ffc1d945ee8a1a77ac833c7303d46e/model.safetensors) | 463,221,266 | `58757d657ff44051314c8030b68e04cb1bb618ca9a4885418f111f6fb708185a` |

READ train/validation plus Qwen weights total **9,368,942,875 bytes**, before
small supporting files. This is a download estimate, not a Modal runtime/VRAM
measurement. Fit activations, optimizer states, image tokens, sequence length and
checkpoint retention dominate additional resource needs. Platform must measure a
bounded smoke and propose a cost ceiling before comparative runs.

Qwen's report describes multilingual OCR and position-aware document training;
its grounding coordinates use [0,1000]. That makes image-to-line JSON adaptation
plausible, but general OCR benchmark claims do not demonstrate this manuscript
task. Its source data include web/internal/synthetic documents; neither it nor
Florence provides enough inspected provenance to exclude Bentham/READ overlap.
Record pretraining contamination as **UNKNOWN** and avoid clean zero-shot claims.
Florence is a cheaper comparator, not an established historical German baseline.
Its native quadrilaterals also need verified conversion to the application's
axis-aligned line regions; word-level detections cannot be relabelled as lines.

TrOCR is excluded from the first page adapter because hidden true boxes cannot
be used to supply its required line crops. A future image-derived detector plus
TrOCR is a distinct two-stage system whose detector training provenance, missed
lines and compute must be counted. A known-layout/GT-crop recognition-only study
would change the task and require an explicit methodology decision.

## Proposed pilot controls

### Smallest useful sequence

1. **Data audit, then zero-budget base check.** Preserve official partitions and
   establish grouping before admitting comparative runs. Inspect ten fixed
   validation pages for parser, coordinate and reading-order failures, then use
   the full official validation set for reported exploratory metrics. Model
   inputs contain image pixels/metadata only. Keep development choices on
   validation; do not choose “easy” images by model performance.
2. **Random engineering pilot.** Propose seed 824, ten pages per acquisition,
   cumulative budgets 10 then 20 from the audited train pool. Fit from the same
   pinned base each round; do not carry optimizer/adapters from round one to two.
   Verify finite loss, an actual parameter update, selected-only supervision,
   checkpoint save/reload and predictions on unseen validation images. A failed
   recognition or geometry smoke is useful negative evidence and a stop point.
3. **Only after acceptance:** compare random, least-confidence and entropy at
   equal page budgets 20/40/60 with batch 20 and paired seeds 824/825/826. Each
   seed has identical first selection across strategies; freeze membership,
   initialization, training recipe and evaluator before the comparison. This
   three-seed pilot is exploratory, not a powered significance study.

These are bounded starting values, conditional on data/compute review. Do not
launch all stages together. A page budget counts every revealed page including
the initial batch; it does not establish equal labor for pages of different
length/density. Report page counts and training cost, with annotation seconds
unknown. Reference-character counts may be reported retrospectively by the
evaluator/oracle, never used to acquire unseen pages.

### Training and image/label boundaries

Propose Qwen LoRA with frozen base/vision, fresh adapter and optimizer per fit,
rank 16, alpha 32, dropout 0, text-attention q/v projections, batch 1,
gradient accumulation 4, learning rate 1e-4 and three fixed epochs as an initial
smoke recipe. Exact module names and a stable dependency lock must be confirmed
by Development. These hyperparameters are unmeasured recommendations, not
optimal settings. Freeze them after validation development; any later tuning
starts a separately identified experiment. Use the same epochs and effective
batch rule at a matched budget, report actual tokens/updates, and do not silently
truncate targets to meet a time limit. Disable unreviewed augmentation initially.

Only cumulative oracle-revealed examples may build training targets. Use full
page images and ordered line text/geometry targets. Mask padding, image tokens,
system/user content and prompt from supervised loss; train assistant target
content and intended termination. Preserve source diplomatic spelling,
punctuation and illegibility flags. Do not give a model source XML, global GT
manifest, validation references, hidden line counts or GT-derived crops.
Training on *revealed* geometry is permitted; using unrevealed geometry for
image preprocessing or acquisition is not. Pin all resizing and save reversible
coordinate transforms; no region-specific processing based on hidden labels.

For Qwen, propose one JSON object with an ordered `regions` list, each containing
line text and `[x1,y1,x2,y2]` in [0,1000]. Convert deterministically to pixel
`Box(x,y,width,height)` using original dimensions. Record raw output and conversion
version. Validate finite, in-bounds positive geometry, unique IDs and line
granularity. A whole-page transcript with an invented full-page box is not a
successful line OCR prediction. If line output is unreliable, stop for a reviewed
detector or explicit transcript-only task change, not hidden-GT assistance.

Choose deterministic decoding (one beam, no sampling, no repetition penalty or
no-repeat-gram constraint) for the proposed pilot; freeze output/image token
limits after validation checks. This is our proposed control, not the model
card's recommended chat sampling recipe. Record truncation, refusal, empty and
invalid outputs as failures; never drop difficult pages from reported results.

Store base/adapter/config/checkpoint hashes, code revision, dependency versions,
GPU type, precision, seeds, complete training and preprocessing parameters,
revealed IDs and run UUID. Reload the checkpoint and reproduce predictions before
declaring a fit successful. Retry from the immutable base and identical selected
set; a crashed paid call must not start a duplicate concurrent training job.
Real resume needs stricter verification than the fixture's unchanged-environment
assumption.

### Evaluation, layout and test isolation

Propose corpus micro-CER and micro-WER on validation: sum per-page Levenshtein
edit counts, divided by summed reference code points or words. Report both
numerators/denominators, plus page-level distributions and failure counts.
Do not average the current single-string CER helper to approximate corpus CER.
Empty references contribute zero denominator and still incur insertions; if the
whole evaluated corpus has no reference text, report undefined text rates with
counts rather than manufacture a finite score. CER/WER may exceed 100%.

Version the initial text view: preserve raw strings; evaluation-only NFC,
CRLF-to-LF, join lines with LF in declared reading order, no case-folding,
spelling modernization, dehyphenation or punctuation removal. WER uses Unicode
whitespace tokenization with punctuation attached. This differs from some
competition tokenization, so no direct leaderboard comparison. Decide treatment
of source illegibility markers before scoring; missing annotations are not blank
pages. Do not silently remove illegible regions from the image or reference.

Use explicit PAGE reading-order metadata where available; XML node order must
not be presumed semantic order. Audit ambiguous marginalia/interline ordering
and propose a source-specific versioned rule before reporting page CER. Do not
repair prediction order using GT. Reference/predicted polygon-to-axis-aligned
envelopes are lossy views; retain the originals and report this limitation.

Text quality alone is insufficient. Also propose one-to-one line matching by
maximum total IoU with threshold 0.5, deterministic ties, and line precision/
recall/F1 plus matched IoU. Use geometry only for this matching, not transcription
similarity. A predicted block covering several lines is not several true-positive
lines. For operational parse failures, report rate and an empty-text hypothesis
convention in an explicitly labelled all-page metric; this counts omissions but
does not bound hypothetical hallucination errors. Never turn parser failure
into a valid empty annotation or report only surviving pages. Existing Prediction
has no rich failure status: Planning must specify where this evidence lives.

Validation serves development and exploratory reporting, so it is not an unbiased
final test. Do not merge it into training. Keep test archive unmaterialized and
test images/references out of prompts, tuning, acquisition and checkpoint selection.
Final evaluation requires a separately accepted frozen dataset/grouping/split
manifest, evaluator, methods, seeds, hyperparameters, checkpoints and stop rule.
No final-test access or final-method freeze is authorized by this note.

### Uncertainty: precise proposal, separately gated

Random requires no confidence scores. Implement and review extraction before
enabling either uncertainty ranker; a number in [0,1] is not proof of validity.
HF documentation distinguishes processed generation `scores` from raw `logits`.
Use raw logits from the original model at each generated prefix (or an equivalent
teacher-forced replay of its **own** tokens), never hidden reference tokens,
beam sequence scores, top-k-renormalized probabilities or generated verbal ratings.

Candidate `generated-text-token-v1` for the structured Qwen output:

- Parse the original assistant output as strict JSON while retaining encoded
  byte spans of each `regions[*].text` string's **content**, excluding its quotes.
  Map original generated token IDs to exact byte spans using the pinned tokenizer;
  require lossless reconstruction, with no whitespace cleanup. Do not retokenize
  decoded text and assume it is the original token sequence.
- Score only ordinary tokens whose nonempty byte span lies wholly within one
  such content span. Include transcription punctuation, numbers and whitespace;
  exclude prompt/image/padding/BOS/EOS/end-of-turn tokens, JSON keys/brackets/
  delimiters, coordinates and tokens crossing text/structure boundaries. JSON
  escape bytes within a string remain part of this serialized-text definition.
  Log included/excluded token counts and boundary exclusions. If exact alignment
  cannot be verified, this score is unsupported; keep random-only.
- For selected positions S, raw logits z_t and full output vocabulary size V,
  compute p_t = softmax(z_t) in float32, natural logs, temperature 1:

  `confidence = exp(sum(log p_t[y_t] for t in S) / len(S))`

  `entropy = sum(-sum(p_t[v] * log p_t[v] for v in vocab) / log(V) for t in S) / len(S)`

  Use stable log-softmax, with `0 log 0 = 0`. V is the full fixed output-head
  vocabulary, including structural alternatives; do not renormalize a text-only
  vocabulary. Assert finite [0,1] within numerical tolerance. Page aggregation is
  token-weighted across all generated lines, not an unweighted line average.
- No included tokens, invalid JSON, refusal or truncation yields **undefined**,
  not “certain” or a maximum-uncertainty sentinel. The present selector requires
  scores for every candidate: stop the uncertainty run and report the failure;
  an alternative fallback policy needs prior review and a new score version.

Least-confidence ranks `1-confidence`; entropy ranks the normalized mean above.
The formulas remove direct product shrinkage with sequence length but do not
eliminate length, tokenizer, layout, language or omission bias. An omitted hard
line contributes no scored tokens and may make a page appear confident. They
measure conditional generated-token uncertainty along one path, not calibrated
probability that the page is correct or Bayesian information gain.

Compare the two proposals only after constant/uniform-logit, boundary-token,
padding, termination, empty-output and batched/unbatched alignment checks pass.
On validation, examine score versus CER, generated length, detected line count
and parse failures before freezing. Do not select score versions using test data.
Do not min-max normalize scores across the pool. Keep raw sequence probability
(strong length bias) and unweighted mean-line scores as documented alternatives,
not silent substitutions. Compare algorithms within each model/corpus; normalized
entropy values across different tokenizers are not inherently commensurate.

Report paired per-seed error differences at matched budgets and learning curves,
with seed spread and failed runs visible. No requirement is imposed that active
learning beats random; a null or worse result is scientifically admissible. Equal
page budgets control revealed pages, while score computation is an extra cost
that must be reported separately.

## Minimal implementation implications and review gates

Inspected code at `a6f31e42b9bbcb33ad2773b91845fa66f9824e34`:
[simulation adapters](../src/active_ocr/integrations/simulation.py),
[Pipeline](../src/active_ocr/pipeline.py), [models](../src/active_ocr/models.py),
[metrics](../src/active_ocr/evaluation.py), [simulation tests](../tests/test_simulation.py)
and [guide](../docs/PROJECT_GUIDE.md#5-active-learning-lifecycle). Inspection, not a new test run, established:

- `LocalOracle` already supplies only cumulative selected training examples and
  separates validation truth. A PAGE importer should emit the existing strict
  JSONL form while preserving originals and explicit splits; document grouping
  cannot be bypassed because the source validator rejects cross-split documents.
- A real `SimulationModel` can preserve the existing reset-fit interface and
  `Prediction` ownership fields. Modal is an execution boundary, not a reason to
  create another orchestrator, service or public GT mount.
- Random skips pool scoring by default, while an explicit versioned validation
  evaluator can run after fitting. A zero-page-budget run completes before that
  hook: base evaluation needs a separate reviewed invocation, not a fictional
  training round.
- Corpus edit-count aggregation, WER, line matching/order, parser failure evidence,
  model training/loading and real checkpoint recovery are not supplied by the
  fixture. Keep these additions small and independently verifiable.

Requested next decisions: accept the exact source/model staging choice; inspect
archive grouping/label completeness; accept a concrete prompt/geometry/evaluator
contract; approve the bounded smoke recipe and compute ceiling after Platform's
estimate. Enable comparative uncertainty runs only after extraction review.
Any necessary regrouping, GT interpretation change, known-layout task or frozen
final evaluation needs an explicit methodology decision. This recommendation
does not itself release those dependencies.
