# Historical OCR: random ICL versus the existing fine-tuned reference

**3 October 2026 · Development and the bounded readiness check approved; full experiment pending.**
This supersedes the earlier initial protocol. Implementation extends the existing region runner;
[EXP-014 preparation](../experiments/EXP-014.md) records verification and execution commands.
The broader primary-paper review is preserved at the end as background.

## 1. Goal and initial scope

Test whether the original Qwen model improves historical line recognition when each query receives
**0, 8 or 24 labelled line examples**, and compare its CER/WER with the existing 24-page LoRA result.

Use one fixed 24-page TRAIN bank, seed **824**, and random demonstrations only. The initial pilot has
no new fine-tuning, additional bank seeds, repeated context draws, visual retrieval, prompt search,
LoRA+ICL arm or faithfulness/mechanism audit. These are outside this execution scope.

Two research questions:

1. How does original-model recognition change from zero to eight to 24 demonstrations per query?
2. Can either ICL condition approach the existing LoRA checkpoint trained using the same label bank?

This is an exploratory one-bank comparison. Per-query random examples vary the context, but do not
measure robustness across bank choices or repeated draws. Do not claim equivalence from a small gap
or assume ICL is cheaper than fine-tuning.

## 2. Frozen setup and existing references

| Item | Setting |
| --- | --- |
| Task/data | READ2016 v1.2.0; supplied text-line boxes; recognition of crops, rejoined in source reading order. |
| Model | Qwen/Qwen3-VL-4B-Instruct, revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`. |
| Label bank | Exact EXP-013 random round-1 selection: 24 TRAIN pages / 589 labelled lines, selection seed 824. |
| Examples/query | 0, 8 or 24 distinct line crop/transcript pairs; one context draw per query. |
| Query input | One validation line crop, with its transcript withheld from the model. |
| Crop processing | Existing clamped 8-pixel padding; longest edge at most 1024, preserving aspect ratio. |
| Instruction | `Transcribe only the text in this cropped line. Return only the transcription.` |
| Decoding | Existing greedy generation, repetition penalty 1.1, 64 new tokens, BF16. |
| Pipeline check | Eight fixed VAL8 crops from Seite0355–0362, chosen using IDs/geometry before looking at outputs. |
| Full evaluation | Same VAL42, Seite0363–0404: 861 lines; existing `page-text-nfc-v1` evaluator. |
| Runtime | Existing Transformers/PyTorch pipeline on Modal L4 with the working CPU/memory allocation. |

Reuse EXP-013's recorded membership, rather than generate another selection that merely uses seed 824.
Bank receipt: `.local/verification/exp-013/region-exp013-v2-random-seed824/selection-round-1.json`.
Verify the 24 unique TRAIN pages, all 589 line IDs and source identity; preserve original labels/splits.
The settings remain aligned with [the known-box recipe](../experiments/recipes/read2016-known-box-al-24.json).

| Comparison | Label use | CER / WER | Execution status |
| --- | --- | --- | --- |
| Original model, 0-shot | No weight updates; no demonstrations | 0.803646 / 1.087103 | Reuse matching EXP-012 zero-page predictions, as previously reused in EXP-013. |
| Original model, 8-shot | Eight line examples/query sampled from the 24-page bank; no weight updates | To measure | One new full VAL42 pass after the pipeline check. |
| Original model, 24-shot | 24 line examples/query from the same bank; no weight updates | To measure | One new full VAL42 pass after the pipeline check. |
| Existing 24-page LoRA, 0-shot | All 589 bank lines used by its one-epoch fit; no inference demonstrations | 0.410374 / 0.825421 | Reuse EXP-013 random round-1 predictions and checkpoint. |

Original baseline receipt:
`.local/verification/region-ocr-v2/region-exp012-random-seed824/round-0.json`.
LoRA receipt: `.local/verification/exp-013/region-exp013-v2-random-seed824/round-1.json`.
Adapter: `.local/models/exp-013/random/seed-824/round-1/adapter/`.

Before accepting either reference, match model revision, source/validation IDs, crops, processor,
instruction, decoding and evaluator; recompute its saved predictions locally. Verify the adapter's
recorded weight hash. Preserve original receipts and label results **reused**, not newly executed.
If identities do not match, stop to resolve the mismatch rather than silently substitute a control.

Eight or 24 **shots** are line boxes, not pages. ICL uses a **24-page available label bank** even though
only a subset appears in one prompt. Fine-tuning and ICL share available labels but expose them to the
model differently; this is not identical data exposure or a measurement of human annotation time.

## 3. Random context assembly and prediction flow

- Sort the bank by stable region ID. For each query, derive a deterministic random seed from seed 824,
  the bank identity and query ID, using a stable hash rather than Python's process-dependent hash.
- Uniformly sample 24 distinct bank lines without replacement. The 8-shot condition uses the first
  eight of that same sample; the 24-shot condition uses all 24, preserving sample order. This controls
  the size comparison without adding more draws or seeds.
- Different queries usually get different examples; overlap/reuse is allowed. Lines can come from
  the same TRAIN page. Do not force artificial page diversity or use query transcripts for selection.
- Assemble paired user crop/instruction and assistant true-transcript demonstrations, followed by
  one user query crop/instruction. Numeric coordinates and whole pages are not fed to the model;
  supplied coordinates serve cropping and source-order reconstruction only.
- Generate only the query transcription. Each query is independent: no prior predictions, validation
  labels or conversation history are carried forward; model weights remain unchanged.
- Preserve the original single-crop input exactly for 0-shot. Save per-query example IDs/order,
  predictions, input/image token counts, cap/failure flags and processing identities for replay.

One full condition predicts every line of every VAL42 page: **861 logical predictions**. Batching is
permitted; this does not require a separate cloud job for every crop. Join predictions by original
reading order and score against validation truth in the existing evaluator only.

## 4. Execute in stages: verify first, then run full validation

**Permanent execution rule for this plan:** after any relevant pipeline change, pass the smallest local
checks and a bounded GPU check before launching a full validation pass. A working pipeline and an OCR
improvement are separate outcomes. No accuracy threshold is required to make a negative result valid.

### A. Freeze inputs, reuse checks and implement the small change

After approval, extend the existing region runner/input builder and use one concise ICL recipe.
Reuse crop processing, generation, evaluation, receipts and resumable output; add no new dependencies,
feature index, orchestration layer or training pipeline. Existing EXP-012/013 modes stay supported.

Run focused local checks only:

- bank membership and TRAIN-only labels; no validation label can enter demonstrations or query input;
- correct crop/transcript associations, exact 0/8/24 counts and distinct demo IDs;
- deterministic contexts on replay; eight examples are the prefix of the corresponding 24;
- original 0-shot inputs unchanged and matching baseline/provenance accepted;
- completed outputs reused only when model, recipe, source, prompt, query, context order and decoding
  identities match; mismatches rejected.

### B. Small L4 pipeline check — required gate

Freeze eight diagnostic VAL8 crop IDs, one per page, spanning narrow and wide lines using geometry.
Run all three shot counts on those same crops: **at most 24 generated outputs**. Start with the
24-shot query having the largest processed input among them, so a capacity failure stops work early.

Verify actual input tokens plus 64 reserved output tokens fit the runtime/model limit; all 25 images
remain present in a 24-shot prompt; no demo or query is truncated; image/text associations are correct;
and L4 can generate and save outputs with measured peak memory and latency. Locally check saved
coverage, ordering and metric computation. Inspect a few prompt/prediction pairs for assembly errors,
without introducing a faithfulness study or selecting a better-performing context.

**Proceed only after these checks pass.** If 24 shots cannot run correctly on L4, pause and report the
specific token/memory failure for a scope decision. Do not silently reduce shots, shrink crops, change
model/GPU/decoding, or launch full evaluation anyway. The check belongs to this experiment's evidence,
not a separate EXP. VAL8 check results are not reported as full VAL42 results.

### C. Full evaluation

Freeze the recipe after technical verification. Execute the 8-shot VAL42 pass first; verify complete
861-line coverage and locally recompute its metrics before starting the 24-shot pass. Use the same bank,
query IDs and nested contexts for the size comparison. Runtime token checks also apply to each query.

**Initial workload:** zero new fits, two new full passes / **1,722 predictions**, plus at most 24 GPU
check outputs. Reused original/LoRA evaluations add no full GPU pass when matching. Keep all outputs,
including empty/capped predictions; record failures rather than filtering them to improve scores.

Resume completed matching queries after interruption. Preserve interrupted attempts separately. Pause
on leakage/identity/assembly failures, OOM or materially unexpected resource use; resolve the issue and
repeat the small affected check before resuming full work. Existing experiment evidence stays intact.

### D. Verify and report

Check complete unique query coverage, saved context identities and unchanged source data. Recompute
all final CER/WER locally from saved predictions, including the two reused controls. Report all four
conditions, character/word edit counts, empty and capped outputs, unique demonstration usage, prompt
tokens, measured GPU runtime/memory and reuse provenance. Record measured efficiency; do not infer it
from the absence of training. No multiple-seed averages, significance/equivalence claims or linguistic
audits are part of this initial pilot.

Interpret outcomes against both references: improvement over zero-shot, remaining gap to the existing
24-page adapter, and whether adding shots helps or hurts. One bank and an already examined validation
slice cannot establish repeatable adaptation savings or independent generalization. The historical
LoRA fit is a single realized reference, not an average over initialization/training variation.

## 5. Records, assets and approval boundary

For the next approved experiment, follow the existing conventions:

- reader report `results/EXP-ID.md` and technical record `experiments/EXP-ID.md`;
- one discrete recipe with bank/context seed 824, 0/8/24 shots and all frozen settings;
- ignored `.local/verification/exp-ID/icl-random-seed824/` for contexts, predictions, receipts,
  baseline provenance, diagnostic check, metrics, app IDs/logs and interrupted attempts;
- reference the existing base and EXP-013 adapter; **no newly trained model/adapter is produced** and
  no empty or duplicate model directory is needed. Retain the existing local mirror/remote originals.

Source data: `.local/assets/read2016-1.2.0/`; prepared data: `.local/prepared/read2016-v1-8db5833/`;
frozen source: `.local/verification/active-learning-source/runs/`;
base model: `.local/assets/qwen3-vl-4b-ebb281e/`.

Update the results index and concise project state only after verified execution. EXP-014 is reserved for this pilot; its technical record distinguishes readiness evidence from
full results. The user approved development and the bounded readiness check in this turn. Full VAL42
execution is deferred to a later prompt; no full experiment or new fit has run.

## Verified project position before ICL development (background)


Canonical checkout: `/Users/bowenbryant24/Downloads/FYP`, main commit
`914977622435fa17f480fccb4d78ffcd9b217576`; executed EXP-013 runner `95bcd6b`.
Reviewed the [project guide](../docs/PROJECT_GUIDE.md), [reader results](../results/README.md),
EXP-009–013 records/recipes, current region/acquisition code and historical research notes.

| Evidence | What it establishes | Important boundary |
| --- | --- | --- |
| EXP-001–008 | Joint page line-box/text execution was engineered, but predictions remained malformed, repetitive or capped. | Paused that recipe; not evidence that every joint OCR approach fails. |
| [EXP-009](../results/EXP-009.md) / [010](../results/EXP-010.md) | Whole-page random controls improved with adaptation: final 192-page CER/WER 0.216/0.618 for LightOnOCR and 0.211/0.559 for Qwen. | Similar numerical recipes do not equal identical processors/prompts; no selector advantage follows. |
| [EXP-011](../results/EXP-011.md) | Three-seed whole-page selection comparison found no consistently superior selector across the tested budgets/metrics. | Shared random 64-page warm start; small differences and capped outputs complicate interpretation. |
| [EXP-012](../results/EXP-012.md) | Known-box Qwen random CER/WER: 0 pages 0.803646/1.087103; 64 pages 0.189807/0.580935; 192 pages 0.094367/0.370467. | One seed; one-epoch region recipe differs from the whole-page recipe. |
| [EXP-013](../results/EXP-013.md) | At 24 pages entropy led (CER 0.355909 vs random 0.410374). At 48/72 pages random led both metrics (CER 0.235749/0.175020). | One configured seed; LoRA initialized before Trainer seeding. Small gaps are not controlled evidence of selection superiority. |

READ2016 v1.2.0 has 350 official TRAIN and 50 VAL pages. Eight VAL pages serve development;
reported VAL42 has 861 supplied text-line crops. Annotation is simulated using source transcripts.
The region pipeline recognizes text **with supplied boxes and reading order**, not box detection.
Document/writer grouping and pretraining overlap are unknown. No independent final test was used.
Fine-tuning substantially improved these validation slices; repeatable active-selection savings remain
unproven. This motivates studying **use of available labels**, without claiming active learning is futile.

The current runner uses Hugging Face Transformers/PyTorch + PEFT LoRA on Modal L4. Its input builder
accepts one query crop and an instruction; multi-image demonstrations are not implemented or verified.
The existing one-epoch adapter is a fixed comparison reference, not an optimized upper bound on
fine-tuning. No feature extraction or retrieval is required for the initial random-only ICL pilot.

## Primary-paper review retained as background

The earlier Manager/Research review is retained for scientific continuity. Its broader directions and
controls are not requirements for this initial random-only pilot; no new literature search was needed
for this scope revision.


Search/access date: **3 October 2026**. Manager and Research independently searched for historical HTR,
multimodal ICL, visual/textual retrieval, writer adaptation, OCR correction and transcription rewriting.
Queries included `historical handwritten text recognition vision language models in context learning`,
`handwritten OCR few shot transcription`, `OCR retrieval demonstrations`, and
`vision language model OCR faithfulness linguistic bias`. Primary arXiv full texts and official
CVF/COLM/ICLR records were preferred. Included work with a relevant protocol or diagnostic;
excluded generic product claims and secondary summaries as evidence. This is a focused review,
not an exhaustive systematic review or meta-analysis; scores below are not cross-paper benchmarks.

| Primary source, status and read scope | Evidence and critical limitation |
| --- | --- |
| Eric Ayllon, Abel Gandia, Jorge Calvo-Zaragoza. [Exploring In-Context Learning for Handwritten Text Recognition](https://arxiv.org/html/2609.37195v1), September 2026 preprint. §§3–6, Appendix A/Table 5. | Six collections, four VLMs, 0–16 shots, 30 sampled contexts. Qwen3-VL-8B ICFHR2016 mean CER: 80.95% zero-shot → 78.78% at 16 shots; 1–8 shots worsen it. Minimum scores choose contexts per query using reference errors, an oracle diagnostic. Different model/protocol; no matched LoRA comparison established. |
| Tom Simon, Stéphane Nicolas, Pierrick Tranouez, Clément Chatelain, Thierry Paquet. [Few-shot Writer Adaptation via Multimodal In-Context Learning](https://arxiv.org/html/2603.29450v1), March 2026 preprint. §§3–4, Table 2. | A specially trained 8M CNN–Transformer plus OCR fusion reaches IAM CER 4.22% → 3.92%, RIMES 2.91% → 2.34% with nine context lines. Requires writer identities and labelled support from the query's own split. Positive evidence for context, not off-the-shelf Qwen or our TRAIN-only support policy. |
| Shuo Chen et al. [Can Multimodal Large Language Models Truly Perform Multimodal In-Context Learning?](https://openaccess.thecvf.com/content/WACV2025/html/Chen_Can_Multimodal_Large_Language_Models_Truly_Perform_Multimodal_In-Context_Learning_WACV_2025_paper.html), WACV 2025. Full-text §§3–5 via [author manuscript](https://arxiv.org/html/2311.18021v2). | Demonstration-image removal often retains gains; MMICES combines visual filtering and textual reranking. Mainly older models and VQA/captioning: useful controls and retrieval prior art, not proof that handwriting examples are unnecessary. |
| Folco Bertini Baldassini et al. [What Makes Multimodal In-Context Learning Work?](https://openaccess.thecvf.com/content/CVPR2024W/PV/html/Baldassini_What_Makes_Multimodal_In-Context_Learning_Work_CVPRW_2024_paper.html), CVPR Workshops 2024. Full-text §§3–5 via [author manuscript](https://arxiv.org/html/2404.15736v2). | Image importance depends on task; identifies text shortcuts and recency/copying. Image-to-text tasks can require demonstration images. Classification majority-vote findings must not be transferred directly to literal OCR. |
| Gwang Gook Lee et al. [Do VLMs Read or Rewrite? On Transcription Faithfulness in Vision-Language Models](https://arxiv.org/html/2607.21617v2), 2026 preprint. Dataset/evaluation, failure filtering, Appendix F. | Printed English/Chinese/Korean perturbations, not historical handwriting. Qwen3-VL-4B no-correction instruction reduces scramble degradation +2.04 → +1.07 WER points, but clean WER rises 0.29% → 0.66%. Main metrics condition on successful outputs. Shows a possible tradeoff; does not prove our errors are rewrites. |
| Ben Gutteridge et al. [Judge a Book by Its Cover: Investigating Multi-Modal LLMs for Multi-Page Handwritten Document Transcription](https://arxiv.org/html/2502.20295v3), 2025 preprint revised 2026. §§3–5, Table 2. | Bentham Gemini OCR+one selected page CER 8.48%, versus OCR text alone 9.54%. Motivates image-grounded correction, but uses predicted OCR and document context, different models and limited repetition. READ's unresolved grouping prevents direct document-level replication. |
| Moshiur Farazi et al. [When Do VLMs Help Arabic Manuscript OCR? A Cross-Dataset Study](https://arxiv.org/html/2608.22366v1), August 2026 preprint. §§4–6, Table 8. | OCR-conditioned correction helps some collections and harms Maghribi/student handwriting. Some rows use different aggregation/mitigation conditions. Useful competing direction; script, preprocessing and OCR-prior quality differ from our setting. |
| Ari Vesalainen, Eetu Mäkelä, Laura Ruotsalainen, Mikko Tolonen. [Error Patterns in Historical OCR: A Comparative Analysis of TrOCR and a Vision–Language Model](https://arxiv.org/html/2602.14524v1), February 2026 preprint. §§3–5. | Fine-tuned Qwen has lower aggregate error but a different real-word/glyph error profile than TrOCR on printed eighteenth-century English. Unequal models/pretraining do not isolate architecture; released-code references include a placeholder. Error proxies motivate inspection, not causal attribution. |
| Shuo Chen et al. [True Multimodal In-Context Learning Needs Attention to the Visual Context](https://arxiv.org/html/2507.15807v2), COLM 2025 ([proceedings](https://openreview.net/pdf?id=n4JdyBGu6T)). §§3–4 and model/task tables. | DARA trains attention-balancing parameters to improve tasks requiring visual-context mappings. Its artificial reasoning/concept tasks differ from historical transcription; this is an additional trained intervention, not inference-only prompting. |

Additional leads assessed by Research: [Bhandari and Harit, Devanagari post-OCR ICL/retrieval](https://arxiv.org/abs/2609.21595)
(2026 preprint, methods/results read by Research) supports character retrieval as a possible comparator,
but is text correction of rendered news, not image HTR. [Teaching VLMs to Admit Uncertainty in OCR from
Lossy Visual Inputs](https://proceedings.iclr.cc/paper_files/paper/2026/hash/67b0e7c7c2a5780aeefe3b79caac106e-Abstract-Conference.html)
(Guan et al., ICLR 2026) was accessible at abstract level only; the paper copy was blocked. Its
uncertainty-tagging/RL idea is a future alternative, not a verified quantitative basis for this proposal.

**Relevance to this pilot:** image/transcription demonstrations followed by a query image, with random
contexts sampled per query, have direct precedent in Ayllon et al. This is a compatible setup, not an
exact replication or evidence that ICL will match LoRA. Existing evidence includes improvements and
failures; this initial experiment measures the outcome for our pinned model and bank.
