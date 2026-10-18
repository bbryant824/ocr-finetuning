# Linguistic context and language adaptation in historical VLM OCR

> Historical post-EXP-015–017 analysis/proposal; its dated evidence is preserved below. For the current
> direction and approval state after EXP-021, use [research index](README.md) and
> [current synthesis](exp-018-contrastive-icl-analysis.md). Older proposed work is not a release.

**Post EXP-015–017 research analysis · 5 October 2026**  
Owner: Manager, collaborating with the existing Research agent.  
Status: verified literature and saved-output analysis; proposed experiments have **not** run.

## 1. Recommendation and research purpose

Our next question should be **what kinds of linguistic information help a VLM transcribe historical handwriting, and when that information overrides the source image**. EXP-016 makes this question worth investigating: textual demonstrations help almost as much as image/transcript demonstrations, but neither approaches the benefit of domain fine-tuning.

Start with a small diagnostic separating **explicit language identification, transcript vocabulary/spelling and ordered text**. If language adaptation remains the chosen direction, follow with a matched **text-only versus image/transcript adaptation** study. Predicted neighbouring text is a promising subsequent inference-time alternative. These are staged proposals for review, not approval to run all three.

This develops the earlier [post-EXP-015–017 review](post-exp015-017-analysis.md). Its LoRA-plus-ICL proposal remains an alternative; it is not an agreed next experiment. Active selection has not consistently beaten random in our pilots. That does not invalidate adaptation: it shifts attention from selecting more pages toward understanding how the model uses the information already available.

**Potential contribution:** a controlled account of language conditioning and adaptation in difficult, small-budget historical handwriting recognition, including the improvements and newly introduced errors. We cannot claim that text-only adaptation, OCR correction, contextual HTR or linguistic priors are new inventions.

## 2. What our experiments actually establish

### Fixed local setup

- READ2016 supplied text-line locations are cropped outside the model. The task is **known-location recognition**, not detection. Each inference predicts one crop's text.
- Qwen3-VL-4B-Instruct, verified revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`.
- The bank is the fixed seed-824 random selection from EXP-013: **24 TRAIN pages, 589 lines, 12,293 NFC characters and 1,940 words**. It is a small text corpus.
- Each query uses a reproducible random draw of 8 or 24 bank examples. The eight are a prefix of the 24; matched controls retain support IDs and ordering.
- Paired ICL: TRAIN crop plus transcript; transcript-only ICL: transcript without demonstration image. **Both still include the real query image.** Mismatched ICL rotates demonstration images while keeping transcripts and order.
- Fresh chats, native processing, RGB crops with 8-pixel padding, longest edge 1,024; greedy BF16 decoding, 64 new tokens and repetition penalty 1.1. The original instruction is `Transcribe only the text in this cropped line. Return only the transcription.`
- VAL42 has **861 lines, 16,899 reference characters and 2,675 reference words**. Raw line-level NFC corpus CER/WER retains every saved prediction, including capped outputs. Validation truth is used only for scoring and retrospective analysis.

See [EXP-016 reader report](../results/EXP-016.md), [technical record](../experiments/EXP-016.md), [ICL control specification](../experiments/recipes/read2016-known-box-icl-controls.json) and [ICL implementation](../src/active_ocr/icl.py). The source, input bundle, actual recipes and reused-output hashes are preserved in the original receipts.

### Results on the same validation queries

These are **raw line metrics**, recomputed in EXP-016 and independently checked again for this review. They differ from the older joined-page metric that replaced capped predictions with empty text. CER/WER can exceed 100% when insertions dominate; percentages below are error rates, not accuracy.

| Condition | Raw CER | Raw WER | Saved cap flags | Lines with Cyrillic |
| --- | ---: | ---: | ---: | ---: |
| Original Qwen, no examples; reused EXP-012 | 86.360% | 117.458% | 2 | 164 |
| Paired 8; reused EXP-015 | 79.833% | 112.112% | 2 | 2 |
| Paired 24; reused EXP-015 | 80.892% | 112.224% | 3 | 0 |
| Transcript-only 8; EXP-016 | 80.596% | 113.009% | 1 | 4 |
| Transcript-only 24; EXP-016 | 79.809% | 109.869% | 0 | 1 |
| Mismatched 8; EXP-016 | 96.887% | 129.308% | 2 | 1 |
| Mismatched 24; EXP-016 | 94.870% | 126.355% | 5 | 0 |
| Saved random LoRA on 24 pages; reused EXP-013 | 43.050% | 83.477% | 0 | 0 |

Paired 8 versus transcript-only 8 differs by 0.76 CER points; transcript-only 24 beats paired 24 by 1.08 points. Paired 8 and transcript-only 24 differ by just **four character edits** over the corpus. This is closeness in one pilot, not statistical equivalence or evidence that demonstration images can never help.

The mismatch result is informative: **expected-script output can improve while character recognition worsens**. EXP-017's LightOnOCR failure under the transferred multi-turn Qwen protocol is preserved, but is not a calibrated native handwriting benchmark or a general conclusion about VLM language ability.

### New offline decomposition: where the improvement occurs

Manager independently recomputed per-line Levenshtein distances for all eight conditions, checked identical coverage, source/recipe/bank hashes and the provenance of reused receipts. Define a descriptive subgroup as the **164 lines whose frozen zero-page prediction contains a U+0400–U+04FF Cyrillic character**. None of the reference labels contains Cyrillic. The other subgroup has 697 lines.

| Condition | Total character edits | Edits on the 164 baseline-Cyrillic lines | Edits on the other 697 lines | Lines improved / worsened / tied versus zero |
| --- | ---: | ---: | ---: | ---: |
| Zero | 14,594 | 2,930 | 11,664 | 0 / 0 / 861 |
| Paired 8 | 13,491 | 2,406 | 11,085 | 422 / 287 / 152 |
| Paired 24 | 13,670 | 2,488 | 11,182 | 418 / 306 / 137 |
| Transcript-only 8 | 13,620 | 2,464 | 11,156 | 381 / 304 / 176 |
| Transcript-only 24 | 13,487 | 2,421 | 11,066 | 384 / 316 / 161 |
| Mismatched 8 | 16,373 | 2,772 | 13,601 | 296 / 425 / 140 |
| Mismatched 24 | 16,032 | 3,098 | 12,934 | 262 / 465 / 134 |
| LoRA 24 | 7,275 | 1,350 | 5,925 | 774 / 30 / 57 |

Transcript-only 24 saves 1,107 edits: **509, or 46.0%, occur in the baseline-Cyrillic subgroup**, and 598 elsewhere. Thus the improvement is not confined to wrong-script cases. Nevertheless, that subgroup still has **86.4% CER** after transcript-only context, despite Cyrillic almost disappearing. Text-only 24 worsens **316 of 861 lines** even while improving the corpus total. LoRA improves substantially more lines and saves most of its edits outside the wrong-script subgroup.

**Interpretation limit:** subgroup membership is based on a baseline prediction and may correlate with difficulty. These counts locate changes; they do not causally attribute 46% of the gain to script correction, establish grammar learning or provide an inference-time routing rule.

### What fine-tuning changed is still an open question

The historical LoRA used `all-linear`. Inspection of its saved safetensors header finds **208 visual and 504 language-component adapter tensors**. It is not a decoder-only adapter. This checks parameter membership, not which component caused the gain.

The model saw all 589 supervised crop/transcript pairs during training; an ICL query sees only 8 or 24 supports. The bank is matched, but per-query exposure and computation are not. The result cannot be described as a clean contest between visual learning and language learning. A newly controlled comparison is needed to investigate that distinction.

## 3. Focused primary-literature review

Search/access date: **5 October 2026**. Manager and Research used the installed literature-review and scientific-brainstorming skills. This is a focused review of **19 primary studies**, including rechecking three ICL foundations, not an exhaustive systematic review or pooled meta-analysis.

Representative searches combined `historical OCR language model post correction`, `VLM historical handwriting linguistic priors`, `handwritten recognition metadata context`, `OCR neighbouring lines shuffled prior`, `text-only domain adaptation handwriting` and paper-title searches. Sources were ACL Anthology, arXiv, CEUR, CVF and publisher full texts. Included work informs language conditioning, correction, adaptation, context or visual-versus-linguistic controls. Product claims and secondary summaries were not treated as scientific evidence. Relevant methods/results/limitations were read; not every appendix was exhaustively inspected. Read versions are identified below; later-version equivalence is not assumed.

### A. Linguistic information can help, but its form and match matter

| Primary source / status / inspected scope | Setup, finding, limitation and lesson |
| --- | --- |
| **Meyer, Guichard, Soullard, Coquenet, Gravier & Coüasnon (2026), [n-gram injection into transformers for dynamic language model adaptation in handwritten text recognition](https://arxiv.org/html/2603.03930v1).** arXiv v1 preprint; §§4–6, Tables 4–6. | A 2.1M word recognizer is trained to consume projected character-five-gram distributions, then switches its external language model at inference. IAM/RIMES use separate target-text banks; target test answers are not the bank. Table 4 IAM k-means target CER is **18.0→10.1%** with injection; ordinary rescoring reaches **9.7%**, combination **8.4%**. Training is required to enable injection; it is not a Qwen plugin. Constructed modern word-level shifts and extra text resources differ from READ. Supports domain-language adaptation, not guaranteed superiority of architectural injection. |
| **Arjafellah, Yousfi & Hadmi (2026), [A controlled study of CTC, attention, and hybrid models for Arabic handwritten text-line recognition](https://link.springer.com/article/10.1007/s44443-026-00970-6).** Journal of King Saud University Computer and Information Sciences; §§3.5–5. | Holds encoder architecture/training protocol fixed while comparing decoders. A TRAIN-only character six-gram is combined with CTC via shallow fusion; CER improves over the strongest neural baseline **8.77→7.76% on KHATT**, **13.11→11.67% on Muharaf**. Shows that local character statistics help without sentence-level discourse. It is not VLM prompting, and common remaining errors suggest limits to decoder changes alone. |
| **Kiessling & Clérice (2024), [Does Context Matter? Enhancing Handwritten Text Recognition with Metadata in Historical Manuscripts](https://ceur-ws.org/Vol-3834/paper30.pdf).** CHR proceedings; architecture/data, Tables 5–7. | Conformer HTR on CATMuS uses language, script and century metadata. Micro character accuracy is **90.32% baseline, 91.14% all-context, 91.13% when context is zeroed at inference**. Much of the gain survives removing inference metadata; do not present this as strong evidence that a language prompt alone helps. It studies trained metadata use, not an off-the-shelf VLM. |
| **Levchenko (2025), [Evaluating LLMs for Historical Document OCR: A Methodological Framework for Digital Humanities](https://aclanthology.org/2025.lm4dh-1.7.pdf).** LM4DH/RANLP proceedings; §§3–6, Tables 2–5. | Historical Russian print, full pages/lines/three-line windows, prompt-language and metadata variations. Effects differ by model; added context does not universally help. Measures both archaic-character preservation and inappropriate archaic insertion. The prompts change multiple factors and input geometry varies. Historical cues can induce an incorrect convention as well as fix one. |

The shallow-fusion study explicitly scores candidate text with a combination of recognizer and language-model log probabilities. This is a controlled way of varying a language prior. Qwen's next-token distribution mixes visual, linguistic and instruction information internally; we cannot infer equivalent components simply from its CER.

### B. Correction needs useful evidence; fluency is not sufficient

| Primary source / status / inspected scope | Setup, finding, limitation and lesson |
| --- | --- |
| **Boros, Ehrmann, Romanello, Najem-Meyer & Kaplan (2024), [Post-Correction of Historical Text Transcripts with Large Language Models: An Exploratory Study](https://aclanthology.org/2024.latechclfl-1.14.pdf).** LaTeCH-CLfL proceedings; §§3–5 and prompt/evaluation definitions. | Fourteen LLMs, eight heterogeneous text-transcription benchmarks, five prompt designs and zero/three-example correction without images. Most configurations harm recognition; examples usually do not fix it. Output cleaning/length filtering and a relative edit-similarity metric prevent direct comparison with our raw CER. This is a useful negative result, not a verdict on all modern VLM correction. |
| **Kanerva, Ledins, Käpyaho & Ginter (2025), [OCR Error Post-Correction with LLMs in Historical Documents: No Free Lunches](https://aclanthology.org/2025.resourceful-1.8.pdf).** RESOURCEFUL proceedings; §§3–6, Tables 2–4. | Historical English/Finnish text-only correction; roughly 200/100-word segments, decoding/output alignment controls. GPT-4o's Finnish **relative CER improvement is +11.9% with normalization but −10.1% without**. Language, initial OCR and normalization affect the outcome. Printed corpora and longer coherent segments are different from our short German handwritten crops. Primary scoring must preserve the source's transcription conventions. |
| **Danilova & Aangenendt (2025), [Post-OCR Correction of Historical German Periodicals using LLMs](https://aclanthology.org/2025.resourceful-1.26.pdf).** RESOURCEFUL proceedings; §§4–5, Tables 1–5. | Fine-tunes German Llama/Mistral on noisy-text/reference pairs. Test has 376 paragraphs, 207 initially perfect. Best Mistral reduces WER **0.09→0.07**, while Table 5/body CER stays **0.02**; some variants harm performance. Abstract wording about CER conflicts with the body. Clean twentieth-century print differs from READ. Measure damage to already correct input, not only mean improvement. |
| **Farazi et al. (2026), [When Do VLMs Help Arabic Manuscript OCR? A Cross-Dataset Study](https://arxiv.org/html/2608.22366v1).** Under-review arXiv v1; §§4–5.3, Tables 6–8. | Qwen2.5-VL-7B receives an image plus Tesseract draft. Muharaf main results report normalized CER **60.6%**, versus **69.2%** Tesseract and **70.4%** standalone Qwen. Shuffled-prior/image-withheld controls are worse. Some lines worsen despite corpus gains. A subset table repeats whole-test baseline values, so exact denominators need verification before recomputing contrasts. Supports conditional recoverability; does not establish a universal quality threshold. |
| **Gutteridge, Jackson, Kukurin & Dong (2026 read version), [Judge a Book by Its Cover: Investigating Multi-Modal LLMs for Multi-Page Handwritten Document Transcription](https://arxiv.org/html/2502.20295v3).** arXiv v3; venue unverified; §§3–5/Table 2. | Supplies whole-document noisy OCR plus one chosen page image. Bentham/Gemini CER is **8.67%** with each page image and **8.48%** with the selected-image method. This combines relevant text, document grouping and changed visual input; it is not a clean discourse ablation or random labelled ICL. Our document IDs are unknown, so exact adoption would need new grouping evidence. |
| **Greif, Griesshaber & Greif (2025 identifier), [Multimodal LLMs for OCR, OCR Post-Correction, and Named Entity Recognition in Historical Documents](https://arxiv.org/html/2504.00414v1).** arXiv v1 preprint; §§3–6/Table 2. | Thirty German printed directory pages; source image plus Transkribus text conditions Gemini/GPT-4o. Normalized and stricter scores are distinct. Their own-model self-correction was not notably helpful in pretests; successful correction uses another recognizer's draft. Small selected data and prompt refinement limit transfer. A second pass is not automatically useful. |
| **Humphries et al. (2024), [Unlocking the Archives: Large Language Models Achieve State-of-the-Art Performance on the Transcription of Handwritten Historical Documents](https://arxiv.org/html/2411.03340v1).** arXiv v1 preprint; data/evaluation/correction, Tables 3–7. | Fifty English manuscript pages, 33 hands, repeated runs; correction sees images and prior transcripts. Cross-model correction helps more than self-correction. Strict scoring differs from a modified score forgiving some modernization/case/punctuation. Private data and frontier APIs limit reproduction and comparability with 4B German-line recognition. |
| **Bourne (2024), [Scrambled text: training Language Models to correct OCR errors using synthetic data](https://arxiv.org/html/2409.19735v1).** arXiv v1 preprint; §§2–4. | LoRA text correction uses generated nineteenth-century-style English text and learned character corruption, then evaluates newspaper OCR. Noise distribution and training-example length matter. Synthetic corrupted text is not real handwriting recognition; best configurations selected across many runs remain exploratory. Training on text to repair drafts is a different task from teaching Qwen to read its image. |

### C. Language can override the visible form

| Primary source / status / inspected scope | Setup, finding, limitation and lesson |
| --- | --- |
| **Vesalainen, Mäkelä, Ruotsalainen & Tolonen (2026), [Error Patterns in Historical OCR: A Comparative Analysis of TrOCR and a Vision–Language Model](https://arxiv.org/html/2602.14524v1).** arXiv v1 preprint; §§3–6, Tables 1–6. | Fine-tunes TrOCR and Qwen2.5-VL-7B on approximately 195,000 historical English printed lines; evaluates presegmented lines. Qwen is more accurate overall but selectively modernizes historical spellings and makes plausible lexical substitutions. Scale, pretraining and resizing differ, so architecture alone is not isolated. Code/split links include `github.com/tbd`, limiting independent verification. Useful error taxonomy, not an expected READ performance level. |
| **Liang et al. (2026), [Visual Merit or Linguistic Crutch? A Close Look at DeepSeek-OCR](https://arxiv.org/html/2601.03714v2).** arXiv v2 preprint; §§2–4, Tables 1–5. | Renders natural, perturbed and random text at different compression levels. Random-text precision is **19.84% in Tiny versus 61.70% in Base**: the headline collapse is mode-dependent. Linguistic regularity and visual load interact. Synthetic print is a diagnostic, not evidence that correct natural-text predictions are hallucinations or that READ has the same mechanism. |
| **Yao, Liao, Zhang, Li & Zhao (2026), [PAR: Training-Free Positional Perturbation and Attention Recycling for Faithful OCR](https://aclanthology.org/2026.acl-long.1065.pdf).** ACL proceedings; §§2–4/Table 3/Appendix C. | Synthetic altered familiar quotes/poems expose repair-to-canonical behaviour; positional and attention interventions include Qwen3-VL. English uses substantial clean-text padding. Layer choices require calibration; training-free does not mean implementation-free. Relevant prior art for linguistic overcorrection, but rendered familiar text differs from our short, unfamiliar manuscript lines. |
| **Cao et al. (2026), [HisDoc-OCR: Restoring Visual Grounding in MLLMs for Chinese Historical Document OCR](https://aclanthology.org/2026.findings-acl.301.pdf).** Findings of ACL; §§3–4/Tables 2–4. | Freezes vision components and adapts the language component with layout delimiters, first-occurrence-character loss weighting and attention distillation. Large historical Chinese data and five-epoch training; ablations test the package. Shows that changing a decoder can improve visual grounding, so decoder adaptation is not synonymous with grammar learning. This is substantial prior art and engineering beyond our first diagnostic. |

### D. Direct relationship to our ICL observations

| Primary source / status / inspected scope | Setup, finding, limitation and lesson |
| --- | --- |
| **Chen et al., [Can Multimodal Large Language Models Truly Perform Multimodal In-Context Learning?](https://arxiv.org/html/2311.18021v2).** WACV 2025; read arXiv v2, §§3–5. | VQA/captioning with demonstration-image removal, text corruption and context selection. Demonstration text often contributes strongly while query-image evidence remains essential. Older models/tasks limit mechanistic transfer. Our text-only condition therefore needs language-content controls rather than an automatic claim that Qwen ignores vision. |
| **Bertini Baldassini et al., [What Makes Multimodal In-Context Learning Work?](https://arxiv.org/html/2404.15736v2).** CVPR Workshops 2024; read v2, §§3–5. | Classification/VQA/captioning vary retrieval, image/text content and ordering. Pairing, copying and recency are task-dependent. Whole-example order permutation tests recency; it does not test whether disconnected random lines form useful discourse. |
| **Ayllon, Gandia & Calvo-Zaragoza (2026), [Exploring In-Context Learning for Handwritten Text Recognition](https://arxiv.org/html/2609.37195v1).** arXiv v1 preprint; methods and READ tables. | Qwen8B, TRAIN support and repeated context draws. Average READ CER is **80.95% zero, 81.96% eight shots, 78.78% sixteen**. Ground-truth-best-context scores are oracle analyses, not deployable selectors. Model, splits/export and decoding differ from ours. Modest/variable READ ICL gains are compatible with our results; they do not show a local pipeline fault. |

### Evidence synthesis

The literature supports **conditional benefits**, not a single positive story. Local character distributions can help even short words. Coherent neighbouring text can supply information that random examples do not. A draft must contain useful anchors for correction, and better-looking language can still damage literal transcription. Metadata effects can arise through training rather than the inference cue itself.

What remains unestablished locally is whether our gain comes primarily from identifying the language, learning historical vocabulary/conventions, ordered-text information, improved visual-to-text use, or decoder interactions. These explanations can coexist. Raw CER alone cannot separate them.

## 4. Hypotheses and worthwhile alternatives

| Question / hypothesis | How to challenge it | Assessment for this project |
| --- | --- | --- |
| **H1: examples mainly establish language/script and answer format.** | Compare explicit language cue with that cue plus transcripts; check CER as well as script counts. | Highest-priority cheap diagnostic; directly follows EXP-016. |
| **H2: transcript content supplies useful domain vocabulary/character patterns beyond the cue.** | Match intact and word-shuffled texts; later compare text-only adaptation with paired adaptation. | Strong fit to the linguistic research interest; failure of word ordering to matter does not exclude morphology or character-level priors. |
| **H3: relevant local text resolves ambiguity better than unrelated random text.** | Fixed first-pass neighbour predictions versus unrelated predictions of similar length; no references supplied. | Promising inference-time direction after the diagnostic. Avoid claiming true discourse when source reading order crosses columns or unrelated regions. |
| **H4: weight adaptation makes language information more useful.** | Compare the same conditions on base and one fixed adapter. | Existing hybrid proposal is useful as a control; it is not automatically the main next study. |
| **H5: linguistic support improves fluent output but damages source-specific forms.** | Count line wins/losses, initially correct content damaged, and fixed-sample spelling/name/number changes. | Necessary result interpretation, not a separate expensive faithfulness programme. |

**Alternatives considered:** generic self-correction is easy but already studied and can simply re-read or reinforce errors. Visual/text retrieval is established and would mix selection quality into the first language diagnostic. Synthetic rendered perturbations are useful later but cannot replace historical handwriting evidence. PAR-style attention hooks and new external-LM injection add architecture/calibration work before we know the behavioural problem. Larger random-shot sweeps do not answer which information is useful. A new model/dataset now would also make our comparisons less controlled.

## 5. Staged proposed study

### Stage 0 — reuse the verified evidence

Completed in this research turn: saved-output arithmetic, common query coverage, subgroup decomposition, bank size and historical adapter target membership. No GPU inference or new fitting. The previous independent source/context-flow audit remains valid for unchanged code; we have not re-established a changed pipeline because none was implemented.

### Stage 1 — language cue versus domain text

**RQ1: Does transcript context add useful information beyond explicitly identifying the source language, and does within-line word order matter?**

Start on the fixed **VAL8 / 182-line** screen with original Qwen and four conditions:

| Arm | Query / context | Comparison |
| --- | --- | --- |
| A | Existing instruction + real query crop; no support | Original baseline; reuse only an exact matching receipt. |
| B | A plus one frozen, accurate collection-language cue | B−A tests the cue. Do not simultaneously add several new prompting rules. |
| C | B plus 8 intact TRAIN transcripts in the existing chat format | C−B tests additional transcript content. |
| D | B plus those same 8 transcripts with words shuffled within each transcript | C−D tests sensitivity to local ordering with much lexical content preserved. |

Keep the existing seed-824 bank/draws. Use collection-level metadata only; do not infer a query's language from its reference, force German output for mixed-language visible text, or modernize spelling. The common transcription target stays unchanged. The old no-cue transcript-only result is a reference, not a matching substitute for C.

Define the shuffle deterministically, retaining words/punctuation and separators where practical. Report how many supports actually change; one-word/repeated-word lines may be unchanged. Word shuffling changes plausibility and can change tokenization, so it is not a pure grammar intervention. Log actual input lengths/token inventories and outputs. If whole-example recency is a concern after a positive effect, a small whole-example-order permutation is a separate diagnostic, not another mandatory initial arm.

Initial matrix is **728 predictions before reuse**; at most 546 are new if all 182 A outputs match. This is a screen, not a large-shot/model/seed sweep. Use a small correctness gate first to inspect the actual messages, images, selected IDs, loss-free inference, token counts, completion slicing and evaluator. Expand promising frozen conditions to VAL42 only after that gate and methodological review.

Possible outcomes: B close to C supports simple priming as an adequate explanation under this protocol. C/D both better than B supports useful content beyond the cue. C better than D suggests ordering adds value, but does not prove new grammar learning. All conditions remaining poor supports limits to these prompts, not the absence of linguistic ability in general.

### Stage 2 — compare domain text learning with paired adaptation

**RQ2: Can learning the same historical transcripts without their images recover part of the recognition benefit of supervised image/transcript adaptation?**

If this remains the preferred scientific question, compare unchanged base Qwen, **a new text-only domain adapter**, and **a newly matched image/transcript adapter**. Keep the historical EXP-013 adapter as a clearly labelled conditional reference, not the matched training control.

Use the same 589 TRAIN transcripts, transcript order, one-epoch target-text exposure and optimizer schedule. Start both new arms from identical seeded adapter initialization. Restrict the new adapters to the same explicitly enumerated language-component modules, freeze visual weights, and verify trainable parameter lists. This follows the principle of adapting the decoder while retaining visual inputs, without claiming to reproduce HisDoc-OCR's objective.

Text-only training should have a declared next-token/domain-language objective; paired training retains image-conditioned answer-only transcription loss. Match native text formatting and loss-bearing transcript tokens wherever possible; ignore instruction/image tokens in loss. Evaluate **all arms on the same real-crop OCR task with no demonstration context**. Verify tiny-fit learning, finite gradients, saved/reloaded adapters and actual target-token accounting before full evaluation. Use existing HF/PEFT/Modal L4 components; no new training framework is required by this proposal.

**Design decision before execution:** freeze the exact text-only objective/interface and eligible adapter modules. Missing-image transcription training is not automatically a valid domain-language objective. Similar chat formatting does not remove task mismatch. Equal target-token exposure also does not equal gradient information or visual compute. With only 1,940 words, memorization or loss of instruction behaviour is plausible; a null result would be specific to this small bank and protocol. No external text corpus or broader training sweep is approved.

A useful text-only gain would show that parameter adaptation from textual resources can improve subsequent image recognition. A paired advantage would demonstrate added value of image-conditioned supervision under the matched protocol. Neither cleanly assigns a percentage of success to an internal “vision” or “language” subsystem: a decoder can learn to interpret visual features. Generalizing to unseen writers/documents requires later independent evaluation.

### Stage 3 — relevant predicted context as a practical extension

**RQ3: After adaptation, does locally relevant predicted text help more than unrelated domain text?**

Use one fixed adapted recognizer. Freeze its first-pass predictions before constructing any second-pass input. Compare no context, neighbouring predicted lines from verified page reading order, and unrelated predicted lines of approximately matched length. Keep the target crop at the same resolution and retain the same instruction; return only the target line. Add reversed order only if actual neighbours help.

All neighbour text is **model output**, never validation labels. Record neighbour IDs and boundaries. Restrict initially to within-page source order; page-number adjacency does not establish document or writer identity, and multiple columns may weaken semantic adjacency. No recursive propagation of corrected text. This measures practical context usefulness without changing visual crop geometry.

If studying correction instead, compare frozen draft unchanged, query image alone under the same correction instruction, image plus draft, draft without image, and mismatched draft plus image on identical IDs. The image-only control distinguishes useful prior information from a better second-pass prompt. Reusing a LoRA draft does not by itself make same-model self-correction effective. This is an alternative design for review, not an additional automatic workload.

## 6. Fair comparison, measurements and stopping decisions

- Keep model revision, source images, crop processing, query coverage, target convention and scoring fixed. Changing a prompt is a declared experimental factor; it invalidates automatic reuse of old condition outputs.
- Preserve **raw NFC line CER/WER** and all predictions. Report actual EOS/length stops separately from legacy cap flags; retain capped strings in the metric. Do not select conditions with the old cap-as-empty score.
- Report script changes, output length, empties/echoes, per-line improvement/worsening and initially correct lines made wrong. Inspect a fixed reproducible sample of changed names, numerals, abbreviations and spellings; no modern spellchecker defines correctness. This supports the linguistic question without requiring a separate broad faithfulness benchmark.
- Repetition penalty 1.1 applies to prompt tokens by default in the decoder interface; transcript content therefore changes penalties as well as model conditioning. The [Transformers documentation](https://huggingface.co/docs/transformers/en/internal/generation_utils#transformers.RepetitionPenaltyLogitsProcessor) explains the scope. The previously approved 1.0 test worsened raw recognition, so a new unrestricted sweep is not justified. A narrowly scoped prompt-excluded-penalty control is relevant only if needed to interpret a promising lexical effect; check the frozen runtime API before proposing its implementation.
- There is one fixed acquisition bank, not multiple new seeds. If tiny differences drive a scientific claim, repeated matched context draws/permutations within that same bank can assess context variability; query-level randomness alone is not a replication estimate. Do not choose a best context using validation truth.
- Use the small screen to reject malformed or plainly harmful conditions. A **two absolute CER-point** improvement can be a predeclared practical screening threshold, not a significance test or literature constant. Script suppression alone is insufficient. Predeclare any permissible damage to initially correct lines and report it transparently.
- Already inspected validation remains development evidence. Choose and freeze the method before a separately approved final evaluation. Do not claim document-independent generalization when grouping is unknown. Equal page counts do not establish equal human annotation time; text resources, training and inference efficiency remain quantities to measure.

## 7. Evidence, artifacts and collaboration

**New Git-tracked outputs:** this research note and its link in [research index](README.md). No new EXP number is assigned: this is literature and retrospective analysis, not a new cloud experiment. Original reader/technical EXP-015–017 records and prior research notes remain intact.

**New ignored evidence:** `.local/verification/exp-016/linguistic-subgroup-review.py` and `.json`. The local-only script recomputes eight conditions, verifies 861 common query IDs, hashes input/provenance files, checks all raw CER/WER numerators/denominators and records bank/adapter-header facts. It writes only its own receipt. Reproduce from repository root:

```bash
python3 .local/verification/exp-016/linguistic-subgroup-review.py
```

**Existing evidence reused:** `.local/verification/exp-016/icl-exp016-controls-seed824/summary.json`, `recipe.json`, `bank.json` and original referenced EXP-012/013/015 receipts. Source run `781d319efaa94b68bdd8027a2ae4e01c`; prepared manifest SHA-256 `ac0a02a218a2142c6ee92ae964e7828ddfc65545dd6ca12f695815bb4493bdd6`; bank SHA-256 `b170d6bac83fd4caa5ec0fee57d358820175712e4695b5e39d55aeeb45cb0cd3`. Input bundle `94ed4f7f79d5d1110b7454c12a3b86d643cdb9dc0dece83d63734c2e4d87dcb6`. Frozen EXP-016 Modal image `im-jgZ7uRYWA9OT6Tl7sPh8iN`.

**Data unchanged:** `.local/assets/read2016-1.2.0/`, `.local/prepared/read2016-v1-8db5833/`, frozen source `.local/verification/active-learning-source/runs/`.

**Models unchanged; no new model or adapter:** original Qwen `.local/assets/qwen3-vl-4b-ebb281e/`; LightOn weights `.local/assets/lightonocr-2-1b-base-c1bc8eb/`; reused 24-page random adapter `.local/models/exp-013/random/seed-824/round-1/adapter/`, weight SHA-256 `7e875d37b7e85032729f2b44d0d9f619f343679752812af6899596e579f4cab6`. Remote originals remain in their original experiment storage.

Manager independently reviewed primary methods, current code and saved outputs; Research reviewed literature, interpretations/novelty and proposed controls. Existing Research task `01a0853d-58a6-7223-878f-0ac4d1c02163` supplied main handoff turn `01a10a9f-3cca-7fb2-a840-95640d1884bf` and focused supplements after Manager shared the subgroup findings and new NGI paper. No further specialist tasks, schedules or cloud work. Code inspection base: main `b5dc6cf403c10a933879fac875d83039205b0879`.

**Current stage:** engineering and exploratory adaptation/ICL pilots completed; linguistic mechanisms remain hypotheses. **Decision for the user:** whether to begin the Stage 1 language-content diagnostic, then pursue matched text-only adaptation or predicted local context. Implementation and experiment execution require explicit approval of that scope.
