# EXP-018: interpreting contrastive in-context handwriting recognition

Review date: 17 October 2026. **Completed analysis of independently audited EXP-018; follow-up proposal only.** Primary papers were checked on 16 October.

## Question and current conclusion

Does contrasting correctly paired and mismatched demonstrations improve recognition beyond ordinary paired or transcript-only context, with the model, labelled bank and query images fixed?

**Measured result:** on 861 validation lines, contrastive-eight reaches **75.5429% CER**, improving on paired-eight by **4.2902 percentage points** and transcript-only-eight by **5.0536 points**. The gain observed in the pilot carries over to disjoint pages under the unchanged recipe. Insertion/deletion changes account for most of the net gain; substitution errors increase slightly against transcript-only support. The result supports useful inference-time reweighting of this model's sequence preferences, not a demonstrated improvement in glyph recognition or a separation of linguistic and visual knowledge. Harmful changes remain frequent, absolute error remains high, and the saved LoRA24 reference is substantially stronger.

VAL8 is Seite0355–0362; VAL42 is Seite0363–0404. They have zero page overlap: the latter is 861 additional queries, not an extension containing the 182 pilot queries. Both belong to official validation; VAL42 has already been examined in earlier experiments. Neither is an untouched final test or evidence of independence between documents/writers. This is confirmation across these pages, not repeated-bank/context robustness, fine-tuning parity or reduced annotation effort.

## Fixed comparison and evidence scope

This is German historical handwriting from READ2016v1.2.0, with source line boxes supplied. It measures line recognition, not text detection or end-to-end page OCR. Original `Qwen/Qwen3-VL-4B-Instruct` weights at revision `ebb281ec70b05090aa6165b016eac8ec08e71b17` remain frozen. The labelled bank is the EXP-013 seed-824 selection: 24 TRAIN pages, 589 lines. Each query receives the same deterministic eight-support draw used by the corresponding controls. No validation transcription enters generation or support selection.

Let `z+` and `z−` denote next-token raw logits under paired and mismatched support respectively, at the **same generated answer prefix**. The implemented score is `zC = z+ + (z+ − z−)`, with α=1 and β=0.1 positive-branch plausibility restriction. Repetition processing is applied once to the combined scores; native branch positions and caches remain separate. The mismatch cyclically rotates the same eight images while keeping transcripts/order fixed. Both branches retain the real query image. This perturbs demonstration associations; it does not remove query vision or supply a pure language-model distribution.

RGB crops, padding 8, longest edge 1024, native instruction, BF16, greedy decoding, repetition penalty 1.1 and maximum 64 new tokens remain fixed. Primary scoring is `line-text-nfc-raw-v1`: retain every generated string, sum per-line edit counts, then divide by reference characters/words. WER can exceed 100% because insertions count. Historical page scoring that empties capped outputs is unsuitable as the headline comparison.

The [development record, §7](icl-decoding-and-adaptation-scope.md#7-completed-bounded-development--12-october-2026) documents execution at `2942cdd`, recipe identity `f88f3f6d120162b962c5293fef7ccfe9b2728af07b64d7d1c053704282f543a1`, Torch 2.8.0+cu128 and Transformers 5.6.0. Its saved audit confirms all 182 outputs, three native-equivalence contrastive replays, six query pairs and final lens calibration. This review inspected those receipts; it did not rerun inference or claim a fresh GPU verification.

## Full EXP-018 result and what improved

The frozen full execution uses source `764f9c4` and recipe identity `0889e2e55c22456059faabd48c7e423dc1b0ed23f1c0c82d45ff025a9a4a8ff9`. All rows share 861 lines, 16,899 reference characters and 2,675 words. Contrastive predictions are new. Zero-shot is reused from EXP-012, paired-eight from EXP-015, transcript-only/mismatched from EXP-016, and LoRA24 from EXP-013. Original weights mean zero pages used for fitting in this study; contextual conditions still consume the labelled TRAIN bank.

| Condition | Character edits | CER (%) | Word edits | WER (%) | Exact lines |
|---|---:|---:|---:|---:|---:|
| Zero-shot, reused | 14,594 | 86.3601 | 3,142 | 117.4579 | 19 |
| Paired-eight, reused | 13,491 | 79.8331 | 2,999 | 112.1121 | 18 |
| Transcript-only-eight, reused | 13,620 | 80.5965 | 3,023 | 113.0093 | 17 |
| Mismatched-eight, reused | 16,373 | 96.8874 | 3,459 | 129.3084 | 18 |
| **Contrastive-eight, new** | **12,766** | **75.5429** | **2,895** | **108.2243** | **18** |
| Saved LoRA24, reused | 7,275 | 43.0499 | 2,233 | 83.4766 | 19 |

| Comparator | Character edits saved | CER gain (points) | Lines improved / harmed / tied | Pages improved / harmed / tied | Previously correct lines damaged |
|---|---:|---:|---|---|---:|
| Paired-eight | 725 | 4.2902 | 324 / 182 / 355 | 37 / 4 / 1 | 0 |
| Transcript-only-eight | 854 | 5.0536 | 413 / 236 / 212 | 37 / 4 / 1 | 0 |

The gain is spread across pages but is not uniform. Because contrastive and paired each have 18 exact lines and no paired-correct line is damaged, contrastive produces **no additional exact line versus paired**. This does not negate reduced error on partially correct lines. LoRA24 still saves another 5,491 character edits relative to contrastive, a 32.4930-point CER gap. It uses all 589 bank lines during fitting, while each ICL query sees eight; the labelled-bank comparison is informative without implying equal exposure or compute.

The saved audit's single optimal Levenshtein alignment gives:

| Condition | Substitutions | Deletions | Insertions | Total output characters |
|---|---:|---:|---:|---:|
| Paired-eight | 9,402 | 1,482 | 2,607 | 18,024 |
| Transcript-only-eight | 9,243 | 2,109 | 2,268 | 17,058 |
| Contrastive-eight | 9,300 | 1,537 | 1,929 | 17,291 |

Against paired support, **678 fewer insertions** dominate the 725-edit gain: 102 fewer substitutions are partly offset by 55 more deletions. Against transcript-only support, **572 fewer deletions and 339 fewer insertions** outweigh 57 more substitutions. These are alignment-dependent operation counts, not a human error taxonomy or proof of particular perceptual mistakes. Multiple optimal alignments can distribute operations differently.

This rejects three simple descriptions of the aggregate gain. It is not solely removal of capped-output failures: 605/725 paired and 775/854 transcript-only edits are saved outside the old limit-hit lines. It is not uniform shortening: contrastive is shorter than paired in aggregate but **longer than transcript-only**. It is not principally correction of previously Cyrillic-containing output: savings on those lines are 0 and 8 respectively, with 3 contrastive Cyrillic outputs versus 2 paired and 4 transcript-only. Nor do fewer overall errors establish better glyph substitutions: the transcript-only comparison has more substitutions under this alignment. Query-sensitive completion, repetition/omission control and more generic sequence regularization remain competing explanations.

All 861 contrastive outputs end in EOS, with zero empty outputs and zero actual length stops. Positive/negative input lengths agree and range from 1,052 to 1,826 tokens. The independent saved audit covers all 14 chunks, token decodes, common-prefix/cache traces, originating calls, split/image isolation and frozen hashes. This review additionally rehashed all 20 audit-listed JSON files and independently recomputed all six conditions' character/word edit totals, denominators, output lengths and exact-line counts using Python dynamic programming. No new inference was run.

The pilot/full contrastive CERs are 76.6879% and 75.5429%; gains over transcript-only are 1.8089 and 5.0536 points. These descriptive differences reflect different query pages, not a learning curve or a larger-sample treatment effect. Do not pool the slices into a newly claimed final test.

## Development pilot: gain and harm together

All rows use the same VAL8 references: 3,925 characters and 614 words. Zero/paired controls are reused from EXP-015's penalty-1.1 pilot; transcript-only/mismatched controls are reused from EXP-016. “Transcript-only” describes the **supports**; the query image is still present.

| Condition | Character edits | CER (%) | WER (%) |
|---|---:|---:|---:|
| Zero-shot, reused | 3,574 | 91.0573 | 121.9870 |
| Paired-eight, reused | 3,154 | 80.3567 | 117.5896 |
| Transcript-only-eight, reused | 3,081 | 78.4968 | 114.4951 |
| Mismatched-eight, reused | 3,850 | 98.0892 | 137.6221 |
| Contrastive-eight, newly executed | 3,010 | 76.6879 | 111.5635 |

| Comparator | Character edits saved | CER improvement (percentage points) | Lines improved / worsened / tied | Previously correct lines damaged |
|---|---:|---:|---|---:|
| Paired-eight | 144 | 3.6688 | 78 / 40 / 64 | 0 |
| Transcript-only-eight | 71 | 1.8089 | 83 / 56 / 43 | 1 |

Paired, transcript-only and contrastive outputs have zero length caps; contrastive also has zero empty outputs, and every sequence ends in EOS. The gain over these controls therefore is not explained by removing capped outputs. Shorter repetitions, insertions, substitutions or different EOS decisions could still contribute; aggregate CER does not separate these mechanisms. Absolute recognition quality remains poor.

Read-only recomputation from saved predictions and the hash-verified source manifest reproduces all three edit totals. Contrastive improves all eight pages against paired support and seven against transcript-only support. The exception is Seite0360: 415 versus 350 transcript-only character edits, a loss of 65. The other seven pages jointly save 136 edits, leaving the net gain of 71. These are descriptive page comparisons, not eight independent document trials. Total normalized prediction lengths are 4,216 characters for contrastive, 4,412 for paired and 3,985 for transcript-only; the gain over transcript-only is accompanied by longer aggregate output, rather than universal shortening.

The LoRA24 CER of 43.0499% belongs to **VAL42**, so it is a matched reference in the full table above, not in this pilot table. [EXP-016](../results/EXP-016.md), [post-EXP-015–017 review](post-exp015-017-analysis.md).

## What the diagnostic establishes

All six preselected query swaps changed the transcript. Four outputs moved closer to the replacement image's reference; two tied. All anchor transcripts remained imperfect. This demonstrates an effect of the changed query pixels in these cases. It does not establish correct reading, population-level grounding or the mechanism of the contrastive gain.

The 24 lens contexts use exact saved **transcript-only-eight output token IDs**, for up to three answer positions. They do not use reference transcripts or reconstructed paired-output tokens. At the first answer position the pre-block readouts are identical; differences appear after the first decoder block. Final log-probability differences for the original run's selected first token range from 0.8465 to 7.9537 nats under original versus replacement images. These are selection-conditioned query-swap effects, not image-removal gain or a percentage attributable to vision.

A maximum final-readout calibration error of 0.0 validates that final projection against native logits. It does not establish semantic faithfulness of intermediate projections. Late separation cannot establish that vision recognized the correct text before language overrode it. The paired and mismatched branches have separate hidden trajectories; their score combination does not create a single contrastive hidden trajectory.

## Focused primary-source review

Checked on 16 October 2026: the four requested papers, their version identities, and one directly relevant recent handwriting study. Searches included paper titles and `OCR "contrastive decoding" "in-context"` on arXiv/ACL Anthology. Included sources that change the method comparison, interpretation or novelty boundary; unrelated contrastive training and open-domain retrieval studies were excluded. This is a focused review, not an exhaustive novelty search. Different tasks, models, normalizations and aggregation units prevent pooling reported effects.

| Source and inspected scope | Evidence that matters here |
|---|---|
| **Peng et al., [Enhancing Input-Label Mapping in In-Context Learning with Contrastive Decoding](https://aclanthology.org/2025.acl-short.77/)**. ACL 2025; [arXiv v2, 7 June 2025](https://arxiv.org/html/2502.13738v2), §§2–5. | ICCD contrasts original demonstrations with inputs drawn from a different label class while keeping demonstration labels fixed. Its evaluation concerns seven NLU classification tasks and text LMs, not image transcription. Llama-3.1-8B random-context average accuracy is 77.6→79.4; a no-demonstration negative scores 73.0 and changed-label negative 77.3. Thus negative construction matters. Our same-multiset image rotation is an adaptation, not an exact replication. Use the published paper/current v2, not an earlier preprint's headline gain. |
| **Leng et al., [Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding](https://arxiv.org/pdf/2311.16922)**. CVPR 2024; arXiv v1, 28 November 2023, method and evaluation. | VCD contrasts clean and noise-distorted **query images**, with a positive plausibility constraint. Its main evidence concerns object hallucination and general vision-language evaluation. EXP-018 instead preserves the query and disrupts **support associations**. Neither distorted images nor mismatched support form a pure language baseline. The shared score family is prior art; the intervention and target task differ. |
| **Karamolegkou et al., [Reading or Guessing? Visual Grounding Failures of Vision-Language Models for OCR in Ancient Greek and Arabic Editions](https://arxiv.org/pdf/2605.27750v2)**. v2, 7 September 2026; main results and intervention/image-gain appendices. | Historical **print**, not German handwriting. The PDF's title includes Arabic even though some metadata omit it. Query-noise VCD slightly worsens Greek and improves Arabic Qwen CER; image-free M3ID has the reverse direction. Effects use method-matched baselines and median CER, not our corpus-weighted score. Their teacher-forced generated-token image-gain analysis also shows that erroneous lexical choices can remain image-dependent, particularly for Greek Qwen3-VL-8B. Accordingly, neither output fluency nor query dependence diagnoses correct grounding; OCR contrastive decoding already has direct precedent. |
| **Krojer et al., [LatentLens: Revealing Highly Interpretable Visual Tokens in LLMs](https://arxiv.org/html/2602.00462v5)**. v5, 22 July 2026; methods, rendered-text case study and limitations. | Contextual textual nearest neighbours make visual tokens more interpretable than a direct vocabulary lens in their studied systems. In a rendered-text example, LogitLens can yield a plausible continuation rather than the visible word. The authors explicitly leave downstream causal effects unestablished. This cautions against interpreting visual-patch projections literally; our answer-position lens addresses a different readout. Neither validates intermediate Qwen3-VL-4B states as a causal visual/language decomposition. |
| **Ayllon, Gandia and Calvo-Zaragoza, [Exploring In-Context Learning for Handwritten Text Recognition](https://arxiv.org/html/2609.37195v1)**. v1, 29 September 2026; setup and per-model in-domain results. | Direct recent precedent for frozen-VLM handwriting ICL and sensitivity to context selection. On ICFHR16, Qwen3-VL-8B average CER is 80.95% at zero examples, 81.96% at eight and 78.78% at sixteen. Thus average gains are not automatic even where favourable contexts exist. Oracle minimum error across sampled contexts is not a deployable label-free selector. This supports preserving matched draws and reporting conditional findings; it is not a replication of our model/protocol. |

## Interpretation and claim boundaries

**Supported by both slices:** the fixed support-context contrast improves aggregate recognition beyond paired and transcript-only support on these pages. The pilot independently shows query-pixel effects in six controlled examples. These facts do not establish that the improvements occur because of those query effects. Existing checks support correct execution of the specified decoder within their tested scope.

**Competing hypotheses, not established mechanisms:** the contrast may amplify query-specific evidence about what to continue or omit, suppress preferences induced by conflicting supports, or regularize output length/repetition without improving visual interpretation. These can coexist. The insertion/deletion findings make sequence behaviour a stronger immediate target than a glyph-substitution account. They do not identify a purely linguistic mechanism: visual errors can also insert or omit text. Subtracting a poorly performing distribution does not isolate newly learned mappings, and useful language priors may help ambiguous ink. One bank, one support draw per query and one negative permutation cannot establish robustness to context choice.

**Potential contribution:** a controlled empirical test of mismatched-support contrast for this fixed-bank historical-handwriting setup, with explicit transcript-only and adapted-model references and preserved harmful outcomes. Generic contrastive OCR, ICCD, frozen-VLM handwriting ICL and lens visualizations are prior art. The focused search does not establish that the exact combination is first. No annotation-saving or active-learning strategy conclusion follows from reusing one random labelled bank.

**Efficiency remains unmeasured comparatively.** The full run records 1,623.02 seconds of query work and 9.06 GiB maximum allocated GPU memory on L4 with Torch 2.8.0+cu128 / Transformers 5.6.0. The saved app-state receipt records stopped/zero tasks at 00:04:18 SGT on 17 October, after creation at 23:10:54 on 16 October. App lifetime and summed query-work time are different quantities. The query timer includes input preparation after model/image loading, whereas historical native timers exclude preparation. These do not establish relative speed, billing cost, cost-effectiveness or memory advantage. Two decoding branches add work, but no exact runtime multiplier is inferred. No new model or adapter was produced.

## Full-slice preparation and comparison review

Read-only inspection of execution commit `764f9c4` and frozen full recipe `0889e2e55c22456059faabd48c7e423dc1b0ed23f1c0c82d45ff025a9a4a8ff9` found no scientific blocker in the extension reviewed. Preparation selects the established 42 pages/861 lines, checks TRAIN-only bank records against source truth, excludes pilot overlap, verifies ordered control coverage and support IDs, and rejects equal-transcript accidental mismatches. The adapted reference additionally checks selected bank pages, 589 training lines and the saved adapter weight hash. All five referenced file hashes match the frozen record. This preserves a matched recognition comparison; it does not make training exposure or compute equal.

Independent AST comparison confirms all five modeling/worker functions match both `2942cdd` and the reused gate's hashes. Pilot/full model-file hashes, configuration, helper checks and readiness inputs agree. The readiness record explicitly identifies reuse, rather than presenting a fresh three-query execution. Query payloads contain IDs/geometry without transcripts; offline scoring receives truth. The imported configuration still names historical page evaluation and training settings, but the executed spec/receipt identifies raw line scoring and the inference path performs no fitting. Interpret executed paths and the primary metric, not unused inherited fields.

**Reporting qualification:** full-slice zero-shot and paired-eight receipts each contain two 64-token outputs with `truncated=true`, but no exact output token IDs or stop reason. Their EOS-at-limit status cannot be reconstructed from stripped text. Transcript-only and mismatched controls preserve exact tokens and confirm one and two length stops respectively. Preserve the older counts as “outputs at the 64-token limit,” with genuine stop reason unknown; all strings remain included in CER/WER. This limits failure-mode comparison, not the edit-count comparison. The final independent audit confirms complete full-run coverage and the metrics above.

## Single next question: query evidence in improvements versus harms

**Recommendation: C, a small teacher-forced query-image intervention study, subject to separate authorization.** Ask: **Are the continuation preferences associated with contrastive improvements over transcript-only output more query-specific than those associated with harm, or are they compatible with largely query-insensitive sequence regularization?** The full gain justifies retaining the small diagnostic, while its insertion/deletion composition changes the hypothesis and gate: assess continuation and stopping choices, not a claim of better glyph reading. This connects controlled query effects to scored outcomes; sensitivity alone is not success. Its contribution would be a bounded empirical association, not a new lens or a separation into pure linguistic and visual components.

| Alternative | What it tests and why it is not the default next step |
|---|---|
| **A: one alternative fixed derangement** | Tests dependence on the chosen wrong associations while preserving image/text multisets and image count. This is the cleanest narrow robustness check. Changed image order can still alter positional interactions, and two permutations cannot establish general robustness. It does not explain why transcript-only supports rival paired ones or whether an improvement uses correct query evidence. Prefer it if the immediate objective becomes replication of the decoder gain rather than explanation. |
| **B: transcript-only-support negative, real query retained** | Tests whether amplifying paired-versus-transcript-only score differences improves decoding without introducing wrong support images. Keep α, β, positive branch and queries fixed. However, removing eight images changes image count, prompt length, positions and contextual interactions; it is not subtraction of a pure language model. A win would support a useful alternative contrast, not prove visual grounding. Prefer it if the objective changes to improving recognition and the existing mismatch branch is the practical concern. |
| **C: observed-prefix query intervention** | Holds each support context and generated prefix fixed while replacing only query pixels with a matched-grid real crop. Compare paired, mismatched and transcript-only contexts at the same answer decision. This tests whether their preferences depend on the particular query and whether that dependence differs between net improvements and harms. It cannot prove that a specific glyph was recognized correctly or eliminate interactions between support images and positional layout. Its advantage is answering the attribution question directly enough to reject simple explanations without launching another decoding method. |

These choices follow the distinctions in the primary-source table: ICCD motivates association contrasts; VCD changes query evidence; Reading or Guessing separates image dependence from correctness; LatentLens limits semantic claims from intermediate readouts. Neither A nor B alone identifies the cause of a gain. C is also a conditional diagnostic, not a complete mechanistic explanation.

### Smallest proposed study, not authorized or executed

Use existing exact contrastive and transcript-only token sequences. At their **first differing token**, take the shared generated prefix and two observed candidates, **including EOS when one candidate stops**. This revises the earlier non-EOS-only proposal to reflect the observed sequence-length findings without adding evaluations. Do not retokenize stripped text or teacher-force reference text. Classify lines as improved or harmed by whole-line raw character edits; a net line improvement does **not** label its first differing token as correct. Attach each line's output-length difference and alignment-based substitution/deletion/insertion changes for interpretation. Reference text is used only offline. The first divergence is an early decision probe: it need not coincide with the edits responsible for the final gain, and a null effect there cannot rule out later query-dependent stopping or repetition control.

For each eligible line, score both candidates under paired (`P`), mismatched (`M`) and transcript-only (`T`) supports with (i) the original query and (ii) one deterministically chosen real donor crop from another page with identical processor-grid geometry. Freeze support IDs/order, prompt tokens and generated prefix. Verify that changing the query preserves token IDs, image count and native position/cache geometry **within each context**. The three support contexts need not have identical positions to one another; compare within-context query effects and disclose that their differences can include context-layout interactions. A donor image is a controlled replacement, not absence of vision.

Let `m_s(I)` be the raw-logit margin of the contrastive candidate over the transcript-only candidate under context `s`, image `I` and their shared prefix. Define `D_s = m_s(original) − m_s(donor)`, `D_C = 2D_P − D_M`, and `G = D_C − D_T`. `G` describes excess query sensitivity of the raw contrastive margin; it is not an attributable amount of visual knowledge. Contrastive reweighting also changes score scale. Therefore record vocabulary ranks of both candidates under `zP`, `zM`, `zT` and `2zP−zM`; use the analogous query-induced change in rank advantage as a scale-insensitive cross-check, computed from the same evaluations. Larger logit differences alone cannot pass the interpretation gate.

Concretely, rank advantage is `rank(transcript-only candidate) − rank(contrastive candidate)`, where rank is one plus the number of strictly higher vocabulary scores. Compute its original-minus-donor change for the composite and transcript-only vectors, then subtract the latter from the former. Rank the composite vector directly; do not combine component ranks linearly.

Keep raw margins separate from actual processed decoding scores. Reproduce original-image next-token choices for transcript-only and contrastive decoding at the common prefix; record repetition processing, β=0.1 positive plausibility membership and EOS. A query swap can change the admissible candidate set. Report mask toggles separately from preference changes when both candidates remain admissible; do not treat an excluded token's score as a finite probability. If the result is carried by the mask, interpret it as a constraint-mediated effect rather than evidence that raw contrast isolated visual information. Native final readouts suffice; no layer sweep is needed.

**Selection caveat:** both candidates were chosen under the original image. Positive original-image preference is partly selection-conditioned, so `G>0` alone is not evidence of correct grounding. Report every margin, both outcome groups, original/donor conditions and within-context effects. Compare the improved-versus-harmed pattern with prefix length, output-length change, edit-operation changes and original margins visible. If the pattern occurs only on shorter outputs, limit the explanation to that subset and compare its edit savings with those on longer outputs. Aggregate output length alone cannot locate the source of the net gain. Differences remain conditional associations; oracle-defined groups are not an inference-time detector of trustworthy output.

**Feasibility checked without inference:** pilot tokens provide 83 improved and 56 harmed lines, including two and one EOS-difference cases. Position-zero differences occur in 40/83 improved versus 17/56 harmed cases. Full tokens provide 413 improved and 236 harmed lines, including six and three EOS-difference cases; position-zero counts are 199 and 93. This position imbalance could confound a naive group comparison. Older paired outputs need not be reconstructed because both candidates/prefixes come from contrastive and transcript-only runs. Matched-grid donor feasibility still requires preparation; token availability alone does not establish it.

**Pilot:** select eight improved and eight harmed VAL8 lines by a frozen hash order, with four position-zero and four later-position cases in each group, capping each group at two lines per page. Reserve one later-position slot per group for an EOS-difference case, if a matched-grid donor exists. Fill other slots in the frozen order and disclose unavailable EOS coverage. Donor selection uses geometry and IDs, not probe scores. If the basic balanced sample lacks enough matched-grid cases, report the shortfall instead of changing criteria after inspecting readouts. Six context/image evaluations per line give **96 teacher-forced prefix evaluations**. Ranks, masks and edit summaries add no model passes. No new free decoding, fit or labels; this is a workload count, not a monetary estimate.

**Proposed advance gate:** all prefix/grid and original-choice checks pass; final readouts agree with native logits within the existing 0.001 tolerance. As an exploratory screen, require positive `G` on at least three of four improved cases and a higher improved-than-harmed median `G` **within each position stratum**, with the same direction under the rank-based cross-check. Also inspect the non-EOS, both-candidates-admissible cases separately: a finding supported only by score amplification, EOS choices, mask toggles, a few extreme output lengths or one page does not justify expanding this early-preference hypothesis. Report such a narrower mechanism or unresolved confound instead. These are proposed screening rules, not statistical significance or a power calculation. Preserve negative results; no tuning of thresholds or case selection after seeing probe scores.

**Conditional confirmation:** only after that gate, freeze the same analysis before viewing new probe scores; use 32 improved and 32 harmed VAL42 lines, half position-zero and half later-position within each group, capped at two per group per page, retaining one EOS-difference slot per group where feasible. This requires **384 additional prefix evaluations**. Report all distributions, length/edit summaries and page-grouped results. The improved-versus-harmed distinction must recur in both position strata under margin and rank readouts, without being carried by one page, output-length outliers or only masked decisions; otherwise the pilot explanation is not confirmed. This balanced sample estimates a conditional contrast, not population prevalence or an OCR benchmark. Both slices remain previously examined validation; a replicated association is not final-test evidence.

**Falsification and decision:** the full recognition-gain prerequisite is now met. If improvements and harms have similar query effects, or the distinction disappears under rank/length/mask checks, the proposed early query-specific preference account is unsupported. That does not prove a pure language mechanism or exclude later sequence effects. Stop before confirmation and report what remains unresolved. A becomes relevant only if the priority changes to permutation robustness; B only if it changes to testing a new decoder. Missing matched-grid cases also block C as specified; do not silently substitute image deletion. No follow-up experiment, model family, parameter search or broader matrix is authorized here.

## Reproducibility records reviewed

Public development method/config: [scope §7](icl-decoding-and-adaptation-scope.md#7-completed-bounded-development--12-october-2026), [pilot recipe](../experiments/recipes/read2016-known-box-icl-contrastive-pilot.json). Prior controls: [EXP-016](../results/EXP-016.md), [EXP-017](../results/EXP-017.md). The latter establishes failure of a transferred Qwen protocol on LightOnOCR, not a universal handwriting limitation.

Private, ignored pilot receipts inspected under `.local/verification/contrastive-icl-pilot-2026-10-12/`: `recipe.json`, `summary.json`, `audit.json`. The audit reports independent counts of 3,010 character edits and 685 word edits. Its recorded SHA-256 values distinguish file bytes from the logical frozen recipe identity:

- `recipe.json`: `dbff1f08abe78d335385b61c6bad2050ebd899162edd83e48f4e0f9a855d1c52`.
- `summary.json`: `7bb72b51bc0f94902eebe4f5e2b38a95df0796e69af426fe690e7b2ca29527ed`.
- `pilot/results.json`: `1666f1660c6e36e53334c771e10841f3d143a48c4f456323b127a64652030b3d`.

Full execution/configuration: [EXP-018 technical record](../experiments/EXP-018.md), [full recipe](../experiments/recipes/read2016-known-box-icl-contrastive.json). Source `764f9c4`; logical recipe identity `0889e2e55c22456059faabd48c7e423dc1b0ed23f1c0c82d45ff025a9a4a8ff9`. Raw records remain ignored under `.local/verification/exp-018/icl-exp018-contrastive-seed824/`. Reviewed file-byte SHA-256 values:

| Record | SHA-256 |
|---|---|
| `recipe.json` | `a7f70e986d61d40e3ea8ac3c5a80c81aa8331175d5a473c3c9a941f6e2ffa879` |
| `summary.json` | `67f159cdc4793c2d55266b714226ff03184a1ac86905c7cba55188a908b25df9` |
| `audit.json` | `f90d59ff1795b11194302f8476135696c35d2b4b0352490636a18f075be712ee` |
| `app-state.json` | `697315c888d5884f6e457ab46b78f14309c28bffd25c841c6166c992e4bc1143` |
| `full/results.json` | `38ebdec2d0fba866027459084e4c2933b61f07d04082e695f25085345c03b07c` |

The independent audit uses Python dynamic-programming edit distances for totals and a recorded optimal Levenshtein edit-operation alignment for descriptive S/D/I counts. Review verification rehashed these files and reproduced all six conditions' totals without executing the writer in `audit.py`, changing raw evidence or making cloud calls. Only this research note was authored; the proposed intervention study remains unexecuted.
