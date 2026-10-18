# Research evidence

Research keeps one substantive note per question, such as `RES-001-uncertainty.md`, using
[TEMPLATE](TEMPLATE.md). Combine source notes, synthesis and recommendation unless their size
justifies a split. Progress and approval belong in local PROJECT; scientific content belongs here. Keep the current
synthesis distinct from dated historical proposals. After every substantive review, reconcile
PROJECT, AGENTS and this index using the shared state-sync rule; review completion does not
authorize an experiment. Complex reviews use Superpowers plus the scientific research skills.

Completed ICL studies: [EXP-015 paired examples](../results/EXP-015.md), [EXP-016 modality controls](../results/EXP-016.md), and [EXP-017 LightOn replication](../results/EXP-017.md).

Current findings, 18 October 2026: [EXP-021](../results/EXP-021.md) confirms the frozen 0/2/8-shot comparison on VAL42 (861 lines / 42 additional, previously examined validation pages). Ordinary CER 86.36/81.04/79.83% becomes 80.95/76.44/75.50% with query contrast. Two supports save another 763 edits; eight add 159 more. Contrast mainly reduces insertions; added supports reduce substitutions under contrast. This is an error-pattern hypothesis, not module causality. The common 855-line sensitivity retains gains, but zero-shot contrast introduces one runaway and exact-line counts decline. Historical mismatch contrast is only eight edits worse than query contrast; pilot reference superiority did not replicate. All 3,444 outputs / 5,166 line comparisons audited; both apps stopped, no new data/models/adapters. [Technical evidence](../experiments/EXP-021.md). Next proposal revised: EXP-016 already addresses modality controls; repeating them at two shots is deprioritized. Test whether the first changed contrastive token steers the final transcription, using fixed subsequent decoding and actual CER. At most 184 new continuations on a predeclared 105-query development subset; unapproved and not launched. The [novelty review](exp-018-contrastive-icl-analysis.md#post-exp-019-novelty-review-and-proposed-next-study) remains a dated source review; a new method, mechanism and independent transfer are unestablished. No further GPU work or final-test access is released; Base/chat/blank-image proposals remain deferred.

Previous linguistic review: [Linguistic context and language adaptation](linguistic-context-historical-ocr.md) — Manager + Research primary-literature review, new saved-output subgroup analysis, and a staged proposal separating language cues, transcript content and text-only versus paired adaptation. No new experiment; implementation awaits user review and approval.

Previous review: [Post EXP-015–017 analysis](post-exp015-017-analysis.md) preserves the pipeline audit and earlier fixed-bank LoRA + ICL proposal. Original approved ICL plan: [proposal](inference-adaptation-proposal.md), with [EXP-014 development/testing](../results/EXP-014.md).
