# Research evidence

Research keeps one substantive note per question, such as `RES-001-uncertainty.md`, using
[TEMPLATE](TEMPLATE.md). Combine source notes, synthesis and recommendation unless their size
justifies a split. No scientific output was established during setup. Progress belongs in the
coordination task; scientific content belongs here. Planning turns recommendations into plans.

Completed ICL studies: [EXP-015 paired examples](../results/EXP-015.md), [EXP-016 modality controls](../results/EXP-016.md), and [EXP-017 LightOn replication](../results/EXP-017.md).

Current review, 12 October 2026: [Contrastive ICL and visual versus linguistic evidence](icl-decoding-and-adaptation-scope.md) — Manager + Research primary-literature reassessment, including historical-OCR grounding and lens limitations. Proposes one fixed eight-shot association-contrastive VAL8 pilot plus a small query-image/lens diagnostic. Earlier adaptation and Base/chat stages are deferred. Bounded development is complete:17 focused checks, actual L4 native gate,182-line pilot and six-pair lens audit. Contrastive CER76.69% versus paired80.36% and transcripts-only78.50% on VAL8. Full validation and EXP-018 remain pending review.

Previous linguistic review: [Linguistic context and language adaptation](linguistic-context-historical-ocr.md) — Manager + Research primary-literature review, new saved-output subgroup analysis, and a staged proposal separating language cues, transcript content and text-only versus paired adaptation. No new experiment; implementation awaits user review and approval.

Previous review: [Post EXP-015–017 analysis](post-exp015-017-analysis.md) preserves the pipeline audit and earlier fixed-bank LoRA + ICL proposal. Original approved ICL plan: [proposal](inference-adaptation-proposal.md), with [EXP-014 development/testing](../results/EXP-014.md).
