# Research results for readers

This folder holds short, permanent reports for the student and professor. The [project guide](../docs/PROJECT_GUIDE.md) tracks current progress; [experiments/](../experiments/) retains detailed run recipes, attempts and evidence.

| Report | What it establishes |
| --- | --- |
| [EXP-001–002](EXP-001-002.md) | First remote timeout, then baseline prediction and a failed fit; no completed round. |
| [EXP-003–005](EXP-003-005.md) | Training, reload and evaluation worked; joint OCR quality remained poor. |
| [EXP-006–008](EXP-006-008.md) | Qwen recipe quality gates failed despite stronger fitting and simpler box/text output. |
| [EXP-009](EXP-009.md) | LightOnOCR three-round random-selection page-text control. |
| [EXP-010](EXP-010.md) | Qwen three-round random-selection page-text control with the same page budget and shared hyperparameters. |
| [EXP-011](EXP-011.md) | Qwen k-center visual-diversity pilot against random; uncertainty and repeat-seed comparisons remain open. |

For **every full active-learning experiment**, add a reader report alongside its technical experiment record. Group closely related engineering attempts when one short report explains them more clearly. Explain the question and purpose; the baseline if measured; the round setup; what each metric means; **every completed round and equal-budget control**; the intuitive interpretation; what cannot yet be concluded; and links to technical provenance and raw evidence. State whether the run used validation or a final test. Keep unknowns explicit and never claim an active-learning advantage from a random-only control. Update the report when final numbers are verified, and link it from this index. No separate status log or copy of raw data is needed.
