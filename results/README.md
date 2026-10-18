# Research results for readers

This folder holds short, permanent reports for the student and professor. The [project guide](../docs/PROJECT_GUIDE.md) tracks current progress; [experiments/](../experiments/) retains detailed run recipes, attempts and evidence.

| Reader report | What it establishes | Technical record(s) |
| --- | --- | --- |
| [EXP-001–002](EXP-001-002.md) | First remote timeout, then baseline prediction and a failed fit; no completed round. | [001](../experiments/EXP-001.md), [002](../experiments/EXP-002.md) |
| [EXP-003–005](EXP-003-005.md) | Training, reload and evaluation worked; joint OCR quality remained poor. | [003](../experiments/EXP-003.md), [004](../experiments/EXP-004.md), [005](../experiments/EXP-005.md) |
| [EXP-006–008](EXP-006-008.md) | Qwen recipe quality gates failed despite stronger fitting and simpler box/text output. | [006](../experiments/EXP-006.md), [007](../experiments/EXP-007.md), [008](../experiments/EXP-008.md) |
| [EXP-009](EXP-009.md) | LightOnOCR three-round random-selection page-text control. | [009](../experiments/EXP-009.md) |
| [EXP-010](EXP-010.md) | Qwen three-round random-selection page-text control with the same page budget and shared hyperparameters. | [010](../experiments/EXP-010.md) |
| [EXP-011](EXP-011.md) | Completed three-seed Qwen k-center, entropy and least-confidence comparisons against matched random controls. | [011](../experiments/EXP-011.md) |
| [EXP-012](EXP-012.md) | Qwen known-box line recognition: one completed random-seed 0/64/128/192-page baseline. | [012](../experiments/EXP-012.md) |
| [EXP-013](EXP-013.md) | Completed: strategy-first known-box random/k-center/entropy/confidence pilot at 24/48/72 pages. | [013](../experiments/EXP-013.md) |
| [EXP-014](EXP-014.md) | Completed development/testing: random 0/8/24-shot OCR ICL passed local and small L4 checks. | [014](../experiments/EXP-014.md) |
| [EXP-015](EXP-015.md) | Completed full 8/24-shot random ICL: modest gains over zero-shot, mixed shot-count effect, both behind the saved 24-page LoRA control. VAL8 decoder follow-up retained penalty 1.1 after raw-output/cap checks. | [015](../experiments/EXP-015.md) |
| [EXP-016](EXP-016.md) | Completed Qwen OCR content controls: transcript-only ICL retained modest gains, mismatched pairs harmed recognition, and the saved LoRA24 remained stronger. | [016](../experiments/EXP-016.md) |
| [EXP-017](EXP-017.md) | Completed LightOnOCR replication of paired/text-only/mismatched OCR ICL: all six ICL conditions worsened raw recognition under the fixed Qwen protocol; outputs and comparisons independently audited. | [017](../experiments/EXP-017.md) |
| [EXP-018](EXP-018.md) | Completed contrastive ICL plus query-image and correctness diagnostics: free-running CER improves; clean-prefix text correctness stays flat while terminal choices improve slightly. No established glyph-recognition mechanism; saved LoRA remains stronger. | [018](../experiments/EXP-018.md) |
| [EXP-019](EXP-019.md) | Completed stopping/text ablation: contrast for text retains 140/144 saved edits on VAL8; stopping-only does not reproduce the gain. All outputs and conditional decisions audited; no new model. | [019](../experiments/EXP-019.md) |
| [EXP-020](EXP-020.md) | Reference and0/2/8-shot exploration complete: query-contrast CER77.07/74.47/73.99%; two supports add benefit beyond zero-shot, eight add little. Long-output sensitivity reported;1,092 new outputs audited across both phases; no new model. | [020](../experiments/EXP-020.md) |
| [EXP-021](EXP-021.md) | Completed VAL42 confirmation: contrast CER 80.95/76.44/75.50% at 0/2/8 shots; two supports add 4.52 points beyond zero-shot contrast. All 3,444 new outputs audited; query-noise and prior mismatch references nearly tie; failures and sensitivity retained. | [021](../experiments/EXP-021.md) |

For **every experiment**, including development checks, ICL/decoding and related follow-ups, keep
reader and technical evidence linked here. Group closely related early attempts when appropriate.
Explain purpose, setup, metric meanings, all completed conditions/budgets, baseline reuse, results,
interpretation, limits and exact artifact locations. Include failures and say validation versus final
test. A small follow-up can update its existing EXP; do not invent a new run for a literature review.

Before handoff follow [AGENTS state synchronization](../AGENTS.md#state-synchronization--required-before-handoff):
refresh local PROJECT, AGENTS' brief milestone, this index and relevant research/guide sections.
Keep measured results in their source records and historical proposals dated; no parallel status log.
