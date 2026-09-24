# FYP project instructions

## Purpose and current evidence

Study whether active-learning strategies reduce annotation needed to adapt full-page OCR to
historical documents. The acquisition unit is a page image; existing source transcriptions simulate
human annotation. The recognition target is ordered full-page text; original source boxes
remain only to verify frozen dataset provenance. Reveal labels only for selected TRAIN
pages. Validation truth belongs only to the evaluator; keep final test isolated. There is no labeling frontend in the current workflow.

[Project guide](docs/PROJECT_GUIDE.md) is the maintained architecture and operating reference.
[EXP-009](experiments/EXP-009.md) records the LightOnOCR page-text control, and
[EXP-010](experiments/EXP-010.md) records the Qwen control, each with the untouched
original-model baseline and three random 64-page cumulative fits. The shared settings are in
[the page-text recipe](experiments/recipes/read2016-page-text.json). This supports a working
recognition learning curve, but does not establish line-box detection, strategy advantage,
document-independent generalization or human-time saving. [EXP-005](experiments/EXP-005.md)
and [EXP-008](experiments/EXP-008.md) preserve the earlier LLaMA-Factory/Qwen joint-OCR
engineering run and quality failure. Current assignments and resource decisions are in ignored
`.agent-local/PROJECT.md`.

## Research and engineering rules

- Label FACT, MEASUREMENT, INFERENCE, ESTIMATE and UNKNOWN accurately. Preserve original data,
  splits, checkpoints and failed runs. Record source/data/model/recipe identities, seeds and costs.
- Compare strategies only at equal annotation budgets with documented controls and leakage checks.
  Do not tune on final test or infer scientific benefit from one engineering run.
- Use the installed Ponytail plugin for this entire FYP. Manager and specialists should follow
  `engineering-suite-ponytail:entry-ponytail` and its `ponytail` workflow when available. Do only
  what is needed now; reuse existing code and tools before adding dependencies or coordination.
  Choose the smallest correct change without sacrificing research integrity, data isolation,
  recovery or required approval.
- Search first; read only relevant files/sections and reuse unchanged evidence. Test affected
  behavior first; broaden only for a concrete risk. Paid GPU work needs a bounded approved run.
- For every full active-learning run, publish a concise reader-facing `results/EXP-ID.md`
  with purpose, metric meanings, all rounds, interpretation, limits and a link to the separate
  technical `experiments/EXP-ID.md` evidence. Follow [results/README.md](results/README.md).
- Store every newly trained model/adapter/checkpoint in a verified, Git-ignored
  `.local/models/exp-ID/` mirror; retain its remote original when used. Check the saved weight
  hash against the run receipt and record exact local/remote paths, base-model identity and hashes
  in `experiments/EXP-ID.md`. Reference reused earlier artifacts instead of copying them again.
- Never commit secrets, raw private labels, large assets/checkpoints or generated runtime output.
  Put concise reproducible methods/results in the existing guide, plan or experiment record.
- Preserve the small boundary: `active_learning.py` selects, `LocalOracle` reveals selected
  TRAIN labels and `evaluation.py` scores. `recognition_pipeline.py` coordinates current page-text
  runs with Hugging Face Trainer/PEFT on Modal. `pipeline.py` retains the frozen source and fixture
  lifecycle. Earlier joint-OCR execution is archived in Git and the experiment records.

## Agent use — Protocol 3

At the end of each completed request, report work and verification since that prompt. For
long-running research or pipeline work, give a concise self-contained handoff: purpose, setup,
key results and limits; exact locations of reader and technical reports, source/prepared data,
raw receipts and trained models/adapters/checkpoints (say when none); current research stage
and next step. Distinguish Git-tracked reports from ignored local or remote artifacts. For small
tasks, state briefly when no experiment, data or model changed.

**Manager is the default worker** for questions, documentation, small fixes and ordinary
engineering. Delegate only when a specialist clearly adds value: Development for sustained or
risky coding, Testing for meaningful independent review, Research/Planning for substantive
methodology uncertainty, Platform for real cloud/data operations, QA for deep teaching or decks.
Most tasks need Manager alone or Manager plus one specialist. Never route work through every role.
Keep one owner, one complete assignment and one concise final handoff; no ACK/status loops or
routine polling. Optimize total token/tool/test cost, not just one agent's context.

Read this file, the current `.agent-local/PROJECT.md`, and only the relevant source/evidence.
Read [the protocol](coordination/README.md), TEAM, a role Skill and assigned task only when
coordination is needed. Shared `.agent-local` is ignored local memory, not Git history. The
seven existing Codex tasks remain available but idle without a current release. QA may create
requested reports/slides under its standing authorization without a numbered implementation task.

Manager owns local PROJECT/TEAM and any task releases; a specialist owns its assigned result.
Do not overwrite another worker's active edits. Use existing Codex tasks and native messages for
assignments; a file edit alone does not wake a specialist. Significant or risky changes receive
independent review when it adds value. Technical PASS and OCR/scientific benefit are distinct.

Routine authorized work proceeds without another permission request. Obtain explicit approval
before destructive data/checkpoint/resource deletion, result overwrites, force-pushing, credential
changes, material unexpected spend, or changing ground truth, splits, final evaluation or the
research objective. Record current decisions in PROJECT and permanent method/result decisions
in the relevant public plan or experiment record. Preserve historical local evidence in the
existing private archive. Keep this system understandable by one student.
