# FYP project instructions

## Purpose and current evidence

Study whether active-learning strategies reduce annotation needed to adapt full-page OCR to
historical documents. The acquisition unit is a page image; original line boxes and text simulate
human annotation. Reveal labels only for selected TRAIN pages. Validation truth belongs only to
the evaluator; keep final test isolated. There is no labeling frontend in the current workflow.

[Project guide](docs/PROJECT_GUIDE.md) is the maintained architecture and operating reference.
[EXP-005](experiments/EXP-005.md) independently verified a complete 16-initial + 8-acquired
LLaMA-Factory/Modal engineering round. OCR readiness failed (final 1/8 valid, CER 0.999512, box
F1 zero). No active-learning benefit or human-time saving has been established. Real acquisition
currently supports random only. Current assignments and budget state are in ignored
`.agent-local/PROJECT.md`; historical experiment records stay in Git.

## Research and engineering rules

- Label FACT, MEASUREMENT, INFERENCE, ESTIMATE and UNKNOWN accurately. Preserve original data,
  splits, checkpoints and failed runs. Record source/data/model/recipe identities, seeds and costs.
- Compare strategies only at equal annotation budgets with documented controls and leakage checks.
  Do not tune on final test or infer scientific benefit from one engineering run.
- Prefer existing tools and the smallest correct change. Avoid speculative abstraction, duplicate
  wrappers, extra coordination or tests that merely repeat implementation.
- Search first; read only relevant files/sections and reuse unchanged evidence. Test affected
  behavior first; broaden only for a concrete risk. Paid GPU work needs a bounded approved run.
- Never commit secrets, raw private labels, large assets/checkpoints or generated runtime output.
  Put concise reproducible methods/results in the existing guide, plan or experiment record.
- Preserve the small boundary: `active_learning.py` selects, `pipeline.py` coordinates,
  `LocalOracle` reveals selected TRAIN labels, `evaluation.py` scores, LLaMA-Factory owns model
  training/inference, and Modal executes the reviewed worker.

## Agent use — Protocol 3

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
