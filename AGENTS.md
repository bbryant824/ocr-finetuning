# FYP project instructions

## Purpose and current evidence

Study historical OCR with VLMs, currently focusing on inference-time ICL and contrastive
text decoding: how visual evidence and linguistic context affect recognition. Earlier
fine-tuning and active-learning studies remain adaptation baselines, not the current work queue.
READ2016 supplied line boxes locate crops; the model predicts text, not coordinates. TRAIN labels
may enter fitting or selected demonstrations; validation truth is evaluator-only and final TEST
stays isolated. Active learning acquires pages and source transcripts simulate annotation.

**Current milestone, 18 October 2026:** EXP-001–013 cover engineering/fitting/selection;
EXP-014–017 cover ICL development and modality/model controls; EXP-018–021 cover contrastive
recognition and diagnostics. Latest completed [EXP-021 reader report](results/EXP-021.md) and
[technical record](experiments/EXP-021.md) confirm query-contrast gains on VAL42. Recognition
remains poor; a causal visual/language mechanism, a novel method and independent transfer are
unestablished. The proposed token-choice continuation study is not approved or launched;
repeating EXP-016 modality controls at another shot count is not the next priority.
NUS SoC onboarding and a pinned-Qwen train/predict/reload check are verified in the
[cluster workflow record](docs/verification/nus-soc-cluster.md); this releases no new research run.

`.agent-local/PROJECT.md` is the concise local current state and approval/ownership
record; it is Git-ignored. The [project guide](docs/PROJECT_GUIDE.md) is the public architecture
and progress reference. The [results index](results/README.md) links every published experiment
to reader and technical records; the [research index](research/README.md) identifies the current
synthesis and historical proposals. Detailed results belong there, not in these instructions.
A clone without local PROJECT can use the public records; verify local state before dispatch.

## Required workflows

- Use the installed Ponytail plugin throughout the FYP: `engineering-suite-ponytail:entry-ponytail`
  and the applicable bundled workflow. Reuse tools/code, choose the smallest correct change,
  and avoid speculative abstractions, duplicate reports and unnecessary coordination.
- **Superpowers is mandatory for complex tasks**, for every role. Complex means substantive
  multi-file engineering, pipeline/method changes, nontrivial debugging, experiments or major
  research/architecture decisions. Start with `superpowers:using-superpowers`, announce the
  applicable skills and follow their relevant workflow: `brainstorming` for design,
  `writing-plans` / `executing-plans` for substantial implementation, `systematic-debugging` for
  failures, `test-driven-development` for code behavior changes, and
  `verification-before-completion` before success claims. Use review, worktree and branch-finishing
  skills when that scope needs them. Simple explanations and small documentation corrections
  do not require a full implementation workflow.
- Superpowers supplies rigor; Ponytail keeps the solution and process small. Reuse an existing
  plan, research note or technical experiment record for the design/checklist where practical.
  Do not generate parallel spec/status systems, force delegation, duplicate valid checks or
  automatically commit/push, create cloud jobs or alter scope. Honor approvals at the matching
  stage and do not ask again for authorization already provided. User instructions take precedence.
- **NUS SoC is the primary training/inference backend.** Use the project
  [nus-soc-cluster skill](.agents/skills/nus-soc-cluster/SKILL.md) and its linked full guide.
  Prefer the fastest suitable available cluster resource; recheck queues, actual GPU types,
  account limits and institutional rules each run. Modal remains an explicit fallback, not an
  automatic launch. Use normal SSH/Slurm in owner-authorized discrete sessions; no external
  scheduler, unattended job-management service or automatic retry. Keep the unchanged credential
  guide private/ignored and never expose secrets. Backend access/smoke success does not mean every
  existing Modal entrypoint is migrated. See the [workflow evidence](docs/verification/nus-soc-cluster.md).
  Every cluster experiment/check also needs a [command/session record](docs/cluster/sessions/README.md),
  updated while working: important actual commands, machine context, short explanations, selected
  outputs/job IDs, failures/fixes and artifact paths. Redact secrets and label reconstructions or
  shortened examples; omit noisy repeats. Link it from the EXP technical/check record.
- Research and Manager also use `scientific-brainstorming` for major scientific ideas/design,
  and `systematic-literature-review` for paper reviews. Verify primary sources and novelty;
  distinguish measurements, hypotheses and unknowns. Before proposing another experiment, check
  what earlier studies already answer and state what genuinely new evidence the proposal adds.
- Search first and load only relevant context. Test affected behavior first; broaden for a
  concrete risk. Paid GPU work requires a bounded approved run, not just an approved idea.

## Research integrity and artifacts

- Preserve original data, splits, failed runs and checkpoints. Record model/data/code/recipe
  identities, seeds, resources and measured versus estimated costs. Seed RNG before adapter
  construction; Trainer's seed alone is insufficient. Compare acquisition strategies at equal
  budgets. Never infer scientific benefit from engineering PASS or tune on final TEST.
- Establish the untouched original-model baseline for fitting or inference comparisons; reuse it
  only when revision, inputs, prompt, decoding, slice and metric match. Name the source EXP/receipt and say reused.
  Distinguish raw line scores from older page/cap-masked scores and keep harmful outputs visible.
- Every experiment, including testing, ICL/decoding and follow-ups, needs a concise
  `results/EXP-ID.md` explanation and separate `experiments/EXP-ID.md` technical evidence.
  Group tightly related early attempts if useful. A minor follow-up may stay under its existing
  EXP with explicit conditions/attempts. A literature-only study belongs in `research/`, not a
  fictitious EXP. Preserve labels, failures and provenance rather than overwriting results.
- Mirror every newly trained model/adapter/checkpoint under ignored `.local/models/exp-ID/`;
  verify saved hashes against receipts, retain remote originals and record exact paths/revision.
  Reference reused artifacts instead of copying them. Say explicitly when no model was trained.
  Never commit secrets, private labels, large assets/checkpoints or raw runtime output.
- Keep selection, oracle, evaluation and execution boundaries intact. Existing page-text,
  known-box, ICL and contrastive runners are mapped in the guide; reuse them before adding a
  pipeline. Earlier joint box/text execution is historical and LLaMA-Factory is optional/deferred.

## State synchronization — required before handoff

After every experiment/follow-up or substantive research review, and whenever direction,
status or approval changes, reconcile these existing records in the same handoff:

1. Update the detailed EXP reader/technical records or research note first, using verified
   evidence. Preserve completed, failed, interrupted, proposed and superseded states distinctly.
2. Refresh PROJECT with a brief current summary, relevant EXP/research links, owner, next action,
   approval limits, artifact locations and Git/cloud evidence status. Replace stale current text;
   do not append an endless history. Preserve unique ignored history before shortening it.
3. Refresh this file's current milestone/direction and links; keep permanent rules here and full
   metrics in the EXP records. Update results/research indexes, and the guide when a milestone,
   architecture or entrypoint changes. Historical notes stay dated and point to current direction.
4. Check linked IDs/files, completion status, metric/split/reuse labels, artifact paths and next
   approval against the source records. Documentation validation does not prove OCR/GPU behavior.

## Agent use — Protocol 3

Manager is the default worker. Use a specialist only when expertise or independent review adds
value; most tasks use Manager alone or Manager plus one specialist. Keep one owner, one scoped
assignment and one concise handoff; no ACK loops, status spam, routine polling or all-role routing.
The seven existing tasks remain idle without a release. QA retains standing authority for requested
teaching/decks. Do not overwrite another worker's edits or infer approval from stale task headers.

Read this file and current PROJECT first; consult [the protocol](coordination/README.md), TEAM,
role Skills and task records only when needed. Before dispatch to an older worktree, Manager
includes the current canonical policy/state path or identity; old checkout instructions are not
current project state. A file edit does not wake a specialist. Manager owns PROJECT/TEAM/releases;
specialists own assigned outputs. No global plugin settings or worker models change implicitly.

At completion explain purpose, setup, verification, key results/limits, reader/technical reports,
source/prepared data, raw receipts and trained artifacts (or none), current stage and next step.
Distinguish public reports, ignored artifacts and actual commit/push state. Obtain explicit approval
for destructive data/resource changes, result overwrites, force pushes, credentials, unexpected
material spend or methodology/split/final-evaluation changes; ordinary authorized work proceeds.
