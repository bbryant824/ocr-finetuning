# FYP project constitution

## Project mission

The project studies whether active-learning strategies reduce the human annotation needed
to adapt OCR to historical and low-resource documents. Page images are the acquisition unit;
the current research simulates annotation by revealing existing source ground truth only for
selected training pages. Ground truth contains the source's line boxes and transcriptions.
The practical objective is a small, reproducible, understandable pipeline with fair comparisons.

Prioritize the complete local active-learning loop with simple interchangeable dataset/model
interfaces. Real dataset adapters, OCR models, GPU servers and labeling frontends come later.
Keep implementations minimal, straightforward and correct; do not add speculative infrastructure.
The oracle must not expose unrevealed or held-out labels to fitting or acquisition. Revealed-page
counts represent simulated annotation budgets, not measured human annotation time. Fixture model
outputs verify plumbing only and never establish OCR quality or active-learning improvement.

The local Python pipeline exists. The configured model is `Qwen/Qwen3-VL-4B-Instruct`, but
model loading, prediction, fine-tuning and GPU execution remain placeholders. Corpus/language,
checkpoint revision and final experimental methodology are not yet established by repository evidence.

## Architecture summary

CURRENTLY IMPLEMENTED: one `Pipeline` orchestrator with a local simulation path; a normalized
JSONL source/true-label oracle; frozen images/source/config and run identity; cumulative reset-fit
model interface and deterministic fixture; random/least-confidence/entropy selection; atomic SQLite
round snapshots, resume and JSON/CSV exports; optional isolated validation evaluator; CLI/example.
Fixture scores and empty OCR predictions are not research findings. Resume assumes unchanged code,
dependencies and versioned adapters; code/environment changes are not automatically detected.

Simulation flow: frozen source → select train pages → reveal selected labels → fit cumulative
examples → predict remaining pool → optional isolated validation → commit round → next batch.
Only the evaluator receives validation truth; final-test evaluation is deferred. See docs/simulation.md.

Optional older code remains: JSONL/CSV image import with document-hash splits, Label Studio
payload/client, GPU HTTP client, polling controller, metrics and health endpoint. Qwen and GPU job
execution remain placeholders. Their remote image resolver is missing and they are not needed by
simulation. Runtime state stays ignored; simulation uses its own configured database/artifacts.
The configured Qwen model is not a limitation of the new model interface. Current priorities live
in shared local `.agent-local/PROJECT.md`.

Preserve the shallow module design: algorithms in `active_learning.py`, metrics in `evaluation.py`,
coordination in `pipeline.py`, external/model boundaries in `integrations/`, launchers in `entrypoints/`.
The new agent coordination files supplement this application; they do not replace its SQLite store
or its runtime experiment UUIDs with another service.

## Global research rules

- Distinguish FACT, IMPLEMENTATION STATE, MEASURED RESULT, HYPOTHESIS, ASSUMPTION,
  ESTIMATE and RECOMMENDATION. Never fabricate citations, results, jobs or completion.
- Record code commit, dataset identity and checksums, split version, model/checkpoint revision,
  seeds, software environment and full run configuration before interpreting a result.
- Preserve baselines; compare equal annotation budgets and documented conditions. Record
  deviations as deviations, never silently change methodology midway through a comparison.
- Isolate final test data; never train, acquire samples, tune hyperparameters or select methods
  using it. Check document-level leakage, duplicates and provenance, not just page filenames.
- Separate measured findings from hypotheses; disclose limitations, assumptions and confounds.

## Global engineering rules

- Inspect current code, configuration and tests before changes. Prefer a minimal coherent diff,
  existing interfaces and libraries. No speculative abstraction or coordination infrastructure.
- Every behavior change needs proportionate verification. Run focused and broader relevant
  checks, plus configured lint/types; absence of tests is a gap, not a pass.
- Preserve module boundaries and sensible compatibility; remove duplication only when safe.
- Inspect final diff. Never commit credentials, private datasets, raw private annotation exports,
  large checkpoints or generated run artifacts. Use external artifact locations and checksums.
- Production code belongs to Development. Platform owns authorized infrastructure/integration
  boundaries; assign overlapping files explicitly. Significant changes need independent Testing.

## Agent coordination (protocol 3)

Read this constitution, current main `.agents/skills/<role>/SKILL.md` and
[protocol](coordination/README.md). Read `.agent-local/TEAM.md`, PROJECT.md, your assigned
`tasks/ID.md` and relevant evidence from the shared local memory directory. It is ignored by Git;
all existing worktrees point to the same directory. Never use `git show` for local memory.

One existing persistent Codex task per role is the agent; a Skill is its playbook. Native Codex
messages deliver assignments and handoffs; files alone do not wake agents. The user authorizes
coordination among these seven existing tasks within scope. No external messages to people
without authorization. Temporary subagents, when authorized, cannot replace persistent roles.

Manager owns canonical main, local TEAM, assignment headers/releases, PROJECT and DECISIONS.
Specialists own local task Progress/Result and assigned artifacts on their isolated code branches.
Only one writer edits a task at a time: Manager changes it while its specialist is idle or after
an acknowledged checkpoint. Specialists never self-release dependencies or mark canonical DONE.
QA is read-only by default. Substantial QA writing needs an explicit assignment and isolated scope.

Commit reusable code, tests, approved plans, scientific notes and small reproducibility records.
Keep routine progress, machine paths, task IDs, setup evidence and operational inventory local.
Include reproducible verification and acceptance in public PRs/plans when relevant, without
copying private agent records. Follow the protocol for exact code SHAs and local task fingerprints.

`continue` rechecks current authorization/dependencies; `status` is read-only; `manager sync`
sends one meaningful handoff. No runnable assignment means wait or report a blocker. Avoid ACK
loops. Private conversation is not durable shared memory; uncommitted code is not a publication.

Evidence order: actual code/repository → tests/CI → experiment outputs → commit/diff/PR →
reproducible configuration → agent report → assertion. A PASS applies only to its scope/code SHA.

## Change control

Routine safe work proceeds within task authorization. Existing explicit user approval persists;
do not ask again unless scope materially changes. Obtain explicit approval before deleting
important datasets/checkpoints/cloud resources, overwriting results, destructive migrations,
credential changes, force-pushing shared branches/history rewriting, large unexpected spend,
changing ground truth/splits/final evaluation methodology or replacing the core research objective.
Prepare concrete evidence, alternatives and cost estimates before asking. Record decisions in
`.agent-local/DECISIONS.md`; a proposed plan or dispatch cannot substitute for required user approval.

Keep this FYP understandable by one student. Infrastructure must improve research correctness,
reproducibility, development speed or experiment reliability to justify its existence.
