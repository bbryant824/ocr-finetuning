# FYP project constitution

## Project mission

The project studies whether active-learning strategies reduce the human annotation needed
to adapt OCR to historical and low-resource documents. Page images are the acquisition unit;
the current research simulates annotation by revealing existing source ground truth only for
selected training pages. Ground truth contains the source's line boxes and transcriptions.
The practical objective is a small, reproducible, understandable pipeline with fair comparisons.

Stage 2 engineering and the LLaMA-Factory migration are complete. EXP-005 passed independent
artifact review: 16 initial TRAIN pages, 8 acquired, reset-fit on 24, fresh-process reloads, joint
validation, one committed round, resume without extra calls and matching exports. The final eight
VAL outputs failed OCR readiness: one valid, CER 0.999512, WER 1.0, box F1 zero. Useful OCR and an active-learning
benefit remain unestablished. Real acquisition currently supports random only. Keep source-label
simulation, interchangeable boundaries and minimal straightforward code. A labeling frontend is
not required. Revealed pages are simulated budgets, not human time.

## Architecture and current evidence

Read [the project guide](docs/PROJECT_GUIDE.md) for the complete repository map, data/model/Modal
connections, operating procedure, research stage and evidence index. The current completed run is
[EXP-005](experiments/EXP-005.md); EXP-003 preserves the historical custom-runtime result.
Current assignments and authorization live in shared local `.agent-local/PROJECT.md`, not in historical implementation plans or private conversation.

Preserve the shallow module design: algorithms in `active_learning.py`, metrics in `evaluation.py`,
coordination in `pipeline.py`, external/model boundaries in `integrations/`, launchers in `entrypoints/`.
`LocalOracle` reveals only selected TRAIN labels; only the evaluator receives validation truth.
The supported paths are fixture simulation and real `ModalModel`/LLaMA-Factory execution.
The toolkit owns model training/inference; our code owns selection, selected-only truth reveal,
joint OCR metrics and durable initial-fit/round state. Legacy Label Studio, HTTP GPU services,
polling setup and the custom Qwen trainer have been removed. Use the versioned execution recipe
and exact-source evidence in the guide; source tests alone are not a GPU or OCR-quality result.
SQLite and content-addressed artifacts hold application state. Agent coordination supplements
this application; it never replaces its database or runtime UUIDs with another service.

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
- Minimize token usage and coordination cost. Search first and read only relevant source, task and
  evidence sections; do not reload full histories, enumerate every role or repeat unchanged context.
- Use one implementation owner and one concise result handoff. Involve Research/Planning only for
  an unresolved scientific/design decision and Testing once for significant changes. Do not pass
  routine work through every role, create ACK loops or poll unchanged state.
- Diagnose from an observed failure, make the smallest supported fix and verify its affected behavior.
  Run focused regressions first; run the broader relevant suite once for integration/refactors.
  Repeat only for new changes, failures or a specific unresolved concern. No redundant GPU runs,
  speculative debugging campaigns, duplicated tests or tests that merely restate implementation.
- Keep instructions, plans and progress concise: one current summary, links to evidence, no copied
  transcripts or parallel status documents. Replace obsolete guidance instead of appending history.
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
QA has standing permission to author requested project documentation, reports, mentor PPTX decks
and supporting visuals, without a separate Manager release. Its code/research-state access remains
read-only; use owned artifact paths and the QA Skill. Local authoring must not modify active agents'
files, shared Git state, canonical task records or research decisions.

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
