# FYP agent protocol

Protocol: 3. Manager works by default; specialists are used only when their expertise or an
independent review materially improves the result. The seven existing Codex tasks are registered
in ignored `.agent-local/TEAM.md`. A task message starts work; a file edit does not.

## Active state and evidence

Read `AGENTS.md` and `.agent-local/PROJECT.md` first. Consult TEAM, the assigned task,
the relevant role Skill and source/experiment evidence only as needed. PROJECT is the sole
current local summary; TEAM maps verified task IDs/worktrees; `tasks/ID.md` holds a released
specialist assignment and result. Historical local setup/decision/ops records are retained in
ignored `.local/coordination-history/`, outside routine context. Public methods and
measured runs belong in `plans/`, `research/` and `experiments/`. A fresh clone needs no private
memory to run application code; missing local memory blocks agent dispatch, not ordinary work.

Keep one writer per file. Manager controls releases, acceptance and integration; specialists
write only their assigned result/artifacts. QA has standing authorization for requested teaching,
documents and slides; it does not change code or Manager state. Do not infer authorization from a
stale task header or private cross-task conversation.

## When a specialist is needed

1. Manager checks evidence, names one owner and sends one scoped native message to the verified
   TEAM ID. Create/update a task record and TEAM pointer only for sustained work that needs durable
   assignment state. State owned outputs, limits and acceptance; include relevant source SHA and
   task-file SHA-256 when a task file changes. No message to every role or ACK loop.
2. The specialist reads the message and any assigned task, works in its worktree or explicitly
   assigned checkout, runs proportionate checks, and returns one concise result with exact
   SHA/commands, observations, limits and links. Record it in the task only when one exists.
3. Manager verifies the returned evidence, uses independent Testing only for significant/risky
   scope, integrates accepted work and updates PROJECT when priorities change. Reuse unchanged
   checks. An uncertain or failed provider call must be reconciled before another paid run.

States are BACKLOG, READY, IN_PROGRESS, BLOCKED, WAITING_REVIEW, DONE and CANCELLED. A dependent
task is not released just because its prerequisite finished; Manager checks and releases it.
Technical PASS, completed execution and scientific support have different meanings. Task IDs and
experiment IDs are separate; preserve run UUIDs and failed attempts.

## Storage and checks

Commit small reusable code, instructions, plans, research notes and experiment metadata. Keep
secrets, datasets, checkpoints, raw output, machine paths and routine task/ops notes ignored.
Never push all branches or local archive refs. Preserve shared `.agent-local` when moving the
checkout; worktree symlinks and TEAM paths need explicit verification after a move.

Use `python3 scripts/validate_agent_system.py --live` after role/protocol changes, and
`python3 scripts/test_agent_system.py` only when changing its validator or task schema.
These checks cannot prove message delivery, experiment correctness or cloud state.
