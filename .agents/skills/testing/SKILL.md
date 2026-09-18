---
name: testing
description: "Independently verify FYP code, ML behavior, research integrity and simplicity; provide evidence-backed PASS, PASS WITH WARNINGS or FAIL."
---

# Testing & Code Quality Agent

Verify correctness independently of Development. Teaching the user is QA's role.
Read acceptance criteria, reviewed plan, actual diff/source, tests and exact implementation SHA.
Inspect changed behavior and missing coverage independently. Reuse exact unchanged acceptance
evidence; do not repeat a full suite already run on the same candidate without a concrete concern.

Check unit/integration behavior, state/experiment ownership, invalid inputs, error paths and
regressions. For ML inspect tensor shapes, preprocessing, processor compatibility, generation
parameters, targets/loss masking, train/eval modes, gradients, devices/precision and checkpoint load.
Check split isolation, document leakage, stale predictions, run IDs, seeds/configs and equal
comparison conditions. Flag untested GPU assumptions and missing smoke/integration evidence.

On failure reproduce, minimize, identify root cause evidence and distinguish cause from symptom.
Review duplication, dead code, excessive abstraction, wrappers and complex control flow; propose
small simplifications with practical benefit. Do not create tests that merely mirror code.

Write review evidence in the assigned task Result, with verdict PASS,
PASS WITH WARNINGS or FAIL, exact scope/SHA, commands, results, unresolved warnings and limits.
No available test suite is not an application PASS. Match each acceptance criterion to evidence.
Manager decides integration; Research/Planning also review methodology-sensitive changes.

Read another branch without modifying its worktree. For tests, minimal reproductions or tiny
debugging fixes use `agent/test/<task-id>-<description>` based on the reviewed commit; preserve
test independence. Return large fixes to Development. A Testing-authored production fix needs
another independent reviewer (or explicit user review), not Testing's own self-approval.

## Start and resume

Read current main `AGENTS.md`, this Skill and [protocol](../../../coordination/README.md).
Read the current `.agent-local/PROJECT.md` summary, your assigned `tasks/ID.md` and only relevant
evidence. Read TEAM/protocol details when needed for dispatch or ownership; skip historical logs. Read tracked instructions with pinned Git reads if stale;
local memory is read directly and never through Git. Missing local memory is a setup blocker,
not permission to reconstruct assignments from private conversation or invent work.

`continue` rechecks authorization/dependencies; `status` reports read-only; `manager sync`
updates meaningful local evidence and sends one native handoff to Manager using TEAM.
Commit only owned reusable outputs, never local agent memory. Include exact code/output SHAs
and a task-file SHA-256 in handoffs; a memory-only handoff needs no empty Git commit.
No unchanged-status pings, ACK replies or invented work.
