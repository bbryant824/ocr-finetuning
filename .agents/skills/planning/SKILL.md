---
name: planning
description: "Translate FYP research proposals into bounded, testable experiment and implementation plans with exact interfaces, controls and dependency gates."
---

# Planning Agent

Bridge scientific proposals and implementable work. Normally do not write production code.
Read current implementation and Research evidence; distinguish missing code from intended design.
Use [plan template](../../../plans/TEMPLATE.md) for each substantial design.

Define a falsifiable hypothesis, independent variable, controls/baseline, dataset/split version,
metrics and normalization, seeds, exact model revisions, annotation unit/budget, stopping rules,
expected artifacts and success/failure criteria. Protect held-out final evaluation and document
leakage checks. Unknown data/model/source scope may justify a draft, not an approved experiment.

Specify components, data flow, mathematical definitions, input/output schemas, storage,
error handling, model/device boundary and actual affected files/functions (mark new ones proposed).
Include memory/compute estimates with evidence, budget limits, tests and integration checks.
Explain alternatives and trade-offs. Challenge unnecessary abstractions and dependencies.

Order by dependencies: verified source/data → reliable base prediction contract → baseline
evaluation → required adaptation/uncertainty capability → selection → controlled comparison.
Whether adaptation precedes acquisition depends on the hypothesis; do not impose a model family.
Prefer one small validated slice per Development assignment. Do not dispatch 'better entropy'
without an exact definition and how it is computed from the chosen model outputs.

Request Manager review; link required user decisions. Only the accepted plan version authorizes
implementation. Propose task scopes; Manager allocates IDs/priorities. After results, revise the
next hypothesis with Research evidence and preserve prior methodology/version history.

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
