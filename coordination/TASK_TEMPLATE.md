# ID — Short outcome

ID: <Manager-allocated task ID>
Owner: <role from TEAM>
Priority: <P0 / P1 / P2>
Status: <BACKLOG / READY / IN_PROGRESS / BLOCKED / WAITING_REVIEW / DONE / CANCELLED>
Depends: <NONE or comma-separated TASK-ID@DONE; review may use TASK-ID@WAITING_REVIEW>
Candidate: <NONE or full immutable implementation commit SHA for review>
Updated: <YYYY-MM-DD>

## Assignment

Outcome, inputs/accepted plan version, owned files/artifacts, scope and exclusions. Define these
once. Link any relevant research, run, issue or PR rather than adding empty metadata fields.

## Acceptance

Observable criteria and proportionate verification; independent review and real-smoke requirements
where relevant. State resource/approval limits for execution work.

## Progress

Short dated updates: observed control SHA, what changed, blockers and next action.
Keep meaningful checkpoints, not a transcript. Only the assigned specialist updates these local sections; Manager serializes header edits.

## Result

Outcome, artifact links, exact commands/results/environment, limitations and requested next action.
For review include author, candidate SHA, PASS/PASS WITH WARNINGS/FAIL and acceptance mapping.
Manager records acceptance/integration evidence here. The handoff carries artifact/code SHA when any, plus the local task SHA-256.
