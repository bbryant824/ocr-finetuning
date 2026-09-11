# Using the FYP agents

Protocol: 3

Use the existing **Manager** task for project control and **QA & Understand** for explanations.
A Codex task is the agent; its Skill is the playbook. Native task messages deliver work.
The existing seven tasks and worktrees remain; this protocol does not create a daemon or scheduler.

## What goes to GitHub

| Keep in Git | Keep local and ignored |
| --- | --- |
| Code, tests, dependency locks and safe example configuration | Secrets, environments, datasets, checkpoints and raw runtime outputs |
| AGENTS.md, seven Skills, this guide and reusable templates/checks | `.agent-local/`: task IDs, machine paths, assignments, routine progress, setup history and ops inventory |
| Substantive research notes, reviewed plans, small experiment metadata and analysis | `.local/`: scratch notes and downloads |

Commit outputs selectively after inspecting the diff. Record scientific methodology and accepted
changes in the relevant plan, research note, experiment record or PR so GitHub remains reproducible.
Do not publish `.agent-local`, old setup/archive branches, all refs, raw local reports or credentials.
`.gitignore` does not remove already tracked files or erase earlier commits. Before first publication,
inspect all outgoing commits. Push the intended public branch explicitly; never use `--all` or `--mirror`.

## Shared local memory

The canonical checkout contains `.agent-local/`; specialist worktrees have an ignored symlink to
that same directory. This avoids copying or merging private task records. Read it directly:

```text
.agent-local/
  PROJECT.md       current priorities, roadmap and blockers (Manager)
  TEAM.md          verified task IDs, paths, branches and assignments (Manager)
  DECISIONS.md     local authorization and coordination decisions (Manager)
  HISTORY.md       setup evidence and local recovery references
  tasks/ID.md      assignment, progress, result and review in one record
  ops/STATUS.md    machine/data/service/cost observations (Platform)
```

Each task follows [TASK_TEMPLATE](TASK_TEMPLATE.md). Manager owns headers/releases/acceptance;
the assigned specialist owns Progress/Result. Manager edits a task only while its specialist is
idle or after an acknowledged checkpoint. Write one complete edit at a time, inspect the diff or
prior text, and re-read before replacing a file. Never overwrite another writer's changes.
QA remains read-only unless assigned substantial writing with explicit ownership.

The shared directory is local durable memory, not Git history or a cloud backup. Include it in
normal private computer backups. Preserve it before removing/moving the canonical checkout;
repoint worktree symlinks if that checkout moves. A fresh clone can run the application and portable
checks without it. To move the existing team to another machine, privately transfer the directory,
then verify task IDs/worktrees and update TEAM; never infer live identities from a template.
For a new worktree, Manager links its `.agent-local` to the canonical directory after verifying
the actual main worktree with `git worktree list --porcelain`. Never overwrite an existing local
directory or link silently. Missing/broken memory means report BLOCKED for agent work.

## Control in ordinary language

These are prompts, not installed shell commands.

| Say | Expected behavior |
| --- | --- |
| Manager: `project status` | Inspect evidence and report progress/blockers; do not launch work |
| Manager: `Continue the ready work and coordinate the team` | Coordinate eligible assignments within authorization and review returned evidence |
| Manager: `Assign Research to investigate ...` | Define a bounded local task, update TEAM, then message the existing Research task |
| Specialist: `continue` | Read fresh instructions/local assignment; work only when released and prerequisites met |
| Specialist: `status` | Report evidence, blockers and next step read-only |
| Specialist: `manager sync` | Record meaningful local progress and send one evidence handoff |
| Manager: `Pause the team` | Request safe checkpoints; after receipt record holds and verify stopped work |
| QA: `Trace how an image moves through the code` | Inspect and explain without edits |

READY means eligible on continuation, not already running. No runnable assignment means wait or
report a blocker. A setup/communication check does not authorize research execution or paid jobs.

## Assignment and return

1. Manager reads current code/evidence, writes the local task and TEAM pointer while the owner is
   idle, and checks dependencies. Define inputs, owned files, exclusions, acceptance and limits once.
2. Send one native `send_message_to_thread` to the verified TEAM ID/host. Include task ID,
   current main SHA for tracked instructions and SHA-256 of the local assignment file, computed
   after the edit (`shasum -a 256 .agent-local/tasks/ID.md`). This fingerprint detects stale messages;
   it is not a Git commit or an immutable copy of later task updates.
3. Specialist reads fresh local state. If a received fingerprint differs, re-read authorization and
   dependencies and resolve contradictory scope with Manager. Inspect its own checkout before work.
   Recheck the task header before publishing or costly actions. A local file cannot self-authorize work.
4. Work on the isolated code branch. Commit safe owned code/research/plan/run outputs and record
   exact SHA, observed control, verification, limitations and next action in the local task Result.
   Memory-only work needs no Git commit. Do not commit routine task updates or machine identifiers.
5. Send Manager one `[HANDOFF]` with task ID, artifact/code SHA when any, local task fingerprint,
   changed scope, verdict/blockers and requested next action. Manager inspects exact committed
   outputs and local review evidence, integrates accepted changes and records canonical acceptance.
   Update PROJECT only when priorities change. Publish relevant reproducible verification in PRs/plans.

A native send proves submission; receipt/progress provides stronger evidence. Use `list_threads`
to reverify identities, `read_thread` for context and bounded `wait_threads` for progress/completion.
If delivery fails, retain the local result and committed outputs, report pending delivery and let
Manager inspect directly. Do not recreate agents. No self-messaging, ACK reply loops or status spam.
A handoff alone does not authorize unrelated next work. External messages and recurring schedules
require user authorization. Conversation history alone is not shared project memory.

## Worktrees, gates and review

Same-repository worktrees share Git refs, and the installed symlinks share local memory.
Code checkout files remain isolated. Pin current control when reading tracked instructions:

```sh
control_sha=$(git rev-parse --verify refs/heads/main)
git show "${control_sha}:AGENTS.md"
git show "${control_sha}:coordination/README.md"
# Substitute the current role:
git show "${control_sha}:.agents/skills/development/SKILL.md"
cat .agent-local/TEAM.md .agent-local/PROJECT.md
```

Never use `git show` for local task memory. Main does not contain it. Inspect dirty/untracked work
before deliberately merging current public main into a specialist branch. Preserve owned work;
never merge local archive branches into public development. The pre-publication setup histories
remain on local archive refs only. Other clones/hosts do not automatically share refs or memory.

States: BACKLOG, READY, IN_PROGRESS, BLOCKED, WAITING_REVIEW, DONE, CANCELLED. Manager owns
releases and acceptance. `Depends: RES-001@DONE` requires accepted DONE plus Manager release.
Implementation review uses `DEV-001@WAITING_REVIEW` with an immutable Candidate SHA; an already
DONE candidate can also be reviewed. BLOCKED remains held even after a prerequisite completes.
Testing gives PASS / PASS WITH WARNINGS / FAIL for exact scope/SHA; Manager accepts warnings
explicitly. Changed implementation needs affected-scope review. Research/Planning confirm
methodology-sensitive changes. Technical PASS, completed execution and scientific support differ.

Allocate unique MGR/RES/PLAN/DEV/TEST/EXP task IDs with at least three digits. Experiment records
under `experiments/` describe actual executions and retain runtime UUIDs; they are not local task
records. Preserve Research → Planning → Development → independent Testing → Platform execution
→ Research interpretation. Keep current dependency gates; QA assists independently.

## Checks

`python3 scripts/validate_agent_system.py` checks portable links, Skills and templates, and local
records when present. `--live` requires local records and verifies registered Git branches. Run
`python3 scripts/test_agent_system.py` after protocol/checker changes. These checks cannot prove
native delivery, approvals, scientific validity, model correctness or service readiness.
