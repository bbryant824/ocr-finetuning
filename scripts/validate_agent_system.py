#!/usr/bin/env python3
"""Read-only checks for the FYP Markdown protocol; no services or extra dependencies."""

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import date
from pathlib import Path
from urllib.parse import unquote, urlsplit
from uuid import UUID

ROLES = {
    "manager",
    "research",
    "planning",
    "development",
    "testing",
    "experiment-platform",
    "qa-understanding",
}
TASK_STATES = {
    "BACKLOG",
    "READY",
    "IN_PROGRESS",
    "BLOCKED",
    "WAITING_REVIEW",
    "DONE",
    "CANCELLED",
}
TASK_ID = r"(?:MGR|RES|PLAN|DEV|TEST|EXP)-\d{3,}"
SHA = re.compile(r"[0-9a-f]{40}")
FOLDERS = ("coordination", "research", "plans", "experiments", ".agents/skills")
RETIRED = re.compile(
    r"coordination/(?:dispatch|status|reports|decisions)/|docs/agent-system/|"
    r"(?:PROJECT_STATE|CURRENT_FOCUS|ROADMAP|BACKLOG|THREADS)\.md|"
    r"ops/(?:PLATFORM_STATUS|DATA_STATUS|GPU_STATUS|INTEGRATIONS|COSTS|README)\.md|"
    r"research/(?:literature|synthesis|proposals)/|experiments/(?:registry|results)/"
)


def fields(text: str) -> dict[str, str]:
    """Read header metadata only; body observations must not override task authority."""
    header = text.split("\n## ", 1)[0]
    return dict(re.findall(r"^([A-Za-z][A-Za-z /-]*):[ \t]*(.*)$", header, re.M))


def team_rows(text: str) -> list[dict[str, str]]:
    rows = []
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not line.startswith("| ["):
            continue
        if len(cells) != 7:
            rows.append({"role": "INVALID_ROW"})
            continue
        role = re.fullmatch(r"\[([^]]+)\]\(([^)]+)\)", cells[0])
        task = re.fullmatch(r"\[([^]]+)\]\(([^)]+)\)", cells[6])
        row = dict(
            zip(
                ("role", "title", "id", "host", "path", "branch", "task"),
                [role[1] if role else cells[0], *cells[1:6], task[1] if task else cells[6]],
                strict=True,
            )
        )
        row["skill_link"] = role[2] if role else ""
        row["task_link"] = task[2] if task else ""
        rows.append(row)
    return rows


def validate(root: Path, *, live: bool = False) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    memory = root / ".agent-local"
    has_memory = memory.is_dir()
    if (live or memory.is_symlink()) and not has_memory:
        errors.append("Local agent memory missing or broken: .agent-local")
    documents = [root / "AGENTS.md", root / "README.md"]
    for folder in FOLDERS:
        documents.extend(sorted((root / folder).rglob("*.md")))
    if has_memory:
        documents.extend(sorted(memory.rglob("*.md")))
    required = [
        "AGENTS.md",
        "README.md",
        "coordination/README.md",
        "coordination/TASK_TEMPLATE.md",
        *(
            f"{folder}/{name}.md"
            for folder in ("research", "plans", "experiments")
            for name in ("README", "TEMPLATE")
        ),
    ]
    if has_memory:
        required.extend(
            f".agent-local/{name}.md"
            for name in ("PROJECT", "TEAM")
        )
    for relative in required:
        if not (root / relative).is_file():
            errors.append(f"Missing required file: {relative}")
    for path in documents:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(root).as_posix()
        if not text.strip():
            errors.append(f"Empty document: {relative}")
        # HISTORY explicitly documents old Git paths; current instructions must not use them.
        if not relative.startswith(".agent-local/") and RETIRED.search(text):
            errors.append(f"Retired protocol reference in {relative}")
        for target in re.findall(r"\[[^\]\n]+\]\(([^)\n]+)\)", text):
            target = target.strip().strip("<>")
            if urlsplit(target).scheme or target.startswith("#") or "<" in target:
                continue
            local = unquote(target.split("#", 1)[0])
            if local and not (path.parent / local).exists():
                errors.append(f"Broken link in {relative}: {target}")
    for relative in (
        "coordination/dispatch",
        "coordination/status",
        "coordination/reports",
        "coordination/decisions",
        "docs/agent-system",
        "experiments/registry",
        "experiments/results",
        "research/literature",
        "research/synthesis",
        "research/proposals",
    ):
        if any((root / relative).rglob("*.md")):
            errors.append(f"Retired protocol folder still contains Markdown: {relative}")

    skills = {p.parent.name: p for p in (root / ".agents/skills").glob("*/SKILL.md")}
    if set(skills) != ROLES:
        errors.append(
            f"Skill roles differ: missing={ROLES - set(skills)}, extra={set(skills) - ROLES}"
        )
    for role, path in skills.items():
        front = re.match(r"---\n(.*?)\n---\n", path.read_text(), re.S)
        metadata = fields(front[1]) if front else {}
        if metadata.get("name") != role or not metadata.get("description"):
            errors.append(f"Invalid skill metadata: {role}")

    if not has_memory:
        return errors, {"skills": len(skills), "tasks": 0, "documents": len(documents)}

    tasks = {}
    graph: dict[str, list[tuple[str, str]]] = {}
    for path in sorted((memory / "tasks").glob("*.md")):
        if path.name == "TEMPLATE.md":
            continue
        text = path.read_text()
        metadata = fields(text)
        keys = re.findall(r"^([A-Za-z][A-Za-z /-]*):", text.split("\n## ", 1)[0], re.M)
        if len(keys) != len(set(keys)):
            errors.append(f"{path.stem}: duplicate task metadata is ambiguous")
        tasks[path.stem] = metadata
        if not re.fullmatch(TASK_ID, path.stem) or metadata.get("ID") != path.stem:
            errors.append(f"Invalid task filename/ID: {path.name}")
        for key in ("ID", "Owner", "Priority", "Status", "Depends", "Candidate", "Updated"):
            if not metadata.get(key) or "<" in metadata.get(key, ""):
                errors.append(f"{path.stem}: missing/unfilled {key}")
        for heading in ("Assignment", "Acceptance", "Progress", "Result"):
            if not re.search(rf"^## {heading}\n\s*\S", text, re.M):
                errors.append(f"{path.stem}: missing/empty {heading}")
        if metadata.get("Owner") not in ROLES:
            errors.append(f"{path.stem}: unknown owner")
        if metadata.get("Status") not in TASK_STATES:
            errors.append(f"{path.stem}: invalid status")
        if metadata.get("Priority") not in {"P0", "P1", "P2"}:
            errors.append(f"{path.stem}: invalid priority")
        if metadata.get("Candidate") != "NONE" and not SHA.fullmatch(metadata.get("Candidate", "")):
            errors.append(f"{path.stem}: invalid candidate SHA")
        try:
            date.fromisoformat(metadata.get("Updated", ""))
        except ValueError:
            errors.append(f"{path.stem}: invalid update date")
        graph[path.stem] = []
        if metadata.get("Depends") != "NONE":
            for raw in metadata.get("Depends", "").split(","):
                match = re.fullmatch(rf"({TASK_ID})@(DONE|WAITING_REVIEW)", raw.strip())
                if not match:
                    errors.append(f"{path.stem}: invalid dependency {raw!r}")
                else:
                    graph[path.stem].append((match[1], match[2]))
    for task, dependencies in graph.items():
        for dependency, gate in dependencies:
            target = tasks.get(dependency)
            if not target:
                errors.append(f"{task}: missing dependency {dependency}")
                continue
            if tasks[task].get("Status") in {"READY", "IN_PROGRESS", "WAITING_REVIEW"}:
                allowed = {"DONE"} if gate == "DONE" else {"WAITING_REVIEW", "DONE"}
                if target.get("Status") not in allowed:
                    errors.append(f"{task}: unmet gate {dependency}@{gate}")
                if gate == "WAITING_REVIEW" and not SHA.fullmatch(target.get("Candidate", "")):
                    errors.append(f"{task}: review gate lacks immutable candidate for {dependency}")
    visiting, visited = set(), set()

    def walk(task: str) -> None:
        if task in visiting:
            errors.append(f"Dependency cycle at {task}")
            return
        if task in visited:
            return
        visiting.add(task)
        for dependency, _gate in graph.get(task, []):
            walk(dependency)
        visiting.remove(task)
        visited.add(task)

    for task in tasks:
        walk(task)

    team = memory / "TEAM.md"
    rows = team_rows(team.read_text()) if team.exists() else []
    if len(rows) != 7 or {row["role"] for row in rows} != ROLES:
        errors.append("TEAM must contain exactly the seven roles once")
    ids, writers, branches = set(), set(), set()
    for row in rows:
        role = row["role"]
        skill_target = (team.parent / row.get("skill_link", "")).resolve()
        if skill_target != (memory.resolve().parent / f".agents/skills/{role}/SKILL.md").resolve():
            errors.append(f"{role}: Skill link disagrees with role")
        try:
            UUID(row.get("id", ""))
        except ValueError:
            errors.append(f"{role}: invalid Codex task ID")
        if row.get("id") in ids:
            errors.append(f"{role}: duplicate Codex task ID")
        ids.add(row.get("id"))
        if not row.get("host") or not row.get("title") or not row.get("branch"):
            errors.append(f"{role}: missing host/title/branch")
        path = Path(row.get("path", ""))
        if not path.is_absolute():
            errors.append(f"{role}: worktree path must be absolute")
        if role != "qa-understanding":
            if str(path) in writers or row.get("branch") in branches:
                errors.append(f"{role}: duplicate writable worktree/branch")
            writers.add(str(path))
            branches.add(row.get("branch"))
        task = row.get("task", "")
        if task != "NONE (read-only questions)" and task != "NONE":
            task_target = (team.parent / row.get("task_link", "")).resolve()
            if task_target != (memory / f"tasks/{task}.md").resolve():
                errors.append(f"{role}: assignment link disagrees with task label")
            if task not in tasks:
                errors.append(f"{role}: assignment missing {task}")
            elif tasks[task].get("Owner") != role:
                errors.append(f"{role}: assignment owner mismatch for {task}")
        if live:
            try:
                branch = subprocess.check_output(
                    ["git", "-C", str(path), "branch", "--show-current"],
                    text=True,
                    stderr=subprocess.STDOUT,
                ).strip()
                if branch != row.get("branch"):
                    errors.append(f"{role}: live branch differs: {branch}")
            except (OSError, subprocess.CalledProcessError):
                errors.append(f"{role}: cannot inspect registered Git worktree")
    return errors, {"skills": len(skills), "tasks": len(tasks), "documents": len(documents)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--live", action="store_true", help="Require local memory and read registered Git branches"
    )
    args = parser.parse_args()
    errors, counts = validate(args.root.resolve(), live=args.live)
    for error in errors:
        print(f"FAIL: {error}")
    if errors:
        return 1
    print("PASS: " + ", ".join(f"{value} {key}" for key, value in counts.items()))
    print("Checked portable instructions and any installed local tasks/TEAM.")
    print(
        "Does not verify message delivery, approvals, experiments, model correctness or services."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
