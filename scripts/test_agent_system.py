"""Regression checks for coordination errors, using disposable Markdown copies."""

import tempfile
import unittest
from pathlib import Path

from validate_agent_system import ROLES, fields, validate

ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Stable synthetic tasks: future real task completion must not break these cases.
        for relative in (
            "AGENTS.md",
            "README.md",
            "coordination/README.md",
            ".agent-local/PROJECT.md",
            "coordination/TASK_TEMPLATE.md",
            "research/README.md",
            "research/TEMPLATE.md",
            "plans/README.md",
            "plans/TEMPLATE.md",
            "experiments/README.md",
            "experiments/TEMPLATE.md",
        ):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Synthetic fixture\n")
        assignments = {
            "manager": "MGR-001",
            "research": "RES-001",
            "planning": "PLAN-001",
            "development": "DEV-001",
            "testing": "TEST-001",
            "experiment-platform": "EXP-001",
        }
        team = []
        for index, role in enumerate(sorted(ROLES), 1):
            skill = self.root / f".agents/skills/{role}/SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text(f"---\nname: {role}\ndescription: Synthetic fixture\n---\n")
            task = assignments.get(role)
            task_link = f"[{task}](tasks/{task}.md)" if task else "NONE (read-only questions)"
            team.append(
                f"| [{role}](../.agents/skills/{role}/SKILL.md) | {role} | "
                f"00000000-0000-4000-8000-{index:012d} | local | /tmp/fyp-{role} | "
                f"agent/{role}/fixture | {task_link} |"
            )
            if task:
                depends = {"PLAN-001": "RES-001@DONE", "DEV-001": "PLAN-001@DONE"}.get(task, "NONE")
                self.write_task(task, role, "BLOCKED" if depends != "NONE" else "READY", depends)
        (self.root / ".agent-local/TEAM.md").write_text("\n".join(team) + "\n")
        self.write_task("TEST-002", "testing", "BACKLOG", "DEV-001@WAITING_REVIEW")

    def write_task(self, task, owner, state, depends):
        (self.root / ".agent-local/tasks").mkdir(parents=True, exist_ok=True)
        (self.root / f".agent-local/tasks/{task}.md").write_text(
            f"# {task}\n\nID: {task}\nOwner: {owner}\nPriority: P1\nStatus: {state}\n"
            f"Depends: {depends}\nCandidate: NONE\nUpdated: 2026-09-10\n\n"
            "## Assignment\n\nFixture scope.\n\n## Acceptance\n\nFixture criteria.\n\n"
            "## Progress\n\nNo real work.\n\n## Result\n\nNo real result.\n"
        )

    def replace(self, path, before, after):
        target = self.root / path
        text = target.read_text()
        self.assertIn(before, text)
        target.write_text(text.replace(before, after))

    def assert_error(self, message):
        errors, _ = validate(self.root)
        self.assertTrue(any(message in error for error in errors), errors)

    def test_public_checkout_without_private_memory(self):
        import shutil

        shutil.rmtree(self.root / ".agent-local")
        errors, counts = validate(self.root)
        self.assertEqual(errors, [])
        self.assertEqual(counts["tasks"], 0)
        self.assertTrue(
            any("Local agent memory missing" in e for e in validate(self.root, live=True)[0])
        )

    def test_shared_worktree_memory_and_broken_link(self):
        import shutil

        checkout = self.root / "checkout"
        shutil.copytree(
            self.root, checkout, ignore=shutil.ignore_patterns("checkout", ".agent-local")
        )
        (checkout / ".agent-local").symlink_to(self.root / ".agent-local", target_is_directory=True)
        self.assertEqual(validate(checkout)[0], [])
        self.replace(".agent-local/tasks/DEV-001.md", "Status: BLOCKED", "Status: READY")
        self.assertTrue(any("unmet gate" in e for e in validate(checkout)[0]))
        (checkout / ".agent-local").unlink()
        (checkout / ".agent-local").symlink_to(self.root / "missing", target_is_directory=True)
        self.assertTrue(any("Local agent memory missing" in e for e in validate(checkout)[0]))

    def test_current_protocol(self):
        self.assertEqual(validate(ROOT)[0], [])
        self.assertEqual(validate(self.root)[0], [])

    def test_stale_instruction_link(self):
        with (self.root / "AGENTS.md").open("a") as target:
            target.write("\nUse [old state](coordination/PROJECT_STATE.md).\n")
        self.assert_error("Retired protocol reference")
        self.assert_error("Broken link")

    def test_wrong_assignment_owner(self):
        self.replace(
            ".agent-local/TEAM.md", "[RES-001](tasks/RES-001.md)", "[DEV-001](tasks/DEV-001.md)"
        )
        self.assert_error("assignment owner mismatch")

    def test_duplicate_delivery_destination(self):
        self.replace(
            ".agent-local/TEAM.md",
            "00000000-0000-4000-8000-000000000006",
            "00000000-0000-4000-8000-000000000003",
        )
        self.assert_error("duplicate Codex task ID")

    def test_unreleased_dependency(self):
        self.replace(".agent-local/tasks/DEV-001.md", "Status: BLOCKED", "Status: READY")
        self.assert_error("unmet gate PLAN-001@DONE")

    def test_missing_dependency_and_cycle(self):
        self.replace(".agent-local/tasks/DEV-001.md", "PLAN-001@DONE", "PLAN-999@DONE")
        self.assert_error("missing dependency PLAN-999")
        self.replace(".agent-local/tasks/DEV-001.md", "PLAN-999@DONE", "DEV-001@DONE")
        self.assert_error("Dependency cycle")

    def test_review_without_candidate(self):
        self.replace(".agent-local/tasks/DEV-001.md", "Status: BLOCKED", "Status: WAITING_REVIEW")
        self.replace(".agent-local/tasks/DEV-001.md", "Depends: PLAN-001@DONE", "Depends: NONE")
        self.replace(".agent-local/tasks/TEST-002.md", "Status: BACKLOG", "Status: READY")
        self.assert_error("review gate lacks immutable candidate")
        self.replace(".agent-local/tasks/DEV-001.md", "Candidate: NONE", "Candidate: " + "a" * 40)
        self.assertEqual(validate(self.root)[0], [])

    def test_invalid_task_state(self):
        self.replace(".agent-local/tasks/DEV-001.md", "Status: BLOCKED", "Status: SUCCESS")
        self.assert_error("invalid status")

    def test_body_cannot_override_assignment(self):
        task = self.root / ".agent-local/tasks/DEV-001.md"
        with task.open("a") as target:
            target.write("\nStatus: DONE\nOwner: research\n")
        self.assertEqual(fields(task.read_text())["Status"], "BLOCKED")
        self.assertEqual(fields(task.read_text())["Owner"], "development")
        self.assertEqual(validate(self.root)[0], [])

    def test_conflicting_status_cannot_satisfy_dependency(self):
        """A duplicated header must not turn unfinished research into accepted input."""
        self.replace(".agent-local/tasks/PLAN-001.md", "Status: BLOCKED", "Status: READY")
        self.assert_error("unmet gate RES-001@DONE")
        self.replace(
            ".agent-local/tasks/RES-001.md",
            "Status: READY",
            "Status: READY\nStatus: DONE",
        )
        self.assertTrue(validate(self.root)[0], "Conflicting status silently releases Planning")

    def test_conflicting_dependencies_cannot_release_development(self):
        """A partial metadata edit must not hide the original prerequisite."""
        self.replace(".agent-local/tasks/DEV-001.md", "Status: BLOCKED", "Status: READY")
        self.replace(
            ".agent-local/tasks/DEV-001.md",
            "Depends: PLAN-001@DONE",
            "Depends: PLAN-001@DONE\nDepends: NONE",
        )
        self.assertTrue(validate(self.root)[0], "Conflicting dependencies silently release DEV")

    def test_team_assignment_label_must_match_destination(self):
        """Following a TEAM link must yield the assignment the registry validates."""
        self.replace(
            ".agent-local/TEAM.md",
            "[TEST-001](tasks/TEST-001.md)",
            "[TEST-001](tasks/TEST-002.md)",
        )
        self.assertTrue(validate(self.root)[0], "TEAM label and linked assignment disagree")

    def test_done_prerequisite_does_not_require_automatic_release(self):
        self.replace(".agent-local/tasks/RES-001.md", "Status: READY", "Status: DONE")
        self.assertEqual(validate(self.root)[0], [])
        plan = (self.root / ".agent-local/tasks/PLAN-001.md").read_text()
        self.assertEqual(fields(plan)["Status"], "BLOCKED")

    def test_completed_implementation_can_retain_review_candidate(self):
        self.replace(".agent-local/tasks/DEV-001.md", "Status: BLOCKED", "Status: DONE")
        self.replace(".agent-local/tasks/DEV-001.md", "Candidate: NONE", "Candidate: " + "a" * 40)
        self.replace(".agent-local/tasks/TEST-002.md", "Status: BACKLOG", "Status: READY")
        self.assertEqual(validate(self.root)[0], [])


if __name__ == "__main__":
    unittest.main()
