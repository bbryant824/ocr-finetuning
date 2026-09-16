"""Offline CLI boundaries; no provider or ML package installation."""

import json

from typer.testing import CliRunner

from active_ocr.entrypoints import cli
from active_ocr.integrations.simulation import create_fixture
from active_ocr.models import SimulationConfig
from active_ocr.pipeline import Pipeline


def test_real_help_lists_complete_operator_commands():
    result = CliRunner().invoke(cli.app, ["real", "--help"])
    assert result.exit_code == 0
    for name in ("create", "preflight", "run", "resume", "status", "export", "reconcile"):
        assert name in result.output


def test_real_preflight_rejects_fixture_without_loading_sdk(tmp_path):
    manifest = create_fixture(tmp_path / "data")
    pipeline = Pipeline.for_simulation(tmp_path / "runs")
    run = pipeline.create_simulation(manifest, SimulationConfig())
    result = CliRunner().invoke(
        cli.app,
        [
            "real",
            "preflight",
            str(tmp_path / "runs"),
            run.id,
            str(tmp_path / "missing-settings.json"),
        ],
    )
    assert result.exit_code != 0
    assert "require a real run" in str(result.exception)
    assert pipeline.get_simulation(run.id) == run


def test_real_status_and_export_are_local_and_preserve_validation_membership(tmp_path):
    manifest = create_fixture(tmp_path / "data")
    pipeline = Pipeline.for_simulation(tmp_path / "runs")
    run = pipeline.create_simulation(manifest, SimulationConfig())
    result = CliRunner().invoke(cli.app, ["real", "status", str(tmp_path / "runs"), run.id])
    assert result.exit_code == 0
    state = json.loads(result.output)
    assert state["control"] is None and state["operations"] == []
    export = tmp_path / "export"
    result = CliRunner().invoke(
        cli.app, ["real", "export", str(tmp_path / "runs"), run.id, str(export)]
    )
    assert result.exit_code == 0
    assert json.loads((export / "validation.json").read_text())["page_count"] == 1
