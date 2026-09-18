"""Offline CLI boundaries; no provider or ML package installation."""

import json

from typer.testing import CliRunner

from active_ocr.entrypoints import cli
from active_ocr.integrations.simulation import create_fixture
from active_ocr.models import SimulationConfig
from active_ocr.pipeline import Pipeline


def test_real_help_lists_complete_operator_commands():
    runner = CliRunner()
    root = runner.invoke(cli.app, ["--help"])
    assert root.exit_code == 0 and "simulation" in root.output and "real" in root.output
    for removed in ("setup", "import", "experiment", "tick"):
        assert runner.invoke(cli.app, [removed]).exit_code == 2
    result = runner.invoke(cli.app, ["real", "--help"])
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


def test_cli_real_create_preflight_reconcile_resume_export_with_fake_transport(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from test_modal_model import CompleteTransport, make_settings

    from active_ocr.integrations import artifacts as a
    from active_ocr.integrations import factory_model, simulation
    from active_ocr.integrations import modal_model as m

    settings = make_settings()
    identity = settings.context.real_config.expected_identity
    monkeypatch.setattr(
        simulation,
        "local_contract_identity",
        lambda: (identity.source_sha, identity.dependency_sha256),
    )
    manifest = create_fixture(tmp_path / "data", train_pages=3)
    cfg = SimulationConfig(
        real=settings.context.real_config,
        backend=factory_model.BACKEND,
        evaluator_id="page-text-nfc-v1",
        max_rounds=1,
        validation_page_ids=("page-3",),
    )
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(cfg.model_dump_json())
    directory = tmp_path / "runs"
    runner = CliRunner()

    def invoke(*args, success=True):
        result = runner.invoke(cli.app, ["real", *map(str, args)])
        if success:
            assert result.exit_code == 0, str(result.exception)
        else:
            assert result.exit_code != 0
        return result.output

    run_id = invoke("create", manifest, directory, cfg_file).strip()
    pipeline = Pipeline.for_simulation(directory)
    run = pipeline.get_simulation(run_id)
    bundle = m.InputBundle(
        files=tuple(
            sorted(
                [f for f in settings.bundle.files if f.filename.startswith("model/")]
                + [
                    a.FileEntry(
                        filename="images/" + p.image_sha256,
                        sha256=p.image_sha256,
                        bytes=Path(p.image_uri).stat().st_size,
                    )
                    for p in run.dataset.pages
                ],
                key=lambda f: f.filename,
            )
        )
    )
    settings = m.RuntimeSettings.model_validate(
        {
            **settings.model_dump(),
            "bundle": bundle.model_dump(),
            "context": {
                **settings.context.model_dump(),
                "source_policy": cfg.source_policy,
                "experiment_id": run_id,
                "bundle_sha256": bundle.sha256,
            },
        }
    )
    settings_file = tmp_path / "runtime.json"
    settings_file.write_bytes(a.canonical(settings.model_dump(mode="json")))
    deployment_file = tmp_path / "deployment.json"
    deployment_file.write_text(
        m.DeploymentObservation(
            settings_sha256=a.digest(settings.model_dump(mode="json")),
            function_id="fu-SYNTHETIC",
            app_id="ap-SYNTHETIC",
        ).model_dump_json()
    )
    transport = CompleteTransport(settings)
    monkeypatch.setattr(m, "SDKTransport", lambda runtime, observed: transport)
    assert (
        json.loads(invoke("preflight", directory, run_id, settings_file))["provider"]
        == "not_checked"
    )
    assert transport.preflights == 0
    transport.lose_ack = True
    invoke("run", directory, run_id, settings_file, deployment_file, "--one-round", success=False)
    state = json.loads(invoke("status", directory, run_id))
    assert state["operations"][0]["state"] == "UNKNOWN" and len(transport.payloads) == 1
    op_id = state["operations"][0]["operation_id"]
    invoke(
        "reconcile", directory, run_id, settings_file, deployment_file, op_id, "--call-id", "fc-1"
    )
    transport.lose_ack = False
    baseline = json.loads(
        invoke("resume", directory, run_id, settings_file, deployment_file, "--one-round")
    )
    assert baseline["baseline"]["labelled_count"] == 0
    done = json.loads(invoke("resume", directory, run_id, settings_file, deployment_file))
    assert done["complete"] and len(transport.payloads) == 4
    invoke("export", directory, run_id, tmp_path / "export")
    assert json.loads((tmp_path / "export/validation.json").read_text()) == dict(
        page_ids=["page-3"], page_count=1
    )
