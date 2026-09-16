"""Small command-line interface for the research pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from active_ocr.config import load_settings
from active_ocr.integrations.simulation import create_fixture
from active_ocr.models import SimulationConfig, Strategy
from active_ocr.pipeline import Pipeline

app = typer.Typer(
    help="OCR active-learning research pipeline.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
experiment_app = typer.Typer(help="Create and inspect strategy runs.", no_args_is_help=True)
app.add_typer(experiment_app, name="experiment")
simulation_app = typer.Typer(help="Local true-label simulation; fixture model only.")
app.add_typer(simulation_app, name="simulation")
ConfigOption = Annotated[Path | None, typer.Option(help="YAML configuration path.")]


def _pipeline(config: Path | None) -> Pipeline:
    pipeline = Pipeline.from_settings(load_settings(config))
    pipeline.setup()
    return pipeline


@app.command()
def setup(config: ConfigOption = None) -> None:
    """Create the local database and artifact directory."""

    pipeline = _pipeline(config)
    typer.echo(f"Initialized {pipeline.settings.database}")


@app.command(name="import")
def import_pages(
    source: Annotated[Path, typer.Argument(help="JSONL or CSV page manifest.")],
    config: ConfigOption = None,
) -> None:
    """Import pre-rendered page images listed in a manifest."""

    pages = _pipeline(config).import_pages(source)
    typer.echo(f"Imported {len(pages)} page(s)")


@experiment_app.command("start")
def start_experiment(
    name: str,
    strategy: Annotated[Strategy, typer.Option()] = Strategy.RANDOM,
    config: ConfigOption = None,
) -> None:
    """Create one reproducible sampling-strategy run."""

    experiment = _pipeline(config).create_experiment(name, strategy)
    typer.echo(experiment.id)


@experiment_app.command("status")
def experiment_status(experiment_id: str, config: ConfigOption = None) -> None:
    """Print the persisted state of one strategy run."""

    experiment = _pipeline(config).get_experiment(experiment_id)
    typer.echo(json.dumps(experiment.model_dump(mode="json"), indent=2))


@app.command()
def tick(experiment_id: str, config: ConfigOption = None) -> None:
    """Advance one experiment by a single pipeline step."""

    experiment = _pipeline(config).tick(experiment_id)
    typer.echo(f"{experiment.id}: {experiment.stage}")


@simulation_app.command("fixture")
def simulation_fixture(directory: Path) -> None:
    """Generate distinct synthetic page images and labels in a new directory."""
    typer.echo(create_fixture(directory))


@simulation_app.command("start")
def simulation_start(
    manifest: Path,
    directory: Path,
    strategy: Strategy = Strategy.RANDOM,
    batch_size: int = 2,
    page_budget: int = 6,
    rounds: int = 3,
    seed: int = 824,
    report_predictions: bool = False,
) -> None:
    """Freeze a local JSONL source and create a UUID run without training yet."""
    config = SimulationConfig(
        strategy=strategy,
        batch_size=batch_size,
        page_budget=page_budget,
        max_rounds=rounds,
        seed=seed,
        report_predictions=report_predictions,
    )
    run = Pipeline.for_simulation(directory).create_simulation(manifest, config)
    typer.echo(run.id)


@simulation_app.command("run")
@simulation_app.command("resume")
def simulation_run(directory: Path, run_id: str, one_round: bool = False) -> None:
    """Run/resume the fixture simulation from its last committed round."""
    pipeline = Pipeline.for_simulation(directory)
    run = pipeline.step_simulation(run_id) if one_round else pipeline.run_simulation(run_id)
    typer.echo(run.model_dump_json(indent=2))


@simulation_app.command("status")
def simulation_status(directory: Path, run_id: str) -> None:
    """Read the committed simulation record."""
    typer.echo(Pipeline.for_simulation(directory).get_simulation(run_id).model_dump_json(indent=2))


@simulation_app.command("export")
def simulation_export(directory: Path, run_id: str, output: Path) -> None:
    """Write JSON and CSV to a new directory; never overwrite previous exports."""
    Pipeline.for_simulation(directory).export_simulation(run_id, output)
    typer.echo(output)


real_app = typer.Typer(help="Identity-checked real OCR runs with durable Modal recovery.")
app.add_typer(real_app, name="real")


def _real_model(directory, run_id, settings, deployment=None, *, remote=False):
    from active_ocr.integrations.modal_model import (
        DeploymentObservation,
        ModalModel,
        RuntimeSettings,
        SDKTransport,
    )
    from active_ocr.integrations.qwen import strict_json
    from active_ocr.integrations.simulation import LocalOracle, check_local_identity
    from active_ocr.models import RunKind

    pipeline = Pipeline.for_simulation(directory)
    run = pipeline.get_simulation(run_id)
    if run.kind is not RunKind.REAL:
        raise ValueError("real commands require a real run")
    check_local_identity(run.config.real.expected_identity)
    LocalOracle(run.dataset)
    pipeline._validation_pages(run.config, run.dataset.pages)
    runtime = RuntimeSettings.model_validate(strict_json(settings.read_bytes()))
    transport = None
    if remote:
        if deployment is None:
            raise ValueError("an observed deployment file is required")
        observed = DeploymentObservation.model_validate(strict_json(deployment.read_bytes()))
        transport = SDKTransport(runtime, observed)
    model = ModalModel(runtime, pipeline.store, transport)
    model.check_run(run, pipeline.store)
    return pipeline, model


@real_app.command("create")
def real_create(manifest: Path, directory: Path, configuration: Path):
    """Freeze a real configuration with observed CPU-build identity; no remote call."""
    from active_ocr.integrations.qwen import strict_json
    from active_ocr.models import RunKind

    config = SimulationConfig.model_validate(strict_json(configuration.read_bytes()))
    run = Pipeline.for_simulation(directory).create_simulation(manifest, config, kind=RunKind.REAL)
    typer.echo(run.id)


@real_app.command("preflight")
def real_preflight(
    directory: Path,
    run_id: str,
    settings: Path,
    deployment: Path | None = None,
    provider: bool = False,
):
    """Verify local run/settings. --provider additionally checks existing Function/Volume IDs."""
    pipeline, model = _real_model(directory, run_id, settings, deployment, remote=provider)
    if provider:
        model.transport.preflight()
    typer.echo(
        json.dumps(
            dict(
                run_id=run_id,
                local="verified",
                provider="verified" if provider else "not_checked",
                validation_ids=[
                    p.id
                    for p in pipeline._validation_pages(
                        pipeline.get_simulation(run_id).config,
                        pipeline.get_simulation(run_id).dataset.pages,
                    )
                ],
            )
        )
    )


@real_app.command("run")
@real_app.command("resume")
def real_run(
    directory: Path, run_id: str, settings: Path, deployment: Path, one_round: bool = False
):
    """Execute or reattach known work within the original absolute deadline."""
    pipeline, model = _real_model(directory, run_id, settings, deployment, remote=True)
    method = pipeline.step_simulation if one_round else pipeline.run_simulation
    typer.echo(method(run_id, model=model).model_dump_json(indent=2))


@real_app.command("status")
def real_status(directory: Path, run_id: str):
    """Read committed rounds, redacted operation states and the frozen deadline."""
    from active_ocr.integrations.modal_model import ModelOperation, RunControl

    pipeline = Pipeline.for_simulation(directory)
    run = pipeline.get_simulation(run_id)
    control = pipeline.store.load("model-control", run_id, RunControl)
    operations = [
        r.model_dump(mode="json")
        for r in pipeline.store.list("model-operation", ModelOperation)
        if r.semantic["context"]["experiment_id"] == run_id
    ]
    typer.echo(
        json.dumps(
            dict(
                run=run.model_dump(mode="json"),
                control=control.model_dump(mode="json") if control else None,
                operations=operations,
            ),
            indent=2,
        )
    )


@real_app.command("export")
def real_export(directory: Path, run_id: str, output: Path):
    """Export a committed snapshot with explicit validation membership/count."""
    Pipeline.for_simulation(directory).export_simulation(run_id, output)
    typer.echo(output)


@real_app.command("reconcile")
def real_reconcile(
    directory: Path,
    run_id: str,
    settings: Path,
    deployment: Path,
    operation_id: str,
    call_id: str | None = None,
    completion: Path | None = None,
    cancel: bool = False,
):
    """Attach an observed call or verify completion. Never launches/retries a model operation."""
    from active_ocr.integrations.modal_model import (
        REQUEST_ADAPTER,
        FitRequest,
        ModelOperation,
        RemoteExample,
        remote_page,
    )
    from active_ocr.integrations.qwen import strict_json
    from active_ocr.integrations.simulation import LocalOracle

    pipeline, model = _real_model(directory, run_id, settings, deployment, remote=True)
    record = pipeline.store.load("model-operation", operation_id, ModelOperation)
    if record is None or record.semantic["context"]["experiment_id"] != run_id:
        raise ValueError("operation does not belong to this run")
    if cancel:
        if call_id is not None or completion is not None:
            raise ValueError("cancel cannot also attach/complete")
        typer.echo(model.cancel(operation_id).model_dump_json(indent=2))
        return
    semantic = dict(record.semantic)
    if semantic["operation"] == "fit":
        pages = semantic.pop("pages")
        semantic.pop("target_sha256")
        examples = LocalOracle(pipeline.get_simulation(run_id).dataset).reveal(
            tuple(p["id"] for p in pages)
        )
        request = FitRequest(
            **semantic,
            examples=tuple(
                RemoteExample(page=remote_page(e.page), regions=e.regions) for e in examples
            ),
        )
    else:
        request = REQUEST_ADAPTER.validate_python(semantic)
    response = strict_json(completion.read_bytes()) if completion else None
    result = model.reconcile(request, call_id=call_id, response=response)
    typer.echo(result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
