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

app = typer.Typer(help="OCR active-learning research pipeline.", no_args_is_help=True)
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


if __name__ == "__main__":
    app()
