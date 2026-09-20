"""Small command-line interface for the research pipeline."""

from __future__ import annotations

from pathlib import Path

import typer

from active_ocr.integrations.simulation import create_fixture
from active_ocr.models import SimulationConfig, Strategy
from active_ocr.pipeline import Pipeline

app = typer.Typer(
    help="OCR active-learning research pipeline.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
simulation_app = typer.Typer(help="Local true-label simulation; fixture model only.")
app.add_typer(simulation_app, name="simulation")


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
    initial_batch_size: int = 0,
    page_budget: int = 6,
    rounds: int = 3,
    seed: int = 824,
    report_predictions: bool = False,
) -> None:
    """Freeze a local JSONL source and create a UUID run without training yet."""
    config = SimulationConfig(
        strategy=strategy,
        batch_size=batch_size,
        initial_batch_size=initial_batch_size,
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
