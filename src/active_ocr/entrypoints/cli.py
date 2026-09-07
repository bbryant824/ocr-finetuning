"""Small command-line interface for the research pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from active_ocr.config import load_settings
from active_ocr.models import Strategy
from active_ocr.pipeline import Pipeline

app = typer.Typer(help="OCR active-learning research pipeline.", no_args_is_help=True)
experiment_app = typer.Typer(help="Create and inspect strategy runs.", no_args_is_help=True)
app.add_typer(experiment_app, name="experiment")
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


if __name__ == "__main__":
    app()
