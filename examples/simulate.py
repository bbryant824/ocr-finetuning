"""Run the no-service fixture example in a new directory: python examples/simulate.py PATH."""

import argparse
import json
from pathlib import Path

from active_ocr.integrations.simulation import create_fixture
from active_ocr.models import SimulationConfig, Strategy
from active_ocr.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    manifest = create_fixture(args.directory / "source")
    pipeline = Pipeline.for_simulation(args.directory / "runs")
    run = pipeline.create_simulation(
        manifest,
        SimulationConfig(strategy=Strategy.ENTROPY, batch_size=3, page_budget=5, max_rounds=3),
    )
    pipeline.step_simulation(run.id)
    # Reopen the database as a resumed caller, with a fresh fixture model.
    resumed = Pipeline.for_simulation(args.directory / "runs")
    finished = resumed.run_simulation(run.id)
    resumed.export_simulation(run.id, args.directory / "export")
    print(
        json.dumps(
            {
                "kind": finished.kind,
                "run_id": run.id,
                "batches": [r.selected_ids for r in finished.rounds],
                "labelled_pages": finished.rounds[-1].labelled_count,
                "stop_reason": finished.stop_reason,
                "results": str(args.directory / "export"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
