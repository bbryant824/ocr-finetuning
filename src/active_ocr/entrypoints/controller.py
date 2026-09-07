"""Optional polling process that advances active experiments."""

import logging
import time

from active_ocr.config import load_settings
from active_ocr.models import Experiment, Stage
from active_ocr.pipeline import Pipeline

LOGGER = logging.getLogger(__name__)


def run_once(pipeline: Pipeline) -> int:
    advanced = 0
    for experiment in pipeline.store.list("experiment", Experiment):
        if experiment.stage in {Stage.COMPLETE, Stage.ERROR}:
            continue
        try:
            pipeline.tick(experiment.id)
            advanced += 1
        except Exception:
            LOGGER.exception("Could not advance experiment %s", experiment.id)
    return advanced


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = load_settings()
    pipeline = Pipeline.from_settings(settings)
    pipeline.setup()
    while True:
        run_once(pipeline)
        time.sleep(settings.poll_seconds)


if __name__ == "__main__":
    main()
