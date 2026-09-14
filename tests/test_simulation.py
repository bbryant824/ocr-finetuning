"""Behavioral checks of the local oracle loop, never real OCR evaluation."""

import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from active_ocr.entrypoints.cli import app
from active_ocr.integrations.simulation import FixtureModel, LocalOracle, create_fixture, sha256
from active_ocr.models import SimulationConfig, Split, Strategy
from active_ocr.pipeline import Pipeline


def setup_run(tmp_path, config=None, train_pages=7):
    manifest = create_fixture(tmp_path / "data", train_pages)
    pipeline = Pipeline.for_simulation(tmp_path / "runs")
    run = pipeline.create_simulation(manifest, config or SimulationConfig())
    return pipeline, run, manifest


class SpyModel(FixtureModel):
    def __init__(self):
        self.fits = []
        self.pools = []

    def fit(self, examples, *, seed):
        self.fits.append(examples)
        assert all(e.page.split is Split.TRAIN for e in examples)
        return super().fit(examples, seed=seed)

    def predict(self, pages, **kwargs):
        assert all(p.split is Split.TRAIN and not hasattr(p, "regions") for p in pages)
        self.pools.append(pages)
        return super().predict(pages, **kwargs)


def test_rounds_oracle_fit_and_heldout_isolation(tmp_path, monkeypatch):
    pipeline, run, _ = setup_run(tmp_path, SimulationConfig(strategy=Strategy.ENTROPY))
    requested = []
    original = LocalOracle.reveal

    def reveal(self, ids):
        requested.append(ids)
        return original(self, ids)

    monkeypatch.setattr(LocalOracle, "reveal", reveal)
    model = SpyModel()
    done = pipeline.run_simulation(run.id, model)
    assert [r.labelled_count for r in done.rounds] == [2, 4, 6]
    assert done.complete and done.stop_reason == "page_budget"
    assert len(set(done.rounds[-1].revealed_ids)) == 6
    for i, record in enumerate(done.rounds):
        assert requested[i] == record.revealed_ids
        assert tuple(e.page.id for e in model.fits[i]) == record.revealed_ids
        assert {p.id for p in model.pools[i]}.isdisjoint(record.revealed_ids)
        assert record.annotation_seconds is None
        assert all(e.seconds is None for e in model.fits[i])
        assert all(p.regions == () for p in record.predictions)
    oracle = LocalOracle(run.dataset)
    for page in run.dataset.pages:
        if page.split is not Split.TRAIN:
            with pytest.raises(ValueError, match="train"):
                oracle.reveal((page.id,))


@pytest.mark.parametrize("strategy", list(Strategy))
def test_reproducible_first_batch_and_resume(tmp_path, strategy):
    config = SimulationConfig(strategy=strategy, page_budget=5, max_rounds=10, seed=13)
    pipeline, run, manifest = setup_run(tmp_path, config)
    first = pipeline.step_simulation(run.id)
    assert first.rounds[0].labelled_count == 2
    restarted = Pipeline.for_simulation(tmp_path / "runs")
    finished = restarted.run_simulation(run.id)
    duplicate = restarted.create_simulation(manifest, config)
    repeated = restarted.run_simulation(duplicate.id)
    assert [r.selected_ids for r in finished.rounds] == [r.selected_ids for r in repeated.rounds]
    assert [r.model_id for r in finished.rounds] == [r.model_id for r in repeated.rounds]
    assert [len(r.selected_ids) for r in finished.rounds] == [2, 2, 1]
    random = restarted.create_simulation(manifest, SimulationConfig(seed=13))
    assert (
        restarted.step_simulation(random.id).rounds[0].selected_ids == first.rounds[0].selected_ids
    )
    assert restarted.run_simulation(run.id) == finished
    assert (
        restarted.get_simulation(duplicate.id).rounds[0].predictions
        == repeated.rounds[0].predictions
    )
    for r in repeated.rounds:
        assert all(p.experiment_id == duplicate.id for p in r.predictions)


@pytest.mark.parametrize(
    "train,budget,rounds,counts,reason",
    [
        (3, 9, 9, [2, 3], "pool_exhausted"),
        (7, 9, 1, [2], "round_limit"),
        (0, 9, 9, [], "pool_exhausted"),
        (7, 0, 9, [], "page_budget"),
        (7, 9, 0, [], "round_limit"),
    ],
)
def test_stop_boundaries(tmp_path, train, budget, rounds, counts, reason):
    pipeline, run, _ = setup_run(
        tmp_path, SimulationConfig(page_budget=budget, max_rounds=rounds), train
    )
    done = pipeline.run_simulation(run.id)
    assert [r.labelled_count for r in done.rounds] == counts
    assert done.stop_reason == reason


def test_random_does_not_require_predictions(tmp_path):
    class FitOnly(FixtureModel):
        def predict(self, *args, **kwargs):
            raise AssertionError("random should not score")

    pipeline, run, _ = setup_run(tmp_path)
    assert pipeline.run_simulation(run.id, FitOnly()).complete


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "extra", "run", "round", "model", "score"]
)
def test_bad_predictions_do_not_commit_and_can_retry(tmp_path, fault):
    class BadModel(FixtureModel):
        def predict(self, pages, **kwargs):
            predictions = super().predict(pages, **kwargs)
            if fault == "missing":
                return predictions[:-1]
            if fault == "duplicate":
                return (*predictions, predictions[0])
            fields = {
                "extra": {"page_id": "outside"},
                "run": {"experiment_id": "other"},
                "round": {"round_number": 99},
                "model": {"model_id": "other"},
                "score": {"entropy": None},
            }
            return (predictions[0].model_copy(update=fields[fault]), *predictions[1:])

    pipeline, run, _ = setup_run(tmp_path, SimulationConfig(strategy=Strategy.ENTROPY))
    with pytest.raises(ValueError):
        pipeline.step_simulation(run.id, BadModel())
    assert pipeline.get_simulation(run.id) == run
    done = pipeline.run_simulation(run.id)
    assert [r.labelled_count for r in done.rounds] == [2, 4, 6]


def test_fit_failure_and_failed_commit_recover_without_counting_twice(tmp_path, monkeypatch):
    class BrokenFit(FixtureModel):
        def fit(self, examples, **kwargs):
            raise RuntimeError("interrupted fit")

    pipeline, run, _ = setup_run(tmp_path)
    with pytest.raises(RuntimeError, match="fit"):
        pipeline.step_simulation(run.id, BrokenFit())
    assert pipeline.get_simulation(run.id) == run
    commit = pipeline.store.compare_and_swap

    def fail(*args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(pipeline.store, "compare_and_swap", fail)
    with pytest.raises(OSError):
        pipeline.step_simulation(run.id)
    assert pipeline.get_simulation(run.id) == run
    monkeypatch.setattr(pipeline.store, "compare_and_swap", commit)
    assert pipeline.step_simulation(run.id).rounds[0].labelled_count == 2


def test_stale_writer_cannot_double_commit(tmp_path):
    pipeline, run, _ = setup_run(tmp_path)
    other = Pipeline.for_simulation(tmp_path / "runs")

    class ConcurrentFit(FixtureModel):
        def fit(self, examples, **kwargs):
            other.step_simulation(run.id)
            return super().fit(examples, **kwargs)

    with pytest.raises(RuntimeError, match="concurrently"):
        pipeline.step_simulation(run.id, ConcurrentFit())
    assert len(pipeline.get_simulation(run.id).rounds) == 1
    assert pipeline.run_simulation(run.id).rounds[-1].labelled_count == 6


def read_rows(manifest):
    return [json.loads(line) for line in manifest.read_text().splitlines()]


def write_rows(manifest, rows):
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize(
    "fault",
    [
        "id",
        "split",
        "document",
        "duplicate_image",
        "bounds",
        "dimensions",
        "hash",
        "invalid_image",
        "region_id",
    ],
)
def test_dataset_validation(tmp_path, fault):
    manifest = create_fixture(tmp_path / "data")
    rows = read_rows(manifest)
    if fault == "id":
        rows[1]["id"] = rows[0]["id"]
    elif fault == "split":
        rows[0]["split"] = "mystery"
    elif fault == "document":
        rows[-1]["document_id"] = rows[0]["document_id"]
    elif fault == "duplicate_image":
        rows[-1]["image_uri"] = rows[0]["image_uri"]
        rows[-1]["image_sha256"] = rows[0]["image_sha256"]
    elif fault == "bounds":
        rows[0]["regions"][0]["box"]["width"] = 999
    elif fault == "dimensions":
        rows[0]["width"] = 10
    elif fault == "hash":
        rows[0]["image_sha256"] = "0" * 64
    elif fault == "region_id":
        rows[0]["regions"].append(rows[0]["regions"][0])
    else:
        image = manifest.parent / rows[0]["image_uri"]
        image.write_bytes(b"not an image")
        rows[0]["image_sha256"] = sha256(image.read_bytes())
    write_rows(manifest, rows)
    pipeline = Pipeline.for_simulation(tmp_path / "runs")
    with pytest.raises((ValueError, OSError)):
        pipeline.create_simulation(manifest, SimulationConfig())


def test_missing_selected_labels_rejected_and_empty_labels_preserved(tmp_path):
    manifest = create_fixture(tmp_path / "data", train_pages=1)
    rows = read_rows(manifest)
    rows[0].pop("regions")
    write_rows(manifest, rows)
    pipeline = Pipeline.for_simulation(tmp_path / "runs")
    run = pipeline.create_simulation(manifest, SimulationConfig())
    with pytest.raises(ValueError, match="missing ground truth"):
        pipeline.step_simulation(run.id)
    assert pipeline.get_simulation(run.id).rounds == ()
    rows[0]["regions"] = []  # Explicit blank page is different from missing annotation.
    write_rows(manifest, rows)
    blank = pipeline.create_simulation(manifest, SimulationConfig())
    assert pipeline.run_simulation(blank.id).rounds[0].labelled_count == 1
    rows[0]["regions"] = [
        {
            "id": "line",
            "box": {"x": 0, "y": 0, "width": 10, "height": 10},
            "text": "",
            "illegible": True,
        }
    ]
    write_rows(manifest, rows)
    original = pipeline.create_simulation(manifest, SimulationConfig())
    assert LocalOracle(original.dataset).reveal(("page-0",))[0].regions[0].text == ""


@pytest.mark.parametrize("fault", ["manifest", "labels", "source_image", "frozen_image", "config"])
def test_changed_inputs_cannot_resume(tmp_path, fault):
    pipeline, run, manifest = setup_run(tmp_path)
    first = pipeline.step_simulation(run.id)
    config = None
    if fault == "manifest":
        manifest.write_text(manifest.read_text() + "\n")
    elif fault == "labels":
        rows = read_rows(manifest)
        rows[0]["regions"][0]["text"] = "changed"
        write_rows(manifest, rows)
    elif fault in {"source_image", "frozen_image"}:
        page = run.dataset.pages[0]
        Path(page.source_image if fault == "source_image" else page.image_uri).write_bytes(
            b"changed"
        )
    else:
        config = run.config.model_copy(update={"seed": 999})
    restarted = Pipeline.for_simulation(tmp_path / "runs")
    with pytest.raises(ValueError, match="changed|change"):
        restarted.step_simulation(run.id, config=config)
    assert restarted.get_simulation(run.id) == first


def test_mutation_during_round_and_backend_change_rejected(tmp_path):
    pipeline, run, manifest = setup_run(tmp_path)

    class MutatesSource(FixtureModel):
        def fit(self, examples, **kwargs):
            manifest.write_text(manifest.read_text() + "\n")
            return super().fit(examples, **kwargs)

    class OtherBackend(FixtureModel):
        backend = "other-v1"

    with pytest.raises(ValueError, match="backend"):
        pipeline.step_simulation(run.id, OtherBackend())
    with pytest.raises(ValueError, match="changed"):
        pipeline.step_simulation(run.id, MutatesSource())
    assert pipeline.get_simulation(run.id) == run


def test_cli_fixture_start_run_resume_status_export(tmp_path):
    runner = CliRunner()

    def invoke(*args):
        result = runner.invoke(app, ["simulation", *map(str, args)])
        assert result.exit_code == 0, result.output + str(result.exception)
        return result.output

    manifest = Path(invoke("fixture", tmp_path / "data").strip())
    run_dir = tmp_path / "runs"
    run_id = invoke("start", manifest, run_dir, "--strategy", "entropy").strip()
    first = json.loads(invoke("run", run_dir, run_id, "--one-round"))
    assert len(first["rounds"]) == 1
    done = json.loads(invoke("resume", run_dir, run_id))
    assert done["complete"]
    assert json.loads(invoke("status", run_dir, run_id)) == done
    invoke("export", run_dir, run_id, tmp_path / "export")
    assert json.loads((tmp_path / "export/results.json").read_text()) == done
    with (tmp_path / "export/rounds.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 and rows[-1]["labelled_pages"] == "6"
    assert rows[0]["annotation_seconds"] == ""
    assert not (tmp_path / "state.sqlite3").exists()
    assert (
        runner.invoke(
            app, ["simulation", "export", str(run_dir), run_id, str(tmp_path / "export")]
        ).exit_code
        != 0
    )


def test_hidden_truth_and_input_order_cannot_seed_acquisition(tmp_path):
    config = SimulationConfig(strategy=Strategy.ENTROPY, batch_size=3, page_budget=5)
    pipeline, first, manifest = setup_run(tmp_path, config)
    done = pipeline.run_simulation(first.id)
    acquired = set(done.rounds[-1].revealed_ids)
    rows = read_rows(manifest)
    for row in rows:
        if row["id"] not in acquired:
            row["regions"][0]["text"] = "changed unseen truth 雪"
    write_rows(manifest, list(reversed(rows)))
    second = pipeline.create_simulation(manifest, config)
    changed = pipeline.run_simulation(second.id)
    assert first.dataset.ground_truth_sha256 != second.dataset.ground_truth_sha256
    assert [r.selected_ids for r in done.rounds] == [r.selected_ids for r in changed.rounds]
    assert [len(r.selected_ids) for r in changed.rounds] == [3, 2]
    assert [r.model_id for r in done.rounds] == [r.model_id for r in changed.rounds]
    assert [[(p.page_id, p.confidence, p.entropy) for p in r.predictions] for r in done.rounds] == [
        [(p.page_id, p.confidence, p.entropy) for p in r.predictions] for r in changed.rounds
    ]


def test_interleaved_runs_with_shared_ids_have_separate_labels_and_pools(tmp_path):
    config = SimulationConfig(strategy=Strategy.ENTROPY, batch_size=3, page_budget=5)
    pipeline, first, _ = setup_run(tmp_path, config)
    other_manifest = create_fixture(tmp_path / "other", train_pages=4)
    rows = read_rows(other_manifest)
    for row in rows:
        row["regions"][0]["text"] = "other source truth"
    write_rows(other_manifest, rows)
    other = pipeline.create_simulation(other_manifest, config)
    one = pipeline.step_simulation(first.id)
    two = pipeline.step_simulation(other.id)
    assert one.rounds[0].model_id != two.rounds[0].model_id
    first_done = pipeline.run_simulation(first.id)
    other_done = pipeline.run_simulation(other.id)
    assert [len(r.selected_ids) for r in other_done.rounds] == [3, 1]
    assert first_done.rounds[-1].labelled_count == 5
    for run in (first_done, other_done):
        assert all(p.experiment_id == run.id for r in run.rounds for p in r.predictions)


@pytest.mark.parametrize("bad_score", [-0.5, 1.2, float("nan"), float("inf")])
def test_invalid_scores_are_revalidated_at_adapter_boundary(tmp_path, bad_score):
    pipeline, run, _ = setup_run(tmp_path, SimulationConfig(strategy=Strategy.ENTROPY))

    class BadScores(FixtureModel):
        def predict(self, *args, **kwargs):
            return tuple(
                p.model_copy(update={"entropy": bad_score})
                for p in super().predict(*args, **kwargs)
            )

    with pytest.raises(ValueError):
        pipeline.step_simulation(run.id, BadScores())
    assert pipeline.get_simulation(run.id) == run


def test_failure_after_commit_resumes_from_committed_round(tmp_path, monkeypatch):
    pipeline, run, _ = setup_run(tmp_path)
    save = pipeline.store.compare_and_swap

    def commit_then_fail(*args):
        save(*args)
        raise RuntimeError("process lost response after commit")

    monkeypatch.setattr(pipeline.store, "compare_and_swap", commit_then_fail)
    with pytest.raises(RuntimeError):
        pipeline.step_simulation(run.id)
    restarted = Pipeline.for_simulation(tmp_path / "runs")
    assert len(restarted.get_simulation(run.id).rounds) == 1
    done = restarted.run_simulation(run.id)
    assert len(done.rounds) == 3 and len(set(done.rounds[-1].revealed_ids)) == 6


def test_optional_validation_hook_isolated_and_not_used_for_acquisition(tmp_path):
    class Evaluator:
        identifier = "validation-count-test-v1"

        def __init__(self):
            self.calls = []

        def __call__(self, examples, predictions):
            self.calls.append(examples)
            assert all(e.page.split is Split.VALIDATION for e in examples)
            assert {e.page.id for e in examples} == {p.page_id for p in predictions}
            return {"fixture_validation_pages": len(examples)}

    class Model(FixtureModel):
        def fit(self, examples, **kwargs):
            assert all(e.page.split is Split.TRAIN for e in examples)
            return super().fit(examples, **kwargs)

        def predict(self, pages, **kwargs):
            assert all(p.split is not Split.TEST and not hasattr(p, "regions") for p in pages)
            return super().predict(pages, **kwargs)

    config = SimulationConfig(strategy=Strategy.ENTROPY, evaluator_id=Evaluator.identifier)
    pipeline, run, manifest = setup_run(tmp_path, config)
    evaluator = Evaluator()
    with pytest.raises(ValueError, match="evaluator"):
        pipeline.step_simulation(run.id)
    done = pipeline.run_simulation(run.id, Model(), evaluator=evaluator)
    assert len(evaluator.calls) == 3
    assert all(r.validation_metrics == {"fixture_validation_pages": 1} for r in done.rounds)
    plain = pipeline.create_simulation(manifest, config.model_copy(update={"evaluator_id": None}))
    no_eval = pipeline.run_simulation(plain.id)
    assert all(
        r.validation_metrics is None and not r.validation_predictions for r in no_eval.rounds
    )
    assert [r.selected_ids for r in done.rounds] == [r.selected_ids for r in no_eval.rounds]


def test_validation_failure_is_atomic(tmp_path):
    class BrokenEvaluator:
        identifier = "broken-test-v1"

        def __call__(self, examples, predictions):
            return {"bad": float("nan")}

    pipeline, run, _ = setup_run(
        tmp_path, SimulationConfig(evaluator_id=BrokenEvaluator.identifier)
    )
    with pytest.raises(ValueError, match="finite"):
        pipeline.step_simulation(run.id, evaluator=BrokenEvaluator())
    assert pipeline.get_simulation(run.id) == run
