"""Synthetic adapter acceptance for local contracts; no model or OCR evidence."""

import csv
import json
import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

import active_ocr.integrations.simulation as integration
import active_ocr.pipeline as pipeline_module
from active_ocr.integrations.simulation import FixtureModel, LocalOracle, create_fixture
from active_ocr.models import (
    Box,
    ExecutionTelemetry,
    ExpectedIdentity,
    Prediction,
    PredictionPurpose,
    PredictionStatus,
    RealOCRConfig,
    Region,
    RunKind,
    SimulationConfig,
    SimulationRound,
    SimulationRun,
    SourceRegion,
    Split,
)
from active_ocr.pipeline import Pipeline

SOURCE = "a" * 40
DEPENDENCIES = "b" * 64
CHECKPOINT = "checkpoint:sha256:" + "c" * 64


def recipe():
    identity = ExpectedIdentity(
        source_sha=SOURCE,
        dependency_sha256=DEPENDENCIES,
        model_revision="d" * 40,
        model_manifest_sha256="e" * 64,
        processor_revision="f" * 40,
        processor_manifest_sha256="1" * 64,
        recipe_version="synthetic-v1",
        evaluator_id="page-text-nfc-v1",
    )
    return RealOCRConfig(
        backend="synthetic-contract-v1",
        recipe_version="synthetic-v1",
        model_repository="synthetic/model",
        model_revision=identity.model_revision,
        processor_repository="synthetic/processor",
        processor_revision=identity.processor_revision,
        training_policy_id="synthetic-training-v1",
        decode_policy_id="synthetic-decode-v1",
        expected_identity=identity,
    )


def config(**changes):
    real = recipe()
    return SimulationConfig(
        real=real, backend=real.backend, evaluator_id=real.evaluator_id, **changes
    )


class Adapter:
    kind = RunKind.CONTRACT_TEST
    fit_policy = "reset-fit-cumulative-v1"

    def __init__(self, real=None):
        self.real_config = real or recipe()
        self.backend = self.real_config.backend
        self.identity = self.real_config.expected_identity
        self.calls = []
        self.telemetry = None

    def load_base(self, *, experiment_id):
        self.calls.append(("base", experiment_id))
        return CHECKPOINT

    def fit(self, examples, *, seed, experiment_id, round_number):
        assert all(e.page.split is Split.TRAIN for e in examples)
        self.calls.append(("fit", experiment_id, round_number, seed, examples))
        return CHECKPOINT

    def predict(
        self, pages, *, experiment_id, round_number, model_id, purpose=PredictionPurpose.POOL
    ):
        assert all(p.split is Split.VALIDATION for p in pages)  # No pool/test predictions.
        assert all(not hasattr(p, "regions") for p in pages)
        self.calls.append(("predict", experiment_id, round_number, purpose, pages))
        return tuple(
            Prediction(
                page_id=p.id,
                experiment_id=experiment_id,
                round_number=round_number,
                model_id=model_id,
                purpose=purpose,
            )
            for p in pages
        )


@pytest.fixture(autouse=True)
def synthetic_identity(monkeypatch):
    # Explicitly synthetic identity seam; never claim these are actual source/model hashes.
    monkeypatch.setattr(integration, "local_contract_identity", lambda: (SOURCE, DEPENDENCIES))


def setup(tmp_path, *, train=5, **changes):
    manifest = create_fixture(tmp_path / "data", train_pages=train)
    pipeline = Pipeline.for_simulation(tmp_path / "runs")
    run = pipeline.create_simulation(manifest, config(**changes), kind=RunKind.CONTRACT_TEST)
    return pipeline, run, manifest


def test_creation_and_baseline_never_select_reveal_or_fit(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("baseline accessed acquisition or training")

    monkeypatch.setattr(LocalOracle, "reveal", forbidden)
    monkeypatch.setattr(pipeline_module, "select_pages", forbidden)
    monkeypatch.setattr(Adapter, "fit", forbidden)
    model = Adapter()
    pipeline, run, _ = setup(tmp_path)
    assert model.calls == [] and run.baseline is None and not run.complete
    baseline = pipeline.step_simulation(run.id, model)
    assert baseline.rounds == () and not baseline.complete
    assert baseline.baseline.labelled_count == 0 and baseline.baseline.telemetry is None
    assert [call[0] for call in model.calls] == ["base", "predict"]
    assert model.calls[0][1] == run.id
    assert model.calls[1][1:4] == (run.id, 0, PredictionPurpose.BASELINE_VALIDATION)


@pytest.mark.parametrize(
    "train,budget,rounds,reason",
    [(5, 0, 3, "page_budget"), (5, 3, 0, "round_limit"), (0, 3, 3, "pool_exhausted")],
)
def test_zero_paths_baseline_once_then_no_calls(tmp_path, train, budget, rounds, reason):
    pipeline, run, _ = setup(tmp_path, train=train, page_budget=budget, max_rounds=rounds)
    model = Adapter()
    done = pipeline.step_simulation(run.id, model)
    assert done.baseline and not done.rounds and done.complete and done.stop_reason == reason
    model.calls.clear()
    model.telemetry = ExecutionTelemetry(
        elapsed_seconds=999, peak_memory_bytes=123, device="changed"
    )
    assert pipeline.step_simulation(run.id, model) == done
    assert model.calls == []


@pytest.mark.parametrize("stage", ["load", "predict", "evaluate", "pre_cas", "post_cas"])
@pytest.mark.parametrize("baseline", [True, False])
def test_failure_atomicity_and_response_loss(tmp_path, monkeypatch, stage, baseline):
    if not baseline and stage == "load":
        stage = "fit"
    pipeline, run, _ = setup(tmp_path)
    model = Adapter()
    if not baseline:
        run = pipeline.step_simulation(run.id, model)
    original_cas = pipeline.store.compare_and_swap

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    evaluator = None
    with monkeypatch.context() as patch:
        if stage in ("load", "fit", "predict"):
            patch.setattr(model, {"load": "load_base"}.get(stage, stage), fail)
        elif stage == "evaluate":

            class FailingEvaluator:
                identifier = "page-text-nfc-v1"
                __call__ = fail

            evaluator = FailingEvaluator()
        else:

            def cas(*args):
                if stage == "post_cas":
                    original_cas(*args)
                fail()

            patch.setattr(pipeline.store, "compare_and_swap", cas)
        with pytest.raises(RuntimeError, match="injected"):
            pipeline.step_simulation(run.id, model, evaluator=evaluator)
    saved = pipeline.get_simulation(run.id)
    if stage == "post_cas":
        assert bool(saved.baseline) and len(saved.rounds) == (0 if baseline else 1)
        # Reload is sufficient; a later step legitimately starts the next acquisition.
        assert saved != run
    else:
        assert saved == run
        retried = pipeline.step_simulation(run.id, model)
        assert retried.baseline and len(retried.rounds) == (0 if baseline else 1)


def test_two_baseline_callers_only_one_commit(tmp_path):
    pipeline, run, _ = setup(tmp_path, page_budget=0)
    barrier = Barrier(2)

    class Racing(Adapter):
        def load_base(self, **kwargs):
            result = super().load_base(**kwargs)
            barrier.wait(timeout=10)
            return result

    def attempt():
        try:
            return pipeline.step_simulation(run.id, Racing())
        except RuntimeError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: attempt(), range(2)))
    assert sum(isinstance(result, SimulationRun) for result in results) == 1
    assert sum(isinstance(result, RuntimeError) for result in results) == 1
    saved = pipeline.get_simulation(run.id)
    assert saved.baseline and saved.complete and saved.rounds == ()
    assert pipeline.step_simulation(run.id, Adapter()) == saved


@pytest.mark.parametrize(
    "fault",
    [
        "run",
        "model",
        "round",
        "purpose",
        "missing",
        "extra",
        "duplicate",
        "infinite",
        "negative",
        "zero",
        "bounds",
        "region_id",
        "checkpoint",
    ],
)
@pytest.mark.parametrize("baseline", [True, False])
def test_boundary_failures_leave_state_unchanged(tmp_path, fault, baseline):
    pipeline, run, _ = setup(tmp_path)
    if not baseline:
        run = pipeline.step_simulation(run.id, Adapter())

    class Bad(Adapter):
        def load_base(self, **kwargs):
            return "mutable-name" if fault == "checkpoint" else super().load_base(**kwargs)

        def fit(self, examples, **kwargs):
            return "mutable-name" if fault == "checkpoint" else super().fit(examples, **kwargs)

        def predict(self, pages, **kwargs):
            preds = super().predict(pages, **kwargs)
            first = preds[0]
            if fault == "missing":
                return ()
            if fault == "duplicate":
                return (*preds, first)
            if fault == "extra":
                return (*preds, first.model_copy(update={"page_id": "extra"}))
            update = {
                "run": {"experiment_id": "wrong"},
                "model": {"model_id": "wrong"},
                "round": {"round_number": 7},
                "purpose": {"purpose": PredictionPurpose.POOL},
            }
            if fault in update:
                return (first.model_copy(update=update[fault]),)
            box = Box(x=0, y=0, width=1, height=1)
            values = {
                "infinite": {"width": float("inf")},
                "negative": {"x": -1},
                "zero": {"height": 0},
                "bounds": {"width": pages[0].width + 1},
            }
            if fault in values:
                box = box.model_copy(update=values[fault])
            region = Region(id="line", box=box, text="text")
            regions = (region, region) if fault == "region_id" else (region,)
            return (first.model_copy(update={"regions": regions}),)

    with pytest.raises(ValueError):
        pipeline.step_simulation(run.id, Bad())
    assert pipeline.get_simulation(run.id) == run


def test_output_order_and_positive_round_constraints(tmp_path):
    pipeline, run, _ = setup(tmp_path)
    # Direct boundary: two requested metadata records in an explicit order.
    ids = tuple(p.id for p in run.dataset.pages[:2])
    predictions = tuple(
        Prediction(
            page_id=p,
            experiment_id=run.id,
            model_id=CHECKPOINT,
            round_number=0,
            purpose=PredictionPurpose.BASELINE_VALIDATION,
        )
        for p in ids
    )
    ordered = pipeline._validate_simulation_predictions(
        predictions[::-1],
        ids,
        run,
        0,
        CHECKPOINT,
        require_scores=False,
        purpose=PredictionPurpose.BASELINE_VALIDATION,
    )
    assert tuple(p.page_id for p in ordered) == ids
    for purpose, number in [
        (PredictionPurpose.BASELINE_VALIDATION, 1),
        (PredictionPurpose.POOL, 0),
        (PredictionPurpose.VALIDATION, 0),
    ]:
        with pytest.raises(ValueError):
            Prediction(
                page_id="p", experiment_id="e", model_id="m", round_number=number, purpose=purpose
            )
    for number in [0, -1]:
        with pytest.raises(ValueError):
            SimulationRound(
                number=number,
                selected_ids=(),
                revealed_ids=(),
                labelled_count=0,
                remaining_count=0,
                model_id="m",
            )


@pytest.mark.parametrize(
    "train,budget,rounds,counts", [(5, 3, 9, [2, 3]), (3, 8, 9, [2, 3]), (5, 8, 1, [2])]
)
def test_sampling_seed_cumulative_training_and_no_pool(tmp_path, train, budget, rounds, counts):
    pipeline, run, manifest = setup(tmp_path, train=train, page_budget=budget, max_rounds=rounds)
    model = Adapter()
    done = pipeline.run_simulation(run.id, model)
    fixture = pipeline.create_simulation(
        manifest, SimulationConfig(page_budget=budget, max_rounds=rounds, seed=run.config.seed)
    )
    fixture = pipeline.run_simulation(fixture.id)
    assert [r.selected_ids for r in done.rounds] == [r.selected_ids for r in fixture.rounds]
    assert [r.labelled_count for r in done.rounds] == counts
    fits = [c for c in model.calls if c[0] == "fit"]
    assert [c[2] for c in fits] == list(range(1, len(counts) + 1))
    assert all(c[1] == run.id and c[3] == run.config.seed for c in fits)
    for call, record in zip(fits, done.rounds, strict=True):
        assert tuple(e.page.id for e in call[4]) == record.revealed_ids
        assert len(set(record.revealed_ids)) == record.labelled_count
        assert record.predictions == ()
    assert all(
        p.confidence is None and p.entropy is None
        for r in done.rounds
        for p in r.validation_predictions
    )


@pytest.mark.parametrize("field", list(ExpectedIdentity.model_fields))
def test_all_adapter_identity_mismatches_fail_before_calls(tmp_path, field):
    pipeline, run, _ = setup(tmp_path)
    model = Adapter()
    previous = getattr(model.identity, field)
    replacement = "0" * len(previous) if isinstance(previous, str) else "0" * 64
    model.identity = model.identity.model_copy(update={field: replacement})
    with pytest.raises(ValueError, match="identity mismatch"):
        pipeline.step_simulation(run.id, model)
    assert model.calls == [] and pipeline.get_simulation(run.id) == run


@pytest.mark.parametrize(
    "field",
    [
        "training_policy_id",
        "decode_policy_id",
        "evaluator_id",
        "model_repository",
        "processor_repository",
        "backend",
        "recipe_version",
    ],
)
def test_recipe_mismatch_fails_before_calls(tmp_path, field):
    pipeline, run, _ = setup(tmp_path)
    model = Adapter()
    model.real_config = model.real_config.model_copy(update={field: "changed"})
    with pytest.raises(ValueError):
        pipeline.step_simulation(run.id, model)
    assert model.calls == []


@pytest.mark.parametrize(
    "changed", [("f" * 40, DEPENDENCIES), (SOURCE, "f" * 64), (SOURCE + "-dirty", DEPENDENCIES)]
)
def test_local_identity_mutations_even_on_completed_resume(tmp_path, monkeypatch, changed):
    pipeline, run, manifest = setup(tmp_path, page_budget=0)
    done = pipeline.step_simulation(run.id, Adapter())
    model = Adapter()
    monkeypatch.setattr(integration, "local_contract_identity", lambda: changed)
    with pytest.raises(ValueError, match="code/dependency"):
        pipeline.create_simulation(manifest, config(), kind=RunKind.CONTRACT_TEST)
    with pytest.raises(ValueError, match="code/dependency"):
        pipeline.step_simulation(run.id, model)
    assert model.calls == [] and pipeline.get_simulation(run.id) == done


def test_explicit_kinds_adapter_and_validation_required(tmp_path):
    pipeline, run, manifest = setup(tmp_path)
    with pytest.raises(ValueError, match="explicit adapter"):
        pipeline.step_simulation(run.id)
    with pytest.raises(ValueError, match="kind"):
        pipeline.step_simulation(run.id, FixtureModel())
    with pytest.raises(ValueError):
        pipeline.create_simulation(manifest, config())
    with pytest.raises(ValueError):
        pipeline.create_simulation(manifest, SimulationConfig(), kind=RunKind.CONTRACT_TEST)
    real = pipeline.create_simulation(manifest, config(), kind=RunKind.REAL)
    model = Adapter()
    model.kind = RunKind.REAL
    with pytest.raises(NotImplementedError):
        pipeline.step_simulation(real.id, model)
    assert not model.calls
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows if r["split"] != "validation"))
    with pytest.raises(ValueError, match="nonempty validation"):
        pipeline.create_simulation(manifest, config(), kind=RunKind.CONTRACT_TEST)


def test_precommit_source_and_identity_mutations_are_atomic(tmp_path, monkeypatch):
    pipeline, run, manifest = setup(tmp_path)
    original = manifest.read_bytes()

    class Mutating(Adapter):
        def predict(self, pages, **kwargs):
            results = super().predict(pages, **kwargs)
            manifest.write_bytes(original + b"\n")
            return results

    with pytest.raises(ValueError, match="changed"):
        pipeline.step_simulation(run.id, Mutating())
    assert pipeline.get_simulation(run.id) == run
    manifest.write_bytes(original)

    class Changed(Adapter):
        def predict(self, pages, **kwargs):
            results = super().predict(pages, **kwargs)
            monkeypatch.setattr(
                integration, "local_contract_identity", lambda: ("0" * 40, DEPENDENCIES)
            )
            return results

    with pytest.raises(ValueError, match="identity"):
        pipeline.step_simulation(run.id, Changed())
    assert pipeline.get_simulation(run.id) == run


def test_validation_and_hidden_truth_do_not_influence_training(tmp_path):
    pipeline, run, manifest = setup(tmp_path, page_budget=2)
    model = Adapter()
    original = pipeline.run_simulation(run.id, model)
    selected = set(original.rounds[0].revealed_ids)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    for row in rows:
        if row["id"] not in selected:
            row["regions"][0]["text"] = "x"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    repeated = pipeline.create_simulation(manifest, run.config, kind=RunKind.CONTRACT_TEST)
    other = Adapter()
    repeated = pipeline.run_simulation(repeated.id, other)
    assert original.rounds[0].revealed_ids == repeated.rounds[0].revealed_ids
    assert [c[4] for c in model.calls if c[0] == "fit"] == [
        c[4] for c in other.calls if c[0] == "fit"
    ]
    assert original.baseline.validation_metrics != repeated.baseline.validation_metrics
    assert original.rounds[0].validation_metrics != repeated.rounds[0].validation_metrics
    # Existing completed runs still reject changed source before adapter work.
    other.calls.clear()
    with pytest.raises(ValueError, match="changed"):
        pipeline.step_simulation(run.id, other)
    assert not other.calls


def test_literal_empty_illegible_ground_truth_is_not_substituted(tmp_path):
    pipeline, _, manifest = setup(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    for row in rows:
        if row["split"] == "validation":
            row["regions"][0].update(text="", illegible=True)
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    run = pipeline.create_simulation(manifest, config(page_budget=0), kind=RunKind.CONTRACT_TEST)
    done = pipeline.step_simulation(run.id, Adapter())
    metrics = done.baseline.validation_metrics
    assert metrics["reference_chars"] == metrics["reference_words"] == 0
    assert "cer" not in metrics and "wer" not in metrics
    region = SourceRegion(id="l", text="", illegible=True, box=Box(x=0, y=0, width=1, height=1))
    assert region.text == ""
    pipeline.export_simulation(run.id, tmp_path / "export")
    record = next(csv.DictReader((tmp_path / "export/rounds.csv").open()))
    assert record["cer"] == record["wer"] == ""
    assert record["cer_defined"] == record["wer_defined"] == "0"


def test_separate_process_resume_and_export(tmp_path):
    pipeline, run, _ = setup(tmp_path, page_budget=3)
    # The child imports this explicit synthetic adapter; no CLI or real backend is added.
    script = """
import sys
from pathlib import Path
import active_ocr.integrations.simulation as integration
from active_ocr.pipeline import Pipeline
from test_real_contract import Adapter, SOURCE, DEPENDENCIES
integration.local_contract_identity = lambda: (SOURCE, DEPENDENCIES)
pipeline = Pipeline.for_simulation(Path(sys.argv[1]))
run = pipeline.step_simulation(sys.argv[2], Adapter())
print(run.model_dump_json())
assert not any(name in sys.modules for name in ("torch", "transformers", "modal"))
"""
    environment = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(Path("src").resolve()), str(Path("tests").resolve())]),
    )
    results = []
    for _ in range(3):
        child = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / "runs"), run.id],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        results.append(json.loads(child.stdout))
    assert [len(r["rounds"]) for r in results] == [0, 1, 2]
    assert [r["complete"] for r in results] == [False, False, True]
    pipeline.export_simulation(run.id, tmp_path / "export")
    exported = json.loads((tmp_path / "export/results.json").read_text())
    assert exported == results[-1]
    with (tmp_path / "export/rounds.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert [r["record_type"] for r in rows] == ["baseline", "round", "round"]
    assert [r["round"] for r in rows] == ["0", "1", "2"]
    assert [r["labelled_pages"] for r in rows] == ["0", "2", "3"]
    assert rows[0]["selected_ids"] == rows[0]["revealed_ids"] == "[]"
    assert all(r["kind"] == RunKind.CONTRACT_TEST and r["annotation_seconds"] == "" for r in rows)
    assert rows[0]["purpose"] == "baseline_validation" and rows[1]["purpose"] == "validation"
    assert all(r["cer_defined"] == "1" and r["cer"] == "1.0" for r in rows)
    assert all(json.loads(r["validation_predictions"])[0]["status"] == "ok" for r in rows)
    with pytest.raises(FileExistsError):
        pipeline.export_simulation(run.id, tmp_path / "export")


def test_old_fixture_json_remains_readable_and_resumable(tmp_path):
    manifest = create_fixture(tmp_path / "data", train_pages=5)
    pipeline = Pipeline.for_simulation(tmp_path / "runs")
    run = pipeline.create_simulation(
        manifest, SimulationConfig(report_predictions=True, evaluator_id="page-text-nfc-v1")
    )
    run = pipeline.step_simulation(run.id, evaluator=integration.PageTextEvaluatorV1())
    # Reconstruct the previous on-disk schema, including legacy validation purpose omission.
    raw = run.model_dump(mode="json")
    raw.pop("baseline")
    raw["config"].pop("real")
    for record in raw["rounds"]:
        record.pop("telemetry")
        for prediction in record["predictions"] + record["validation_predictions"]:
            for key in ("purpose", "status", "raw_output_artifact", "finish_reason"):
                prediction.pop(key)
    with sqlite3.connect(pipeline.settings.database) as connection:
        connection.execute(
            "UPDATE records SET payload=? WHERE key=?",
            (json.dumps(raw, separators=(",", ":")), run.id),
        )
    legacy = pipeline.get_simulation(run.id)
    assert legacy.rounds[0].validation_predictions[0].purpose is PredictionPurpose.POOL
    done = pipeline.run_simulation(run.id, evaluator=integration.PageTextEvaluatorV1())
    assert done.complete and done.baseline is None
    assert done.rounds[0].validation_predictions[0].purpose is PredictionPurpose.POOL
    assert done.rounds[1].validation_predictions[0].purpose is PredictionPurpose.VALIDATION
    pipeline.export_simulation(run.id, tmp_path / "export")
    with (tmp_path / "export/rounds.csv").open() as file:
        assert all(r["record_type"] == "round" for r in csv.DictReader(file))


def test_failures_are_persisted_and_exported_as_empty_hypothesis_counts(tmp_path):
    pipeline, run, _ = setup(tmp_path, page_budget=0)

    class Failed(Adapter):
        def predict(self, pages, **kwargs):
            return tuple(
                p.model_copy(
                    update={
                        "status": PredictionStatus.TRUNCATED,
                        "finish_reason": "length",
                        "raw_output_artifact": "sha256:synthetic",
                    }
                )
                for p in super().predict(pages, **kwargs)
            )

    done = pipeline.step_simulation(run.id, Failed())
    metrics = done.baseline.validation_metrics
    assert metrics["pages"] == metrics["truncated_pages"] == metrics["failed_pages"] == 1
    assert metrics["cer"] == metrics["wer"] == 1
    pipeline.export_simulation(run.id, tmp_path / "export")
    with (tmp_path / "export/rounds.csv").open() as file:
        row = next(csv.DictReader(file))
    assert row["truncated_pages"] == row["failed_pages"] == "1"
    assert json.loads(row["validation_predictions"])[0]["status"] == "truncated"


@pytest.mark.parametrize("baseline", [True, False])
def test_identity_rechecked_after_load_or_fit_before_prediction(tmp_path, baseline):
    pipeline, run, _ = setup(tmp_path)
    if not baseline:
        run = pipeline.step_simulation(run.id, Adapter())

    class Changed(Adapter):
        def load_base(self, **kwargs):
            result = super().load_base(**kwargs)
            self.identity = self.identity.model_copy(update={"model_manifest_sha256": "0" * 64})
            return result

        def fit(self, examples, **kwargs):
            result = super().fit(examples, **kwargs)
            self.identity = self.identity.model_copy(update={"processor_manifest_sha256": "0" * 64})
            return result

    model = Changed()
    with pytest.raises(ValueError, match="identity mismatch"):
        pipeline.step_simulation(run.id, model)
    assert not any(call[0] == "predict" for call in model.calls)
    assert pipeline.get_simulation(run.id) == run


def test_telemetry_varies_without_changing_identity(tmp_path):
    pipeline, run, _ = setup(tmp_path, page_budget=2)
    model = Adapter()
    model.telemetry = ExecutionTelemetry(
        device="synthetic", elapsed_seconds=1, peak_memory_bytes=10
    )
    baseline = pipeline.step_simulation(run.id, model)
    model.telemetry = ExecutionTelemetry(
        device="synthetic-2", elapsed_seconds=3, peak_memory_bytes=20
    )
    done = pipeline.step_simulation(run.id, model)
    assert done.complete and done.baseline.telemetry == baseline.baseline.telemetry
    assert done.rounds[0].telemetry == model.telemetry


@pytest.mark.parametrize("changed", ["backend", "fit_policy", "kind"])
def test_adapter_contract_mutation_during_prediction_is_atomic(tmp_path, changed):
    pipeline, run, _ = setup(tmp_path)

    class Changed(Adapter):
        def predict(self, pages, **kwargs):
            results = super().predict(pages, **kwargs)
            setattr(self, changed, "changed")
            return results

    with pytest.raises(ValueError, match="identity mismatch"):
        pipeline.step_simulation(run.id, Changed())
    assert pipeline.get_simulation(run.id) == run


# Retain the actual implementation before the autouse synthetic seam replaces the module attribute.
actual_local_identity = integration.local_contract_identity


def test_local_fingerprint_canonical_order_names_versions_and_dirty_code(monkeypatch):
    class Distribution:
        def __init__(self, name, version):
            self.metadata = {"Name": name}
            self.version = version

    packages = [Distribution("Example_Pkg", "1"), Distribution("Other.Name", "2")]
    monkeypatch.setattr(integration, "distributions", lambda: packages)
    monkeypatch.setattr(integration, "runtime_identity", lambda: (SOURCE, ()))
    monkeypatch.setattr(
        integration.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(args=[], returncode=0, stdout=""),
    )
    original = actual_local_identity()
    packages[:] = [Distribution("other-name", "2"), Distribution("example.pkg", "1")]
    assert actual_local_identity() == original
    packages[0].version = "3"
    assert actual_local_identity()[1] != original[1]
    monkeypatch.setattr(
        integration.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=" M dependency.lock\n"
        ),
    )
    assert actual_local_identity()[0] == SOURCE + "-dirty"


def test_no_local_contract_can_start_with_unknown_or_dirty_source(tmp_path, monkeypatch):
    # The production helper is exercised against the current checkout, with no model loading.
    monkeypatch.setattr(integration, "local_contract_identity", actual_local_identity)
    pipeline = Pipeline.for_simulation(tmp_path / "runs")
    manifest = create_fixture(tmp_path / "data")
    with pytest.raises(ValueError, match="identity"):
        pipeline.create_simulation(manifest, config(), kind=RunKind.CONTRACT_TEST)
    assert pipeline.store.list("simulation", SimulationRun) == ()


def test_legacy_gpu_job_rejects_baseline_result_without_publishing(tmp_path, monkeypatch):
    from test_pipeline import make_pipeline

    from active_ocr.models import Stage, Strategy

    pipeline, _, gpu = make_pipeline(tmp_path)
    experiment = pipeline.create_experiment("legacy", Strategy.RANDOM, batch_size=1, rounds=1)
    experiment = experiment.model_copy(
        update={"stage": Stage.SCORING, "job_id": "job", "model_id": CHECKPOINT}
    )
    pipeline.store.save("experiment", experiment.id, experiment)
    predictions = [
        Prediction(
            page_id=f"page-{i}",
            experiment_id=experiment.id,
            model_id=CHECKPOINT,
            round_number=0,
            purpose=PredictionPurpose.BASELINE_VALIDATION,
        ).model_dump()
        for i in range(1, 4)
    ]
    monkeypatch.setattr(
        gpu, "job", lambda _: {"status": "succeeded", "result": {"predictions": predictions}}
    )
    with pytest.raises(ValueError, match="ownership"):
        pipeline.poll_job(experiment.id)
    assert pipeline.get_experiment(experiment.id) == experiment
    assert pipeline.store.list("prediction", Prediction) == ()
