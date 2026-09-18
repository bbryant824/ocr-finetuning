"""Independent local-contract review. Synthetic only; no model or provider execution.

Run against a clean source checkout (see docs/verification/local-contract-review.md).
Identity checks are real: this module never patches local_contract_identity.
"""

import csv
import hashlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

import active_ocr.integrations.simulation as integration
from active_ocr.evaluation import page_text_metrics
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
    Split,
)
from active_ocr.pipeline import Pipeline


def contract_config(**overrides):
    source, dependencies = integration.local_contract_identity()
    identity = ExpectedIdentity(
        source_sha=source,
        dependency_sha256=dependencies,
        model_revision="1" * 40,
        model_manifest_sha256="2" * 64,
        processor_revision="3" * 40,
        processor_manifest_sha256="4" * 64,
        recipe_version="independent-synthetic-v1",
        evaluator_id="page-text-nfc-v1",
    )
    recipe = RealOCRConfig(
        backend="independent-synthetic-v1",
        recipe_version=identity.recipe_version,
        model_repository="synthetic/model",
        model_revision=identity.model_revision,
        processor_repository="synthetic/processor",
        processor_revision=identity.processor_revision,
        training_policy_id="synthetic-reset-v1",
        decode_policy_id="synthetic-output-v1",
        expected_identity=identity,
    )
    return SimulationConfig(
        real=recipe, backend=recipe.backend, evaluator_id=recipe.evaluator_id, **overrides
    )


class Witness:
    kind = RunKind.CONTRACT_TEST
    fit_policy = "reset-fit-cumulative-v1"

    def __init__(self, recipe):
        self.real_config = recipe
        self.identity = recipe.expected_identity
        self.backend = recipe.backend
        self.events = []
        self.telemetry = None

    def load_base(self, *, experiment_id):
        self.events.append(("base", experiment_id))
        return "checkpoint:sha256:" + "0" * 64

    def fit(self, examples, *, seed, experiment_id, round_number):
        assert all(e.page.split is Split.TRAIN for e in examples)
        targets = [(e.page.id, [r.model_dump() for r in e.regions]) for e in examples]
        self.events.append(("fit", seed, targets))
        digest = hashlib.sha256(json.dumps([seed, targets], sort_keys=True).encode()).hexdigest()
        return "checkpoint:sha256:" + digest

    def predict(self, pages, *, experiment_id, round_number, model_id, purpose):
        assert all(p.split is Split.VALIDATION and not hasattr(p, "regions") for p in pages)
        self.events.append(("predict", [p.id for p in pages], purpose))
        return tuple(
            Prediction(
                page_id=p.id,
                experiment_id=experiment_id,
                round_number=round_number,
                model_id=model_id,
                purpose=purpose,
                regions=(Region(id="line", text="é", box=Box(x=0, y=0, width=1, height=1)),),
            )
            for p in reversed(pages)
        )


def setup_case(root, *, train=5, **overrides):
    manifest = integration.create_fixture(root / "data", train_pages=train)
    pipeline = Pipeline.for_simulation(root / "state")
    config = contract_config(**overrides)
    run = pipeline.create_simulation(manifest, config, kind=RunKind.CONTRACT_TEST)
    return pipeline, run, manifest, Witness(config.real)


@pytest.mark.parametrize("acquired", [False, True])
@pytest.mark.parametrize(
    "status",
    [PredictionStatus.INVALID_OUTPUT, PredictionStatus.TRUNCATED, PredictionStatus.REFUSAL],
)
def test_failed_nonfinite_geometry_cannot_publish_unreadable_state(tmp_path, acquired, status):
    pipeline, run, _, witness = setup_case(tmp_path, page_budget=2 if acquired else 0)
    if acquired:
        run = pipeline.step_simulation(run.id, witness)

    class Failed(Witness):
        def predict(self, pages, **kwargs):
            return tuple(
                p.model_copy(
                    update={
                        "status": status,
                        "regions": (
                            Region(id="partial", box=Box(x=float("inf"), y=0, width=1, height=1)),
                        ),
                    }
                )
                for p in super().predict(pages, **kwargs)
            )

    # Either refuse the malformed structured result atomically or sanitize it to a
    # readable failed record. Raw malformed text may live in a separate artifact.
    try:
        committed = pipeline.step_simulation(run.id, Failed(run.config.real))
    except ValueError:
        assert pipeline.get_simulation(run.id) == run
    else:
        assert pipeline.get_simulation(run.id) == committed
        pipeline.export_simulation(run.id, tmp_path / "export")


@pytest.mark.parametrize("acquired", [False, True])
def test_actual_sqlite_abort_rolls_back_and_retry_roundtrips(tmp_path, acquired):
    pipeline, run, _, witness = setup_case(tmp_path, page_budget=2)
    if acquired:
        run = pipeline.step_simulation(run.id, witness)
    with sqlite3.connect(pipeline.store.database_path) as db:
        db.execute(
            "CREATE TRIGGER abort_update AFTER UPDATE ON records "
            "BEGIN SELECT RAISE(ABORT, 'independent storage abort'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="independent storage abort"):
        pipeline.step_simulation(run.id, witness)
    reopened = Pipeline.for_simulation(tmp_path / "state")
    assert reopened.get_simulation(run.id) == run
    with sqlite3.connect(pipeline.store.database_path) as db:
        db.execute("DROP TRIGGER abort_update")
    retried = reopened.step_simulation(run.id, Witness(run.config.real))
    assert reopened.get_simulation(run.id) == retried
    assert len(retried.rounds) == int(acquired)


@pytest.mark.parametrize("acquired", [False, True])
def test_separate_connections_compete_for_one_commit(tmp_path, acquired):
    pipeline, run, _, witness = setup_case(tmp_path, page_budget=2)
    if acquired:
        run = pipeline.step_simulation(run.id, witness)
    barrier = Barrier(2)

    class Concurrent(Witness):
        def predict(self, pages, **kwargs):
            result = super().predict(pages, **kwargs)
            barrier.wait(timeout=10)
            return result

    def attempt():
        connection = Pipeline.for_simulation(tmp_path / "state")
        try:
            return connection.step_simulation(run.id, Concurrent(run.config.real))
        except RuntimeError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: attempt(), range(2)))
    failures = [r for r in results if isinstance(r, RuntimeError)]
    assert len(failures) == 1 and "concurrently" in str(failures[0])
    saved = pipeline.get_simulation(run.id)
    assert saved.baseline is not None and len(saved.rounds) == int(acquired)


@pytest.mark.parametrize("metric", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_evaluator_is_atomic(tmp_path, metric):
    pipeline, run, _, witness = setup_case(tmp_path, page_budget=0)

    class Evaluator:
        identifier = "page-text-nfc-v1"

        def __call__(self, *args):
            return {"cer": metric}

    evaluator = Evaluator()
    with pytest.raises(ValueError, match="finite"):
        pipeline.step_simulation(run.id, witness, evaluator=evaluator)
    assert pipeline.get_simulation(run.id) == run


def test_actual_previous_writer_unicode_json_cas_resume(tmp_path):
    """Use previous application code to write bytes, not a reconstructed new-model dump."""
    root = Path(integration.__file__).resolve().parents[3]
    parent = tmp_path / "previous"
    parent.mkdir()
    archive = subprocess.check_output(
        ["git", "-C", str(root), "archive", "cce1f14c9bb107b2a1b9dc4c0f8c94c942f37122"]
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as files:
        files.extractall(parent, filter="data")
    program = """
import json, sys
from pathlib import Path
from active_ocr.integrations.simulation import create_fixture
from active_ocr.models import SimulationConfig
from active_ocr.pipeline import Pipeline
root = Path(sys.argv[1])
manifest = create_fixture(root / 'data')
rows = [json.loads(line) for line in manifest.read_text().splitlines()]
for row in rows:
    row['regions'][0]['text'] = 'é 漢字'
manifest.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\\n' for row in rows))
class Legacy:
    identifier = 'legacy-float'
    def __call__(self, *args): return {'méan': 0.25, 'whole': 2.0}
pipeline = Pipeline.for_simulation(root / 'state')
run = pipeline.create_simulation(manifest, SimulationConfig(page_budget=3, batch_size=2,
    report_predictions=True, evaluator_id='legacy-float'))
run = pipeline.step_simulation(run.id, evaluator=Legacy())
print(run.id)
"""
    child = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)],
        cwd=parent,
        env=dict(os.environ, PYTHONPATH=str(parent / "src")),
        check=True,
        capture_output=True,
        text=True,
    )
    pipeline = Pipeline.for_simulation(tmp_path / "state")
    run_id = child.stdout.strip()
    before = pipeline.get_simulation(run_id)
    assert before.rounds[0].validation_predictions[0].purpose is PredictionPurpose.POOL

    class Legacy:
        identifier = "legacy-float"

        def __call__(self, *args):
            return {"méan": 0.25, "whole": 2.0}

    completed = pipeline.run_simulation(run_id, evaluator=Legacy())
    assert completed.complete and [r.labelled_count for r in completed.rounds] == [2, 3]
    assert completed.rounds[0] == before.rounds[0]
    assert completed.rounds[1].validation_predictions[0].purpose is PredictionPurpose.VALIDATION
    assert pipeline.get_simulation(run_id) == completed


def test_unpatched_identity_detects_real_dirty_checkout_and_added_distribution(tmp_path):
    pipeline, run, _, witness = setup_case(tmp_path, page_budget=0)
    done = pipeline.step_simulation(run.id, witness)
    witness.events.clear()
    root = Path(integration.__file__).resolve().parents[3]
    marker = root / "independent-identity-probe.txt"
    assert not marker.exists()
    try:
        marker.write_text("temporary untracked identity test")
        with pytest.raises(ValueError, match="identity"):
            pipeline.step_simulation(run.id, witness)
    finally:
        marker.unlink()
    dist = tmp_path / "packages" / "independent_probe-1.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Name: independent-probe\nVersion: 1.0\n")
    sys.path.insert(0, str(dist.parent))
    try:
        with pytest.raises(ValueError, match="identity"):
            pipeline.step_simulation(run.id, witness)
    finally:
        sys.path.remove(str(dist.parent))
    assert not witness.events and pipeline.get_simulation(run.id) == done
    assert pipeline.step_simulation(run.id, witness) == done


@pytest.mark.parametrize("train,budget,rounds", [(5, 0, 3), (5, 2, 0), (0, 2, 3)])
def test_zero_paths_use_only_base_and_validation(tmp_path, train, budget, rounds):
    pipeline, run, _, witness = setup_case(
        tmp_path, train=train, page_budget=budget, max_rounds=rounds
    )
    assert run.baseline is None and not run.complete and not witness.events
    completed = pipeline.step_simulation(run.id, witness)
    assert completed.baseline and completed.complete and not completed.rounds
    assert [e[0] for e in witness.events] == ["base", "predict"]
    witness.events.clear()
    assert pipeline.step_simulation(run.id, witness) == completed and not witness.events


def test_hidden_truth_seed_order_and_cumulative_budget(tmp_path):
    pipeline, run, manifest, witness = setup_case(tmp_path, page_budget=3, batch_size=2, seed=41)
    completed = pipeline.run_simulation(run.id, witness)
    fixture = pipeline.create_simulation(
        manifest, SimulationConfig(page_budget=3, batch_size=2, seed=41)
    )
    fixture = pipeline.run_simulation(fixture.id)
    assert [r.selected_ids for r in fixture.rounds] == [r.selected_ids for r in completed.rounds]
    assert [r.labelled_count for r in completed.rounds] == [2, 3]
    fits = [e for e in witness.events if e[0] == "fit"]
    assert [[t[0] for t in e[2]] for e in fits] == [list(r.revealed_ids) for r in completed.rounds]
    selected = set(completed.rounds[-1].revealed_ids)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    for row in rows:
        if row["id"] not in selected:
            row["regions"][0]["text"] = "changed hidden truth"
            row["regions"][0]["box"]["width"] = 12
    manifest.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)))
    repeated = pipeline.create_simulation(manifest, run.config, kind=RunKind.CONTRACT_TEST)
    other = Witness(run.config.real)
    repeated = pipeline.run_simulation(repeated.id, other)
    assert [r.selected_ids for r in completed.rounds] == [r.selected_ids for r in repeated.rounds]
    assert [r.model_id for r in completed.rounds] == [r.model_id for r in repeated.rounds]
    assert completed.baseline.validation_metrics != repeated.baseline.validation_metrics
    with pytest.raises(ValueError, match="changed"):
        pipeline.step_simulation(run.id, witness)


def test_hand_counted_unicode_failures_and_independent_denominators():
    result = page_text_metrics(
        [["e\u0301"], [""], ["a b"]],
        [["é"], ["xx"], ["ignored"]],
        [PredictionStatus.OK, PredictionStatus.OK, PredictionStatus.REFUSAL],
    )
    assert result["reference_chars"] == 4 and result["char_edits"] == 5
    assert result["reference_words"] == 3 and result["word_edits"] == 3
    assert result["cer"] == 1.25 and result["wer"] == 1
    assert result["failed_pages"] == result["refusal_pages"] == 1
    spaces = page_text_metrics([["\u2003", ""]], [[]], [PredictionStatus.OK])
    assert spaces["reference_chars"] == 2 and spaces["cer"] == 1
    assert spaces["wer_defined"] == 0 and "wer" not in spaces


def child_step(state, run_id, export=None):
    pipeline = Pipeline.for_simulation(Path(state))
    current = pipeline.get_simulation(run_id)
    witness = Witness(current.config.real)
    witness.telemetry = ExecutionTelemetry(elapsed_seconds=float(len(current.rounds) + 1))
    result = pipeline.step_simulation(run_id, witness)
    if export:
        pipeline.export_simulation(run_id, Path(export))
    assert not any(name in sys.modules for name in ("torch", "transformers", "modal"))
    print(result.model_dump_json())


def test_fresh_process_resume_with_real_identity_and_exports(tmp_path):
    pipeline, run, _, witness = setup_case(tmp_path, page_budget=3, batch_size=2)
    root = Path(integration.__file__).resolve().parents[3]
    environment = dict(
        os.environ, PYTHONPATH=os.pathsep.join([str(root / "src"), str(Path(__file__).parent)])
    )
    program = (
        "from test_contract_independent import child_step; import sys; child_step(*sys.argv[1:])"
    )
    results = []
    for step in range(4):
        args = [sys.executable, "-c", program, str(tmp_path / "state"), run.id]
        if step == 3:
            args.append(str(tmp_path / "export"))
        child = subprocess.run(args, env=environment, check=True, capture_output=True, text=True)
        results.append(json.loads(child.stdout))
    assert [len(r["rounds"]) for r in results] == [0, 1, 2, 2]
    assert results[2] == results[3]
    assert json.loads((tmp_path / "export/results.json").read_text()) == results[-1]
    with (tmp_path / "export/rounds.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert [r["labelled_pages"] for r in rows] == ["0", "2", "3"]
    assert [r["record_type"] for r in rows] == ["baseline", "round", "round"]
    assert all(r["annotation_seconds"] == "" and r["kind"] == RunKind.CONTRACT_TEST for r in rows)
    with pytest.raises(FileExistsError):
        pipeline.export_simulation(run.id, tmp_path / "export")


@pytest.mark.parametrize("acquired", [False, True])
@pytest.mark.parametrize("status", list(PredictionStatus))
def test_correction_raw_nonfinite_payload_keeps_prior_export(tmp_path, acquired, status):
    pipeline, prior, _, witness = setup_case(tmp_path, page_budget=2)
    if acquired:
        prior = pipeline.step_simulation(prior.id, witness)

    class RawPayload(Witness):
        coordinate = "x"
        value = float("nan")

        def predict(self, pages, **kwargs):
            output = [p.model_dump() for p in super().predict(pages, **kwargs)]
            output[0]["status"] = status
            output[0]["regions"][0]["box"][self.coordinate] = self.value
            return output

    raw = RawPayload(prior.config.real)
    for coordinate in ("x", "y", "width", "height"):
        for value in (float("nan"), float("inf"), -float("inf")):
            raw.coordinate, raw.value = coordinate, value
            with pytest.raises(ValueError):
                pipeline.step_simulation(prior.id, raw)
            assert pipeline.get_simulation(prior.id) == prior
    pipeline.export_simulation(prior.id, tmp_path / "prior-export")
    assert json.loads((tmp_path / "prior-export/results.json").read_text()) == prior.model_dump(
        mode="json"
    )
    retried = pipeline.step_simulation(prior.id, witness)
    assert pipeline.get_simulation(prior.id) == retried
    assert len(retried.rounds) == int(acquired)


@pytest.mark.parametrize("acquired", [False, True])
@pytest.mark.parametrize(
    "status",
    [PredictionStatus.INVALID_OUTPUT, PredictionStatus.TRUNCATED, PredictionStatus.REFUSAL],
)
def test_correction_finite_failure_preserves_evidence(tmp_path, acquired, status):
    pipeline, prior, _, witness = setup_case(tmp_path, page_budget=2)
    if acquired:
        prior = pipeline.step_simulation(prior.id, witness)

    class Failed(Witness):
        def predict(self, pages, **kwargs):
            return tuple(
                p.model_copy(
                    update={
                        "status": status,
                        "raw_output_artifact": "synthetic:raw-failure",
                        "finish_reason": "synthetic-stop",
                    }
                )
                for p in super().predict(pages, **kwargs)
            )

    committed = pipeline.step_simulation(prior.id, Failed(prior.config.real))
    assert pipeline.get_simulation(prior.id) == committed
    record = committed.rounds[-1] if acquired else committed.baseline
    prediction = record.validation_predictions[0]
    assert prediction.status is status and prediction.regions[0].text == "é"
    assert prediction.raw_output_artifact == "synthetic:raw-failure"
    assert prediction.finish_reason == "synthetic-stop"
    assert record.validation_metrics["failed_pages"] == 1
    assert record.validation_metrics["char_edits"] == record.validation_metrics["reference_chars"]
    pipeline.export_simulation(prior.id, tmp_path / "export")
    assert json.loads((tmp_path / "export/results.json").read_text()) == committed.model_dump(
        mode="json"
    )
