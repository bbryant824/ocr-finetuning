"""Independent synthetic converter checks. No full actual READ conversion or model work."""

import csv
import hashlib
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from active_ocr.integrations import public_dataset as converter
from active_ocr.integrations.local_data import load_image_pages
from active_ocr.integrations.simulation import (
    FixtureModel,
    LocalOracle,
    SourcePage,
    _read_source,
    local_contract_identity,
)
from active_ocr.models import Page, SimulationConfig, SourcePolicy, Split
from active_ocr.pipeline import Pipeline

POLICY = SourcePolicy.READ2016
NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"


def xml(name):
    # Deliberately different node, geometric and semantic reading orders.
    return f'''<PcGts xmlns="{NS}"><Page imageFilename="{name}" imageWidth="32" imageHeight="32">
<ReadingOrder><OrderedGroup id="order"><RegionRefIndexed index="7" regionRef="late"/>
<RegionRefIndexed index="2" regionRef="early"/></OrderedGroup></ReadingOrder>
<TextRegion id="late" custom="readingOrder {{index:7;}}">
<TextLine id="blank" custom="readingOrder {{index:0;}}"><Coords points="0,0 3,1 1,4"/>
<TextEquiv><Unicode/></TextEquiv></TextLine></TextRegion>
<TextRegion id="early" custom="readingOrder {{index:2;}}">
<TextLine id="two" custom="readingOrder {{index:1;}}"><Coords points="3,5 9,6 4,12"/>
<TextEquiv><Unicode> e&#769; &#13;&#10;漢 &amp; </Unicode></TextEquiv></TextLine>
<TextLine id="one" custom="readingOrder {{index:0;}} unclear {{offset:1; length:1;}}
abbrev {{offset:0; length:1; expansion:expanded;}}"><Coords points="20,21 32,22 22,32"/>
<Baseline points="20,24 30,24"/><TextStyle strikethrough="true"/>
<TextEquiv><Unicode> q. </Unicode></TextEquiv></TextLine>
<TextEquiv><Unicode>region summary must not become a target</Unicode></TextEquiv>
</TextRegion></Page></PcGts>'''.encode()


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(scope="module")
def synthetic_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("independent-synthetic-read")
    serial = 0
    for folder, count in (("Training", 350), ("Validation", 50)):
        base = root / "PublicData" / folder
        (base / "Images").mkdir(parents=True)
        (base / "page/page").mkdir(parents=True)
        for index in range(count):
            name = f"synthetic{index:04}"
            image = Image.new("RGB", (32, 32))
            # Encode a distinct visible binary pattern, not source-derived labels.
            for bit in range(9):
                if serial & (1 << bit):
                    for x in range(bit * 3, bit * 3 + 2):
                        for y in range(32):
                            image.putpixel((x, y), (255, 255, 255))
            image.save(base / "Images" / f"{name}.JPG", quality=100, subsampling=0)
            (base / "page/page" / f"{name}.xml").write_bytes(xml(f"{name}.JPG"))
            serial += 1
        (base / "page/doc.xml").write_text('<document docId="-1"/>')
        (base / "page/list").write_text("synthetic source list\n")
    inventory = tuple(
        converter.SourceFile(
            uri=p.relative_to(root).as_posix(),
            bytes=p.stat().st_size,
            sha256=hash_bytes(p.read_bytes()),
        )
        for p in sorted(root.rglob("*"))
        if p.is_file()
    )
    assert len(inventory) == 804
    assert len({f.sha256 for f in inventory if f.uri.endswith(".JPG")}) == 400
    return root, inventory


@pytest.fixture(scope="module")
def prepared(synthetic_source, tmp_path_factory):
    root, files = synthetic_source
    destination = tmp_path_factory.mktemp("independent-prepared") / "data"
    # Lower-level synthetic publication only. Public archive pin is never patched.
    revision, _ = local_contract_identity()
    assert len(revision) == 40
    return converter._publish(root, destination, files, revision)


@pytest.fixture
def dataset(prepared, tmp_path):
    root = tmp_path / "dataset"
    shutil.copytree(prepared.parent, root)
    return root / "pages.jsonl"


def test_literal_targets_and_lossy_envelopes():
    metadata, regions = converter.parse_page(xml("synthetic.JPG"), "synthetic.JPG")
    assert [(r.id, r.text) for r in regions] == [
        ("one", " q. "),
        ("two", " e\u0301 \r\n漢 & "),
        ("blank", ""),
    ]
    assert regions[0].box.model_dump() == dict(x=20, y=21, width=12, height=11)
    assert regions[1].box.model_dump() == dict(x=3, y=5, width=6, height=7)
    assert metadata["missing_baselines"] == 2 and metadata["empty_lines"] == 1
    assert all(not r.illegible for r in regions)


@pytest.mark.parametrize(
    "old,new",
    [
        (b"index:1;", b"index:4;"),
        (b'regionRef="early"', b'regionRef="absent"'),
        (b"unclear {", b"illegible {"),
        (b"3,5 9,6 4,12", b"3,5 9,6 4,33"),
        (b"3,5 9,6 4,12", b"3,5 3,6 3,12"),
        (b"3,5 9,6 4,12", b"3,5 NaN,6 4,12"),
        (b"<Unicode/>", b"<Unicode/><Unicode/>"),
    ],
)
def test_parser_refuses_ambiguous_input(old, new):
    with pytest.raises(ValueError):
        converter.parse_page(xml("synthetic.JPG").replace(old, new), "synthetic.JPG")


def test_dtd_refusal_does_not_expand_local_entity(tmp_path):
    sentinel = tmp_path / "external.txt"
    sentinel.write_text("never a transcript")
    declaration = f'<!DOCTYPE PcGts [<!ENTITY x SYSTEM "{sentinel.as_uri()}">]>'.encode()
    with pytest.raises(ValueError, match="DTD/entity"):
        converter.parse_page(declaration + xml("synthetic.JPG"), "synthetic.JPG")


@pytest.mark.parametrize("fault", ["orientation", "mode", "dimensions", "format", "truncated"])
def test_image_validation_does_not_silently_transform(fault):
    buffer = io.BytesIO()
    image = Image.new("L" if fault == "mode" else "RGB", (32, 32))
    exif = Image.Exif()
    if fault == "orientation":
        exif[274] = 8
    image.save(buffer, format="PNG" if fault == "format" else "JPEG", exif=exif)
    data = buffer.getvalue()
    if fault == "truncated":
        data = data[: len(data) // 2]
    with pytest.raises((OSError, ValueError)):
        converter.validate_image(data, 31 if fault == "dimensions" else 32, 32)


@pytest.mark.parametrize("document", [None, "missing"])
def test_legacy_importer_does_not_admit_unknown_documents(tmp_path, document):
    manifest = tmp_path / "legacy.jsonl"
    row = {"path": "image.JPG"}
    if document is None:
        row["document_id"] = None
    manifest.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="document_id"):
        load_image_pages(manifest, document_root=tmp_path / "documents")
    assert not (tmp_path / "documents").exists()


def test_original_bytes_determinism_and_explicit_admission(synthetic_source, prepared, tmp_path):
    root, files = synthetic_source
    revision, _ = local_contract_identity()
    second = converter._publish(root, tmp_path / "second", files, revision)
    for name in ("pages.jsonl", "provenance.json"):
        assert (prepared.parent / name).read_bytes() == (second.parent / name).read_bytes()
    for item in files:
        assert (second.parent / "source" / item.uri).read_bytes() == (root / item.uri).read_bytes()
    with pytest.raises(ValueError):
        _read_source(prepared.read_bytes(), prepared.parent)
    rows = _read_source(prepared.read_bytes(), prepared.parent, source_policy=POLICY)
    assert len(rows) == 400 and all(p.document_id is None for p in rows)
    assert sum(p.split is Split.TRAIN for p in rows) == 350
    base = dict(id="p", image_uri="p.JPG", width=1, height=1, document_id=None)
    with pytest.raises(ValueError):
        Page(**base)
    with pytest.raises(ValueError):
        SourcePage(**base, split=Split.TRAIN, image_sha256="a" * 64)
    strict = dict(base, document_id="known")
    assert Page(**strict).document_id == "known"


@pytest.mark.parametrize(
    "fault",
    [
        "truth",
        "document",
        "policy",
        "test",
        "missing_document",
        "provenance",
        "xml",
        "image",
        "doc",
        "snapshot",
    ],
)
def test_drift_never_becomes_a_valid_snapshot(dataset, tmp_path, fault):
    pipeline = Pipeline.for_simulation(tmp_path / "state")
    if fault == "snapshot":
        frozen = LocalOracle.freeze(dataset, pipeline.store, source_policy=POLICY)
        page = frozen.pages[0].model_copy(update={"width": 31})
        with pytest.raises(ValueError):
            LocalOracle(frozen.model_copy(update={"pages": (page, *frozen.pages[1:])}))
        return
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    if fault == "truth":
        rows[0]["regions"][0]["text"] = "edited target"
    elif fault == "document":
        rows[0]["document_id"] = "-1"
    elif fault == "policy":
        rows[0]["source_policy"] = "known-document-v1"
    elif fault == "test":
        rows[0]["split"] = "test"
    elif fault == "missing_document":
        rows[0].pop("document_id")
    else:
        locations = {
            "provenance": "provenance.json",
            "xml": "source/PublicData/Training/page/page/synthetic0000.xml",
            "doc": "source/PublicData/Validation/page/doc.xml",
            "image": rows[0]["image_uri"],
        }
        target = dataset.parent / locations[fault]
        target.write_bytes(target.read_bytes() + b"changed")
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        LocalOracle.freeze(dataset, pipeline.store, source_policy=POLICY)


@pytest.mark.parametrize("namespace", ["simulation-images", "simulation-source"])
def test_freeze_checks_persisted_bytes(dataset, tmp_path, monkeypatch, namespace):
    pipeline = Pipeline.for_simulation(tmp_path / "state")
    save = pipeline.store.save_artifact

    def damaged(kind, data, suffix=""):
        path = save(kind, data, suffix)
        if kind == namespace:
            path.write_bytes(path.read_bytes() + b"changed")
        return path

    monkeypatch.setattr(pipeline.store, "save_artifact", damaged)
    with pytest.raises(ValueError):
        LocalOracle.freeze(dataset, pipeline.store, source_policy=POLICY)


def test_public_api_has_no_synthetic_archive_bypass(synthetic_source, tmp_path):
    root, _ = synthetic_source
    archive = tmp_path / "wrong.tgz"
    archive.write_bytes(b"not the pinned archive")
    for policy in (SourcePolicy.KNOWN_DOCUMENT, POLICY):
        with pytest.raises(ValueError):
            converter.convert_read2016(archive, root, tmp_path / "out", source_policy=policy)
        assert not (tmp_path / "out").exists()
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        converter.convert_read2016(archive, root, tmp_path / "out", source_policy=POLICY)


@pytest.mark.parametrize("kind", ["traversal", "hardlink", "fifo", "absolute"])
def test_archive_refuses_unsafe_members_without_extraction(tmp_path, kind):
    path = tmp_path / "bad.tgz"
    with tarfile.open(path, "w:gz") as tar:
        entry = tarfile.TarInfo("/outside" if kind == "absolute" else "../outside")
        if kind == "hardlink":
            entry.name, entry.type, entry.linkname = "PublicData/file", tarfile.LNKTYPE, "elsewhere"
        elif kind == "fifo":
            entry.name, entry.type = "PublicData/file", tarfile.FIFOTYPE
        tar.addfile(entry)
    with pytest.raises(ValueError):
        converter._tar_inventory(path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bad.tgz"]


def test_concurrent_publication_and_lost_response(synthetic_source, tmp_path, monkeypatch):
    source, files = synthetic_source
    revision, _ = local_contract_identity()
    output = tmp_path / "race"

    def publish():
        try:
            return converter._publish(source, output, files, revision)
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: publish(), range(2)))
    assert outcomes.count(None) == 1
    before = (output / "pages.jsonl").read_bytes()
    assert len(_read_source(before, output, source_policy=POLICY)) == 400
    replace = converter.os.replace
    second = tmp_path / "response-lost"

    def lost(source, target):
        replace(source, target)
        if Path(target).name == "pages.jsonl":
            raise OSError("independent lost response")

    monkeypatch.setattr(converter.os, "replace", lost)
    with pytest.raises(OSError, match="lost response"):
        converter._publish(source, second, files, revision)
    assert (second / "pages.jsonl").read_bytes() == before
    assert len(_read_source(before, second, source_policy=POLICY)) == 400


def test_failed_publication_preserves_unrelated_file(synthetic_source, tmp_path, monkeypatch):
    source, files = synthetic_source
    revision, _ = local_contract_identity()
    output = tmp_path / "incomplete"
    replace = converter.os.replace

    def interrupted(source, target):
        if Path(target).name == "provenance.json":
            (output / "user-note").write_bytes(b"keep")
            raise OSError("independent interruption")
        replace(source, target)

    monkeypatch.setattr(converter.os, "replace", interrupted)
    with pytest.raises(OSError, match="interruption"):
        converter._publish(source, output, files, revision)
    assert [p.name for p in output.iterdir()] == ["user-note"]
    assert (output / "user-note").read_bytes() == b"keep"


def test_policy_isolation_fresh_process_and_during_fit_mutation(dataset, tmp_path):
    pipeline = Pipeline.for_simulation(tmp_path / "state")
    run = pipeline.create_simulation(
        dataset,
        SimulationConfig(
            source_policy=POLICY,
            page_budget=1,
            report_predictions=True,
            evaluator_id="independent-count",
        ),
    )
    observed = []

    class Spy(FixtureModel):
        def fit(self, examples, **kwargs):
            assert len(examples) == 1 and examples[0].page.split is Split.TRAIN
            assert [r.text for r in examples[0].regions] == [" q. ", " e\u0301 \r\n漢 & ", ""]
            observed.append("fit")
            return super().fit(examples, **kwargs)

        def predict(self, pages, **kwargs):
            for page in pages:
                assert set(page.model_dump()) == {
                    "id",
                    "document_id",
                    "width",
                    "height",
                    "image_uri",
                    "image_sha256",
                    "split",
                    "source_policy",
                    "source_image",
                }
                assert page.document_id is None and page.split is not Split.TEST
            observed.append(kwargs["purpose"])
            return super().predict(pages, **kwargs)

    class Evaluator:
        identifier = "independent-count"

        def __call__(self, examples, predictions):
            assert len(examples) == 50 and all(e.page.split is Split.VALIDATION for e in examples)
            observed.append("evaluate")
            return {"pages": len(examples)}

    done = pipeline.step_simulation(run.id, Spy(), evaluator=Evaluator())
    assert observed == ["fit", "pool", "validation", "evaluate"]
    with pytest.raises(ValueError):
        LocalOracle(done.dataset).reveal((done.dataset.pages[-1].id,))
    root = Path(converter.__file__).resolve().parents[3]
    code = """import sys
from pathlib import Path
from active_ocr.pipeline import Pipeline
from active_ocr.integrations.simulation import LocalOracle
p=Pipeline.for_simulation(Path(sys.argv[1]))
r=p.get_simulation(sys.argv[2]); LocalOracle(r.dataset)
assert all(x.document_id is None for x in r.dataset.pages)
p.export_simulation(r.id, Path(sys.argv[3])); print(r.model_dump_json())
"""
    child = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "state"), run.id, str(tmp_path / "export")],
        capture_output=True,
        text=True,
        check=True,
        env=dict(os.environ, PYTHONPATH=str(root / "src")),
    )
    assert json.loads(child.stdout) == done.model_dump(mode="json")
    # Hundreds of serialized pool predictions exceed Python csv's default 128 KiB field cap.
    # CSV itself has no such cap; use a bounded reader appropriate to this synthetic fixture.
    previous_limit = csv.field_size_limit(2 * 1024**2)
    try:
        with (tmp_path / "export/rounds.csv").open() as stream:
            row = next(csv.DictReader(stream))
    finally:
        csv.field_size_limit(previous_limit)
    assert (row["source_policy"], row["document_grouping"], row["engineering_only"]) == (
        POLICY,
        "unknown",
        "true",
    )
    assert row["kind"] == "fixture-simulation-not-ocr-evidence"
    pending = pipeline.create_simulation(
        dataset, SimulationConfig(source_policy=POLICY, page_budget=1)
    )

    class Mutate(FixtureModel):
        def fit(self, examples, **kwargs):
            path = dataset.parent / "source/PublicData/Validation/page/doc.xml"
            path.write_bytes(b"drift during fit")
            return super().fit(examples, **kwargs)

    with pytest.raises(ValueError):
        pipeline.step_simulation(pending.id, Mutate())
    assert pipeline.get_simulation(pending.id) == pending
    with pytest.raises(ValueError):
        pipeline.step_simulation(done.id, Spy(), evaluator=Evaluator())


def test_previous_application_writer_defaults_and_cas(tmp_path):
    root = Path(converter.__file__).resolve().parents[3]
    old = tmp_path / "old"
    old.mkdir()
    data = subprocess.check_output(
        ["git", "-C", str(root), "archive", "dee5e1a5ecbbc326921e0f4019792c8f7e5deab2"]
    )
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        archive.extractall(old, filter="data")
    code = """import sys
from pathlib import Path
from active_ocr.integrations.simulation import create_fixture
from active_ocr.models import SimulationConfig
from active_ocr.pipeline import Pipeline
root=Path(sys.argv[1]); manifest=create_fixture(root/'data')
p=Pipeline.for_simulation(root/'state')
r=p.create_simulation(manifest, SimulationConfig(page_budget=3))
p.step_simulation(r.id); print(r.id)
"""
    child = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=old,
        capture_output=True,
        text=True,
        check=True,
        env=dict(os.environ, PYTHONPATH=str(old / "src")),
    )
    run_id = child.stdout.strip()
    pipeline = Pipeline.for_simulation(tmp_path / "state")
    with sqlite3.connect(pipeline.settings.database) as db:
        payload = db.execute("SELECT payload FROM records WHERE key=?", (run_id,)).fetchone()[0]
    assert "source_policy" not in payload
    done = pipeline.run_simulation(run_id)
    assert done.config.source_policy is SourcePolicy.KNOWN_DOCUMENT
    assert [r.labelled_count for r in done.rounds] == [2, 3]
    pipeline.export_simulation(run_id, tmp_path / "export")
    with (tmp_path / "export/rounds.csv").open() as stream:
        assert all(
            row["document_grouping"] == "known" and row["engineering_only"] == "false"
            for row in csv.DictReader(stream)
        )
