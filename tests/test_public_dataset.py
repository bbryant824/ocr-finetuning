"""Synthetic READ mapping/policy verification; no full real-source conversion or model work."""

import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from active_ocr.integrations import public_dataset as public
from active_ocr.integrations.simulation import FixtureModel, LocalOracle, SourcePage, _read_source
from active_ocr.models import Page, SimulationConfig, SimulationPage, SourcePolicy, Split
from active_ocr.pipeline import Pipeline

POLICY = SourcePolicy.READ2016
NS = public.NAMESPACE


def xml_bytes(name="page.JPG"):
    # Region gaps and shuffled nodes are deliberate; line indices are contiguous per region.
    return f'''<PcGts xmlns="{NS}"><Page imageFilename="{name}" imageWidth="10" imageHeight="12">
<ReadingOrder><OrderedGroup><RegionRefIndexed index="5" regionRef="b"/>
<RegionRefIndexed index="0" regionRef="a"/></OrderedGroup></ReadingOrder>
<TextRegion id="b" custom="readingOrder {{index:5;}}"><Coords points="0,0 10,0 10,12"/>
<TextLine id="empty" custom="readingOrder {{index:0;}}"><Coords points="0,0 2,0 1,2"/>
<TextEquiv><Unicode/></TextEquiv></TextLine></TextRegion>
<TextRegion id="a" custom="readingOrder {{index:0;}} structure {{type:marginalia;}}">
<Coords points="0,0 10,0 10,12"/>
<TextLine id="second" custom="readingOrder {{index:1;}}"><Coords points="2,3 8,4 4,10"/>
<Baseline points="2,4 4,4"/><TextEquiv><Unicode> é &amp;  x </Unicode></TextEquiv></TextLine>
<TextLine id="first" custom="readingOrder {{index:0;}} unclear {{offset:0; length:1;}}
 abbrev {{offset:0; length:1; expansion:test;}} textStyle {{offset:0; length:1;}}
 sic {{offset:0; length:1;}}"><Coords points="0,0 2,0 1,2"/>
<TextStyle strikethrough="true"/><TextEquiv><Unicode>A&#13;&#10; B</Unicode></TextEquiv></TextLine>
<TextEquiv><Unicode>not a line target</Unicode></TextEquiv></TextRegion>
</Page></PcGts>'''.encode()


def rewrite(data, edit):
    tree = ET.fromstring(data)
    edit(tree)
    return ET.tostring(tree)


def test_literal_order_gaps_and_envelope():
    meta, lines = public.parse_page(xml_bytes(), "page.JPG")
    assert [r.id for r in lines] == ["first", "second", "empty"]
    assert [r.text for r in lines] == ["A\r\n B", " é &  x ", ""]
    assert not any(r.illegible for r in lines)
    assert lines[1].box.model_dump() == dict(x=2, y=3, width=6, height=7)
    assert meta == dict(
        width=10, height=12, region_count=2, line_count=3, empty_lines=1, missing_baselines=2
    )


@pytest.mark.parametrize(
    "fault",
    [
        "region_duplicate",
        "region_unknown",
        "region_missing",
        "line_gap",
        "line_duplicate",
        "duplicate_id",
        "text_missing",
        "text_multiple",
        "text_nested",
        "coords_missing",
        "degenerate",
        "out_of_bounds",
        "infinite",
        "fractional",
        "unknown_block",
        "duplicate_order",
        "unknown_flag",
        "rotation",
        "namespace",
        "image_path",
        "dtd",
    ],
)
def test_parser_rejects_ambiguous_or_unsupported_input(fault):
    data = xml_bytes()
    root = ET.fromstring(data)
    group = root.find(f".//{{{NS}}}OrderedGroup")
    lines = root.findall(f".//{{{NS}}}TextLine")
    line = lines[-1]
    if fault == "region_duplicate":
        group[1].set("index", "5")
    elif fault == "region_unknown":
        group[1].set("regionRef", "unknown")
    elif fault == "region_missing":
        group.remove(group[1])
    elif fault == "line_gap":
        line.set("custom", "readingOrder {index:3;}")
    elif fault == "line_duplicate":
        line.set("custom", "readingOrder {index:1;}")
    elif fault == "duplicate_id":
        line.set("id", "empty")
    elif fault == "text_missing":
        line.remove(line.find(f"{{{NS}}}TextEquiv"))
    elif fault == "text_multiple":
        line.append(ET.fromstring(f'<TextEquiv xmlns="{NS}"><Unicode>x</Unicode></TextEquiv>'))
    elif fault == "text_nested":
        ET.SubElement(line.find(f".//{{{NS}}}Unicode"), f"{{{NS}}}TextStyle")
    elif fault == "coords_missing":
        line.remove(line.find(f"{{{NS}}}Coords"))
    elif fault in ("degenerate", "out_of_bounds", "infinite", "fractional"):
        value = {
            "degenerate": "1,1 1,2 1,3",
            "out_of_bounds": "0,0 11,0 1,2",
            "infinite": "0,0 inf,0 1,2",
            "fractional": "0,0 1.1,0 1,2",
        }[fault]
        line.find(f"{{{NS}}}Coords").set("points", value)
    elif fault == "unknown_block":
        line.set("custom", "readingOrder {index:0;} illegible {value:true;}")
    elif fault == "duplicate_order":
        line.set("custom", "readingOrder {index:0;} readingOrder {index:0;}")
    elif fault == "unknown_flag":
        line.set("illegible", "true")
    elif fault == "rotation":
        root.find(f"{{{NS}}}Page").set("orientation", "90")
    elif fault == "namespace":
        root.tag = "PcGts"
    elif fault == "image_path":
        root.find(f"{{{NS}}}Page").set("imageFilename", "../page.JPG")
    data = ET.tostring(root)
    if fault == "dtd":
        data = b'<!DOCTYPE PcGts [<!ENTITY ext SYSTEM "file:///nonexistent">]>' + data
    with pytest.raises((ValueError, ET.ParseError)):
        public.parse_page(data, "page.JPG")


@pytest.mark.parametrize("fault", ["mode", "orientation", "size", "format", "corrupt"])
def test_image_refuses_transform_or_unsupported_bytes(fault):
    image = Image.new("L" if fault == "mode" else "RGB", (10, 12))
    stream = io.BytesIO()
    exif = Image.Exif()
    if fault == "orientation":
        exif[274] = 6
    image.save(stream, format="PNG" if fault == "format" else "JPEG", exif=exif)
    data = b"not an image" if fault == "corrupt" else stream.getvalue()
    with pytest.raises((ValueError, OSError)):
        public.validate_image(data, 9 if fault == "size" else 10, 12)


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-read-source")
    stream = io.BytesIO()
    Image.new("RGB", (10, 12)).save(stream, format="JPEG")
    for folder, count in [("Training", 350), ("Validation", 50)]:
        base = root / "PublicData" / folder
        (base / "Images").mkdir(parents=True)
        (base / "page/page").mkdir(parents=True)
        for index in range(count):
            name = f"page{index:04d}"
            (base / f"Images/{name}.JPG").write_bytes(
                stream.getvalue() + f"{folder}:{index}".encode()
            )
            (base / f"page/page/{name}.xml").write_bytes(xml_bytes(f"{name}.JPG"))
        (base / "page/doc.xml").write_bytes(b"<documents grouping='unknown'/>")
        (base / "page/list").write_bytes(b"synthetic list")
    files = tuple(
        public.SourceFile(
            uri=p.relative_to(root).as_posix(), bytes=p.stat().st_size, sha256=public.file_digest(p)
        )
        for p in sorted(root.rglob("*"))
        if p.is_file()
    )
    return root, files


@pytest.fixture(scope="module")
def prepared(staged, tmp_path_factory):
    root, files = staged
    parent = tmp_path_factory.mktemp("synthetic-read-prepared")
    manifest = public._publish(root, parent / "dataset", files, "a" * 40)
    return manifest.parent


@pytest.fixture
def dataset(prepared, tmp_path):
    result = tmp_path / "dataset"
    shutil.copytree(prepared, result)
    return result


def test_required_nullable_document_field_is_simulation_only():
    base = dict(id="p", image_uri="image.JPG", width=10, height=12, split="train")
    with pytest.raises(ValidationError):
        Page(**base, document_id=None)
    for document in (None, "", "-1"):
        row = dict(
            base,
            document_id=document,
            image_sha256="a" * 64,
            source_policy=POLICY,
            source_provenance=dict(uri="provenance.json", sha256="b" * 64),
        )
        if document is None:
            assert SourcePage(**row).document_id is None
        else:
            with pytest.raises(ValueError):
                SourcePage(**row)
    with pytest.raises(ValueError):
        SourcePage(**base, image_sha256="a" * 64)
    with pytest.raises(ValueError):
        SourcePage(**base, document_id=None, image_sha256="a" * 64)
    with pytest.raises(ValueError):
        SimulationPage(**base, document_id=None, image_sha256="a" * 64, source_image="x")


def test_deterministic_output_and_original_bytes(staged, prepared, tmp_path):
    source, files = staged
    other = public._publish(source, tmp_path / "other", files, "a" * 40)
    for name in ("pages.jsonl", "provenance.json"):
        assert (prepared / name).read_bytes() == (other.parent / name).read_bytes()
    provenance = public.ReadProvenance.model_validate_json(
        (prepared / "provenance.json").read_bytes()
    )
    assert len(provenance.files) == 804 and len(provenance.pages) == 400
    for file in files:
        assert (prepared / "source" / file.uri).read_bytes() == (source / file.uri).read_bytes()


@pytest.mark.parametrize(
    "fault",
    [
        "caller",
        "null_without_policy",
        "mixed_policy",
        "known_id",
        "test",
        "provenance_hash",
        "provenance_missing",
        "wrong_archive",
        "wrong_version",
        "page_map",
        "raw_xml",
        "doc_xml",
        "image",
        "traversal",
        "symlink",
        "duplicate_image",
        "target_text",
    ],
)
def test_admission_and_integrity_fail_closed(dataset, fault):
    manifest = dataset / "pages.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    prov = json.loads((dataset / "provenance.json").read_text())
    if fault == "caller":
        with pytest.raises(ValueError, match="policy"):
            _read_source(manifest.read_bytes(), dataset)
        return
    if fault == "null_without_policy":
        rows[0].pop("source_policy")
    elif fault == "mixed_policy":
        rows[0].update(
            source_policy="known-document-v1", document_id="known", source_provenance=None
        )
    elif fault == "known_id":
        rows[0]["document_id"] = "-1"
    elif fault == "test":
        rows[0]["split"] = "test"
    elif fault == "provenance_hash":
        rows[0]["source_provenance"]["sha256"] = "0" * 64
    elif fault == "provenance_missing":
        (dataset / "provenance.json").unlink()
    elif fault in ("wrong_archive", "wrong_version", "page_map", "duplicate_image"):
        if fault == "wrong_archive":
            prov["archive_sha256"] = "0" * 64
        elif fault == "wrong_version":
            prov["dataset_version"] = "2.0"
        elif fault == "page_map":
            prov["pages"].pop()
        else:
            target = rows[1]["image_uri"]
            first = rows[0]["image_uri"]
            data = (dataset / first).read_bytes()
            (dataset / target).write_bytes(data)
            for file in prov["files"]:
                if file["uri"] == target:
                    file.update(bytes=len(data), sha256=public.digest(data))
            rows[1]["image_sha256"] = public.digest(data)
        data = json.dumps(prov).encode()
        (dataset / "provenance.json").write_bytes(data)
        for row in rows:
            row["source_provenance"]["sha256"] = public.digest(data)
    elif fault in ("raw_xml", "doc_xml", "image"):
        uri = {
            "raw_xml": prov["pages"][0]["xml_uri"],
            "doc_xml": "source/PublicData/Training/page/doc.xml",
            "image": rows[0]["image_uri"],
        }[fault]
        with (dataset / uri).open("ab") as stream:
            stream.write(b"changed")
    elif fault == "traversal":
        rows[0]["image_uri"] = "../outside.JPG"
    elif fault == "symlink":
        path = dataset / rows[0]["image_uri"]
        data = path.read_bytes()
        path.unlink()
        target = dataset.parent / "external.JPG"
        target.write_bytes(data)
        path.symlink_to(target)
    elif fault == "target_text":
        rows[0]["regions"][0]["text"] = "edited truth"
    data = "".join(json.dumps(r) + "\n" for r in rows).encode()
    with pytest.raises((ValueError, OSError)):
        _read_source(data, dataset, source_policy=POLICY)


def test_oracle_policy_isolation_resume_export_and_mutation(dataset, tmp_path):
    pipeline = Pipeline.for_simulation(tmp_path / "run")
    manifest = dataset / "pages.jsonl"
    with pytest.raises(ValueError):
        pipeline.create_simulation(manifest, SimulationConfig())
    run = pipeline.create_simulation(
        manifest,
        SimulationConfig(
            source_policy=POLICY, page_budget=1, report_predictions=True, evaluator_id="witness"
        ),
    )

    class Model(FixtureModel):
        def fit(self, examples, **kwargs):
            assert len(examples) == 1 and examples[0].page.split is Split.TRAIN
            assert examples[0].regions[0].text == "A\r\n B"
            assert "source_provenance" not in examples[0].page.model_dump()
            return super().fit(examples, **kwargs)

        def predict(self, pages, **kwargs):
            for page in pages:
                assert page.document_id is None
                assert set(page.model_dump()) == {
                    "id",
                    "document_id",
                    "image_uri",
                    "width",
                    "height",
                    "split",
                    "source_policy",
                    "image_sha256",
                    "source_image",
                }
            return super().predict(pages, **kwargs)

    class Evaluator:
        identifier = "witness"

        def __call__(self, examples, predictions):
            assert len(examples) == 50 and all(e.page.split is Split.VALIDATION for e in examples)
            return {"pages": len(examples)}

    done = pipeline.step_simulation(run.id, Model(), evaluator=Evaluator())
    assert done.kind == "fixture-simulation-not-ocr-evidence" and done.complete
    oracle = LocalOracle(done.dataset)
    with pytest.raises(ValueError):
        oracle.reveal((done.dataset.pages[-1].id,))
    pipeline.export_simulation(run.id, tmp_path / "export")
    with (tmp_path / "export/rounds.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert (row["source_policy"], row["document_grouping"], row["engineering_only"]) == (
        POLICY,
        "unknown",
        "true",
    )
    code = """import sys; from pathlib import Path
from active_ocr.pipeline import Pipeline
from active_ocr.integrations.simulation import LocalOracle
p=Pipeline.for_simulation(Path(sys.argv[1]))
r=p.get_simulation(sys.argv[2])
LocalOracle(r.dataset)
assert all(x.document_id is None for x in r.dataset.pages); print(r.config.source_policy)
"""
    child = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "run"), run.id],
        check=True,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(Path("src").resolve())),
    )
    assert child.stdout.strip() == POLICY
    with pytest.raises(ValueError):
        pipeline.step_simulation(run.id, config=SimulationConfig())
    (dataset / "source/PublicData/Training/page/doc.xml").write_bytes(b"changed")
    with pytest.raises(ValueError):
        pipeline.step_simulation(run.id, Model(), evaluator=Evaluator())


@pytest.mark.parametrize(
    "field,value",
    [
        ("split", Split.VALIDATION),
        ("width", 11),
        ("document_id", "fake"),
        ("source_policy", SourcePolicy.KNOWN_DOCUMENT),
    ],
)
def test_snapshot_metadata_tampering_rejects(dataset, tmp_path, field, value):
    pipeline = Pipeline.for_simulation(tmp_path / "run")
    snapshot = LocalOracle.freeze(dataset / "pages.jsonl", pipeline.store, source_policy=POLICY)
    first = snapshot.pages[0].model_copy(update={field: value})
    changed = snapshot.model_copy(update={"pages": (first, *snapshot.pages[1:])})
    with pytest.raises(ValueError):
        LocalOracle(changed)


@pytest.mark.parametrize("fault", ["source", "persisted_image", "persisted_manifest"])
def test_mutation_during_freeze_rejects(dataset, tmp_path, monkeypatch, fault):
    pipeline = Pipeline.for_simulation(tmp_path / "run")
    save = pipeline.store.save_artifact
    changed = False

    def mutate(namespace, data, suffix=""):
        nonlocal changed
        path = save(namespace, data, suffix)
        if not changed:
            changed = True
            if fault == "source":
                (dataset / "source/PublicData/Training/page/doc.xml").write_bytes(b"changed")
            elif fault == "persisted_image":
                path.write_bytes(b"changed")
        if fault == "persisted_manifest" and namespace == "simulation-source":
            path.write_bytes(b"changed")
        return path

    monkeypatch.setattr(pipeline.store, "save_artifact", mutate)
    with pytest.raises(ValueError):
        LocalOracle.freeze(dataset / "pages.jsonl", pipeline.store, source_policy=POLICY)


def make_tar(path, staged, fault=None):
    root, files = staged
    with tarfile.open(path, "w:gz") as tar:
        for file in files:
            if fault == "missing" and file.uri.endswith("Training/page/doc.xml"):
                continue
            tar.add(root / file.uri, arcname=file.uri, recursive=False)
        for image, _, _ in public._page_paths(files):
            link = tarfile.TarInfo(image.replace("/Images/", "/page/"))
            link.type = tarfile.SYMTYPE
            link.linkname = "../Images/" + Path(image).name
            if fault == "link_escape":
                link.linkname = "../../outside.JPG"
            tar.addfile(link)
        if fault in ("traversal", "duplicate", "special", "extra"):
            name = {
                "traversal": "../escape",
                "duplicate": files[0].uri,
                "special": "PublicData/fifo",
                "extra": "PublicData/Training/page/extra",
            }[fault]
            member = tarfile.TarInfo(name)
            if fault == "special":
                member.type = tarfile.FIFOTYPE
            tar.addfile(member, io.BytesIO(b""))


@pytest.mark.parametrize(
    "fault", ["traversal", "duplicate", "special", "extra", "missing", "link_escape"]
)
def test_archive_members_reject_without_extracting(staged, tmp_path, fault):
    archive = tmp_path / "test.tgz"
    make_tar(archive, staged, fault)
    with pytest.raises(ValueError):
        public._tar_inventory(archive)
    assert not (tmp_path / "PublicData").exists()


def test_archive_member_inventory_and_public_pin(staged, tmp_path):
    archive = tmp_path / "test.tgz"
    make_tar(archive, staged)
    assert public._tar_inventory(archive) == staged[1]
    with pytest.raises(ValueError, match="pinned"):
        public.convert_read2016(archive, staged[0], tmp_path / "out", source_policy=POLICY)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("collision", ["directory", "file", "symlink"])
def test_publication_never_overwrites_existing(staged, tmp_path, collision):
    dest = tmp_path / "out"
    if collision == "directory":
        dest.mkdir()
    elif collision == "file":
        dest.write_bytes(b"keep")
    else:
        dest.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        public._publish(staged[0], dest, staged[1], "a" * 40)
    assert dest.exists() or dest.is_symlink()
    if collision == "file":
        assert dest.read_bytes() == b"keep"


def test_interrupted_publication_cleans_only_owned_files(staged, tmp_path, monkeypatch):
    dest = tmp_path / "out"
    original = public.os.replace

    def fail(source, target):
        if Path(target).name == "provenance.json":
            (dest / "unrelated").write_bytes(b"preserve")
            raise OSError("injected I/O interruption")
        return original(source, target)

    monkeypatch.setattr(public.os, "replace", fail)
    with pytest.raises(OSError, match="interruption"):
        public._publish(staged[0], dest, staged[1], "a" * 40)
    assert sorted(p.name for p in dest.iterdir()) == ["unrelated"]


def test_concurrent_publication_has_one_winner(staged, tmp_path):
    dest = tmp_path / "out"

    def attempt():
        try:
            return public._publish(staged[0], dest, staged[1], "a" * 40)
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: attempt(), range(2)))
    assert results.count(None) == 1
    assert len(_read_source((dest / "pages.jsonl").read_bytes(), dest, source_policy=POLICY)) == 400


@pytest.mark.parametrize(
    "member",
    [
        "schema_version",
        "mapping_version",
        "source_policy",
        "dataset_record",
        "archive_sha256",
        "engineering_only",
    ],
)
def test_provenance_cannot_omit_explicit_identity(prepared, member):
    raw = json.loads((prepared / "provenance.json").read_bytes())
    raw.pop(member)
    with pytest.raises(ValueError):
        public.ReadProvenance.model_validate(raw)


@pytest.mark.parametrize("uri", [".", "..", "../x", "/x", "a/../x", "a//x", "a\\x"])
def test_paths_are_canonical_relative_uris(uri):
    with pytest.raises(ValueError):
        public.relative_uri(uri)


def test_copied_byte_mutation_prevents_publication(staged, tmp_path, monkeypatch):
    write = public._write_synced
    corrupted = False

    def corrupt(path, data):
        nonlocal corrupted
        write(path, data)
        if not corrupted:
            corrupted = True
            path.write_bytes(data + b"unexpected copy mutation")

    monkeypatch.setattr(public, "_write_synced", corrupt)
    with pytest.raises(ValueError, match="changed"):
        public._publish(staged[0], tmp_path / "out", staged[1], "a" * 40)
    assert not (tmp_path / "out").exists()


def test_staged_symlink_directory_is_not_followed(staged, tmp_path):
    (tmp_path / "PublicData").symlink_to(staged[0] / "PublicData", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        public.verify_inventory(tmp_path, staged[1], "PublicData")


def test_source_change_during_model_work_prevents_commit(dataset, tmp_path):
    pipeline = Pipeline.for_simulation(tmp_path / "run")
    run = pipeline.create_simulation(
        dataset / "pages.jsonl", SimulationConfig(source_policy=POLICY, page_budget=1)
    )

    class Mutating(FixtureModel):
        def fit(self, examples, **kwargs):
            (dataset / "source/PublicData/Validation/page/doc.xml").write_bytes(b"changed")
            return super().fit(examples, **kwargs)

    with pytest.raises(ValueError, match="changed"):
        pipeline.step_simulation(run.id, Mutating())
    assert pipeline.get_simulation(run.id) == run


def test_lost_publication_response_preserves_complete_dataset(staged, tmp_path, monkeypatch):
    original = public.os.replace
    dest = tmp_path / "out"

    def lost_response(source, target):
        original(source, target)
        if Path(target).name == "pages.jsonl":
            raise OSError("lost publication response")

    monkeypatch.setattr(public.os, "replace", lost_response)
    with pytest.raises(OSError, match="lost publication"):
        public._publish(staged[0], dest, staged[1], "a" * 40)
    assert len(_read_source((dest / "pages.jsonl").read_bytes(), dest, source_policy=POLICY)) == 400
