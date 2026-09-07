import asyncio
import json
from pathlib import Path
from xml.etree import ElementTree

import httpx
import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

from active_ocr.config import Settings, load_settings
from active_ocr.entrypoints.cli import app
from active_ocr.integrations.label_studio import (
    percent_to_pixels,
    pixels_to_percent,
    prediction_to_result,
    result_to_annotation,
)
from active_ocr.integrations.local_data import load_image_pages
from active_ocr.integrations.storage import SQLiteStore
from active_ocr.models import Box, Page, Prediction, Region


def test_local_image_import_and_storage(tmp_path: Path) -> None:
    document = tmp_path / "book-1"
    document.mkdir()
    Image.new("RGB", (120, 80)).save(document / "page-1.png")
    Image.new("RGB", (100, 60)).save(document / "page-2.png")
    manifest = tmp_path / "pages.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps({"path": f"book-1/page-{number}.png", "document_id": "book-1"})
            for number in (1, 2)
        ),
        encoding="utf-8",
    )
    document_root = tmp_path / "documents"
    pages = load_image_pages(manifest, document_root=document_root)
    assert {page.document_id for page in pages} == {"book-1"}
    assert len({page.split for page in pages}) == 1
    page = pages[0]
    assert page.image_uri.startswith("/data/local-files/?d=pages/")
    stored_path = document_root / page.image_uri.split("?d=", 1)[1]
    assert stored_path.is_file()

    store = SQLiteStore(tmp_path / "state.sqlite3", tmp_path / "artifacts")
    store.initialize()
    store.save("page", page.id, page)
    assert store.load("page", page.id, Page) == page
    assert store.save_artifact("reports", b"result", ".json").is_file()


def test_csv_manifest_is_supported(tmp_path: Path) -> None:
    Image.new("RGB", (20, 10)).save(tmp_path / "page.png")
    manifest = tmp_path / "pages.csv"
    manifest.write_text("path,document_id\npage.png,document-1\n", encoding="utf-8")
    page = load_image_pages(manifest, document_root=tmp_path / "documents")[0]
    assert page.document_id == "document-1"


def test_label_studio_mapping_round_trip() -> None:
    page = Page(
        id="page",
        document_id="document",
        image_uri="file:///page.png",
        width=1000,
        height=500,
    )
    region = Region(id="line", box=Box(x=10, y=20, width=300, height=40), text="Example")
    result = prediction_to_result(
        Prediction(
            page_id=page.id,
            experiment_id="experiment",
            round_number=1,
            model_id="m",
            regions=(region,),
        ),
        page,
    )
    annotation = result_to_annotation(result, page)
    assert annotation.regions[0].text == "Example"
    assert annotation.regions[0].box == region.box
    assert percent_to_pixels(pixels_to_percent(250, 1000), 1000) == pytest.approx(250)


def test_config_cli_and_static_files() -> None:
    root = Path(__file__).parents[1]
    settings = load_settings(root / "config" / "default.yaml")
    assert settings.gpu.model == "Qwen/Qwen3-VL-4B-Instruct"
    assert Settings(poll_seconds=1).poll_seconds == 1
    assert CliRunner().invoke(app, ["--help"]).exit_code == 0
    ElementTree.parse(root / "config" / "label-studio.xml")
    compose = yaml.safe_load((root / "compose.yaml").read_text(encoding="utf-8"))
    assert "label-studio" in compose["services"]


def test_gpu_health_without_loading_qwen() -> None:
    pytest.importorskip("fastapi")
    from active_ocr.entrypoints.gpu_server import create_app

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/health")

    response = asyncio.run(request())
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "not_loaded"}
