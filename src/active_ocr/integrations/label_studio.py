"""Small Label Studio client and line-annotation payload mappings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from active_ocr.models import Annotation, Box, Page, Prediction, Region


def pixels_to_percent(value: float, total: float) -> float:
    if total <= 0 or not 0 <= value <= total:
        raise ValueError("coordinate must fit within a positive image dimension")
    return value / total * 100


def percent_to_pixels(value: float, total: float) -> float:
    if total <= 0 or not 0 <= value <= 100:
        raise ValueError("percentage must be between 0 and 100")
    return value / 100 * total


def page_to_task(page: Page, batch_id: str) -> dict[str, Any]:
    return {
        "data": {"image": page.image_uri},
        "meta": {
            "page_id": page.id,
            "document_id": page.document_id,
            "batch_id": batch_id,
            "width": page.width,
            "height": page.height,
            "split": page.split.value,
        },
    }


def _validate_box(box: Box, page: Page) -> None:
    if box.x + box.width > page.width or box.y + box.height > page.height:
        raise ValueError("line box extends beyond the page")


def prediction_to_result(prediction: Prediction, page: Page) -> list[dict[str, Any]]:
    """Convert predicted line boxes/text into Label Studio result items."""

    if prediction.page_id != page.id:
        raise ValueError("prediction and page do not match")
    items: list[dict[str, Any]] = []
    for region in prediction.regions:
        _validate_box(region.box, page)
        value = {
            "x": pixels_to_percent(region.box.x, page.width),
            "y": pixels_to_percent(region.box.y, page.height),
            "width": pixels_to_percent(region.box.width, page.width),
            "height": pixels_to_percent(region.box.height, page.height),
            "rotation": 0,
            "rectanglelabels": ["Text Line"],
        }
        items.extend(
            [
                {
                    "id": region.id,
                    "from_name": "bbox",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "original_width": page.width,
                    "original_height": page.height,
                    "value": value,
                },
                {
                    "id": region.id,
                    "from_name": "transcription",
                    "to_name": "image",
                    "type": "textarea",
                    "value": {"text": [region.text]},
                },
            ]
        )
    return items


def result_to_annotation(result: Sequence[Mapping[str, Any]], page: Page) -> Annotation:
    """Convert Label Studio result items back into one page annotation."""

    text_by_id: dict[str, str] = {}
    illegible_ids: set[str] = set()
    boxes: dict[str, Box] = {}
    for item in result:
        region_id = str(item.get("id", ""))
        value = item.get("value", {})
        if not region_id or not isinstance(value, Mapping):
            continue
        if item.get("type") == "rectanglelabels":
            box = Box(
                x=percent_to_pixels(float(value["x"]), page.width),
                y=percent_to_pixels(float(value["y"]), page.height),
                width=percent_to_pixels(float(value["width"]), page.width),
                height=percent_to_pixels(float(value["height"]), page.height),
            )
            _validate_box(box, page)
            boxes[region_id] = box
        elif item.get("type") == "textarea":
            texts = value.get("text", [])
            text_by_id[region_id] = str(texts[0]) if texts else ""
        elif item.get("type") == "choices" and "Illegible" in value.get("choices", []):
            illegible_ids.add(region_id)
    regions = tuple(
        Region(
            id=region_id,
            box=box,
            text=text_by_id.get(region_id, ""),
            illegible=region_id in illegible_ids,
        )
        for region_id, box in boxes.items()
    )
    return Annotation(page_id=page.id, regions=regions)


class LabelStudioClient:
    """Publish batches and poll annotations through Label Studio's REST API."""

    def __init__(self, base_url: str, token: str | None, project_id: int | None) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.project_id = project_id

    @property
    def headers(self) -> dict[str, str]:
        if not self.token:
            raise RuntimeError("Label Studio token is not configured")
        return {"Authorization": f"Token {self.token}"}

    def publish(
        self,
        batch_id: str,
        pages: Sequence[Page],
        predictions: Sequence[Prediction] = (),
    ) -> str:
        """Import a tagged task batch and return its internal batch ID."""

        if self.project_id is None:
            raise RuntimeError("Label Studio project ID is not configured")
        by_page = {prediction.page_id: prediction for prediction in predictions}
        tasks = []
        for page in pages:
            task = page_to_task(page, batch_id)
            if page.id in by_page:
                task["predictions"] = [
                    {
                        "model_version": by_page[page.id].model_id,
                        "result": prediction_to_result(by_page[page.id], page),
                    }
                ]
            tasks.append(task)
        response = httpx.post(
            f"{self.base_url}/api/projects/{self.project_id}/import",
            headers=self.headers,
            json=tasks,
            timeout=30,
        )
        response.raise_for_status()
        return batch_id

    def _export_batch(self, batch_id: str) -> list[dict[str, Any]]:
        if self.project_id is None:
            raise RuntimeError("Label Studio project ID is not configured")
        response = httpx.get(
            f"{self.base_url}/api/projects/{self.project_id}/export",
            params={"exportType": "JSON"},
            headers=self.headers,
            timeout=30,
        )
        response.raise_for_status()
        return [
            task for task in response.json() if task.get("meta", {}).get("batch_id") == batch_id
        ]

    def is_complete(self, batch_id: str) -> bool:
        tasks = self._export_batch(batch_id)
        return bool(tasks) and all(task.get("annotations") for task in tasks)

    def fetch(self, batch_id: str, pages: Mapping[str, Page]) -> tuple[Annotation, ...]:
        annotations = []
        for task in self._export_batch(batch_id):
            page = pages[str(task["meta"]["page_id"])]
            latest = max(task["annotations"], key=lambda item: item.get("updated_at", ""))
            annotations.append(result_to_annotation(latest.get("result", []), page))
        return tuple(annotations)
