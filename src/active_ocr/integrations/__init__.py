"""Shallow integrations with local data, storage, Label Studio, and the GPU."""

from .gpu_client import GPUClient
from .label_studio import LabelStudioClient
from .local_data import load_image_pages
from .storage import SQLiteStore

__all__ = ["GPUClient", "LabelStudioClient", "SQLiteStore", "load_image_pages"]
