"""Shallow integrations; legacy public exports load only when requested."""

from importlib import import_module

__all__ = ["GPUClient", "LabelStudioClient", "SQLiteStore", "load_image_pages"]

_MODULES = {
    "GPUClient": "gpu_client",
    "LabelStudioClient": "label_studio",
    "SQLiteStore": "storage",
    "load_image_pages": "local_data",
}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_MODULES[name]}"), name)
    globals()[name] = value
    return value
