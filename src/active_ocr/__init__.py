"""Active OCR: a human-in-the-loop OCR research pipeline."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("active-ocr")
except PackageNotFoundError:  # Package metadata is unavailable in a source checkout.
    __version__ = "0.1.0"

__all__ = ["__version__"]
