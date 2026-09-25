"""cases.data — lazy/cached loading for large benchmark data."""

from .lazy import LazyDataset, json_cached

__all__ = ["LazyDataset", "json_cached"]
