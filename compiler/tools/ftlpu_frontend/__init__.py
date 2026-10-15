"""Small model-to-StableHLO connection layer; physical binding belongs to the backend."""

from .pipeline import export_model

__all__ = ["export_model"]
