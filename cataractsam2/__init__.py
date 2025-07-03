"""Public‑facing API for CataractSAM‑2."""

from .ui_widget import Object, Reset, Visualize, Propagate
from .masks     import generate_multiclass_masks
from .predictor import Predictor

__all__ = [
    "Object", "Reset", "Visualize", "Propagate",
    "generate_multiclass_masks",
    "Predictor",
]
