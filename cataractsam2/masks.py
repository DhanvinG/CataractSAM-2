from __future__ import annotations
import os
from pathlib import Path
from PIL import Image
import numpy as np

__all__ = ["Masks"]

def Masks(out_dir: str | os.PathLike):
    """Write propagated masks as PNG files.

    Uses the global ``video_segments`` created by :func:`Propagate` and
    writes one PNG per ``(frame, object)`` to ``out_dir``.  Each PNG is a
    single-channel image where mask pixels have value 255 and background
    is 0.
    """

    from .ui_widget import video_segments  # lazy import to avoid cycles

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f_idx, obj_map in video_segments.items():
        if not obj_map:
            continue
        for obj_id, mask in obj_map.items():
            # ensure uint8 0/255 values
            mask_u8 = (mask.astype(bool) * 255).astype(np.uint8)
            Image.fromarray(mask_u8, mode="L") \
                 .save(out_dir / f"frame_{f_idx:03d}_obj_{obj_id:03d}.png")

    print("✅  Saved masks →", out_dir)
