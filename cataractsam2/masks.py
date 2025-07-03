from __future__ import annotations
import os
from pathlib import Path
from PIL import Image
import numpy as np

__all__ = ["Masks"]

def Masks(
    video_segments: dict[int, dict[int, np.ndarray]],
    out_dir: str | os.PathLike
):
    """
    video_segments = {
        frame_idx: {
            obj_id: boolean_mask_numpy_array,
            …
        },
        …
    }
    Writes one multi‑class PNG per frame, where each pixel value is the object ID (0=background).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f_idx, obj_map in video_segments.items():
        if not obj_map:
            # nothing to draw on this frame
            continue

        # Start with a blank 0‑filled canvas same shape as one of the masks
        sample_mask = next(iter(obj_map.values()))
        canvas = np.zeros_like(sample_mask, dtype=np.uint8)

        # Paint each object's pixels with its ID
        for obj_id, mask in obj_map.items():
            # ensure boolean mask
            canvas[mask.astype(bool)] = obj_id

        # Save as a single‑channel grayscale PNG
        Image.fromarray(canvas, mode="L") \
             .save(out_dir / f"frame_{f_idx:03d}.png")

    print("✅  Saved masks →", out_dir)
