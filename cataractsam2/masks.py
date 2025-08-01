from __future__ import annotations
import os
from pathlib import Path
from PIL import Image
import numpy as np

__all__ = ["Masks"]

def Masks(out_dir: str | os.PathLike):

    from .ui_widget import video_segments  # lazy import to avoid cycles
    from .ui_widget import CLASSES 

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f_idx, obj_map in video_segments.items():
        if not obj_map:
            continue
        for obj_id, mask in obj_map.items():
            mask_2d = np.squeeze(mask)
            if mask_2d.ndim != 2:
                mask_2d = mask_2d.squeeze(0)

            mask_u8 = (mask_2d.astype(np.uint8) * 255)
            classname = CLASSES.get(obj_id, f"obj{obj_id}").replace(" ", "_")
            fname = f"frame_{f_idx:03d}_{classname}.png"

            Image.fromarray(mask_u8, mode="L").save(out_dir / fname)

    print("✅  Saved masks →", out_dir)
