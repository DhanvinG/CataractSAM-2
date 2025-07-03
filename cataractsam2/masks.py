import os, numpy as np
from PIL import Image
from typing import Mapping

def generate_multiclass_masks(video_segments: Mapping, out_dir: str | os.PathLike):
    """Save one grayscale PNG per (frame,obj) from `video_segments`."""
    out_dir = os.fspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for f_idx, per_obj in video_segments.items():
        for obj_id, mask in per_obj.items():
            m = np.squeeze(mask)            # (H,W)
            img = Image.fromarray((m.astype(np.uint8) * 255), mode="L")
            img.save(os.path.join(out_dir, f"frame_{f_idx:03d}_obj_{obj_id}.png"))

    print(f"✅ Saved masks → {out_dir}")
