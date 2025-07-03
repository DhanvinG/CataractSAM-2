import os, numpy as np
from PIL import Image
from typing import Mapping

def Masks(directory: str | os.PathLike):
    """
    Save one PNG per (frame, object) in `video_segments`.

    • Call it with an explicit path: generate_multiclass_masks("/tmp/out")
    • Or pass any variable that holds a path: generate_multiclass_masks(current_mask_dir)

    Raises
    ------
    ValueError  if `directory` is not provided or is empty.
    """
    if not directory:
        raise ValueError("Please supply an output directory path.")

    # allow pathlib.Path as well as str
    directory = os.fspath(directory)
    os.makedirs(directory, exist_ok=True)

    for frame_idx, obj_masks in video_segments.items():
        for obj_id, mask in obj_masks.items():
            # squeeze to 2D
            mask_2d = np.squeeze(mask)
            if mask_2d.ndim != 2:
                mask_2d = mask_2d.squeeze(0)

            out_path = os.path.join(
                directory, f"frame_{frame_idx:03d}_obj_{obj_id}.png"
            )
            # convert boolean or {0,1} mask to 0/255 uint8
            img = Image.fromarray((mask_2d.astype(np.uint8) * 255), mode="L")
            img.save(out_path)

    print(f"✅ Saved masks to: {directory}")
