"""
Plotting helpers shared by widget & scripts.
Keep **pure‑Python** – no torch/CUDA here.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import base64, os

__all__ = ["show_mask", "show_points", "show_box", "encode_image"]

orig_size: tuple[int,int] = (0, 0)

# ‑‑‑ mask / point visualisers ‑‑‑ ------------------------------------------------

def show_mask(mask: np.ndarray, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords: np.ndarray, labels: np.ndarray, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0], pos_points[:, 1], color="green", marker="*",
        s=marker_size, edgecolor="white", linewidth=1.25
    )
    ax.scatter(
        neg_points[:, 0], neg_points[:, 1], color="red", marker="*",
        s=marker_size, edgecolor="white", linewidth=1.25
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h,
                      edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )

# ‑‑‑ base‑64 helper for the Jupyter widget ‑‑‑ -----------------------------------

def encode_image(fp: str | os.PathLike, size=(640, 360)) -> str:
    """
    Resize <fp> for the bbox‑widget and return a base‑64 data URI.
    """
    img = Image.open(fp)
    orig_w, orig_h = img.size
    globals()["orig_size"] = (orig_w, orig_h)

    img_small = img.resize(size)
    with BytesIO() as buffer:
        img_small.save(buffer, format="JPEG")
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
