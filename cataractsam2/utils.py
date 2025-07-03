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

# ‑‑‑ mask / point visualisers ‑‑‑ ------------------------------------------------

def show_mask(mask: np.ndarray, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), [0.6]])
    else:
        cmap  = plt.get_cmap("tab10")
        color = np.array([*cmap(0 if obj_id is None else obj_id)[:3], 0.6])

    h, w = mask.shape[-2:]
    ax.imshow(mask.reshape(h, w, 1) * color.reshape(1, 1, -1))


def show_points(coords: np.ndarray, labels: np.ndarray, ax, marker_size=200):
    pos = coords[labels == 1]
    neg = coords[labels == 0]
    ax.scatter(pos[:, 0], pos[:, 1], color="lime", marker="*", s=marker_size,
               edgecolor="white", linewidth=1.2)
    ax.scatter(neg[:, 0], neg[:, 1], color="red", marker="*", s=marker_size,
               edgecolor="white", linewidth=1.2)


def show_box(box, ax):
    x0, y0 = box[:2]
    w,  h  = box[2] - x0, box[3] - y0
    ax.add_patch(plt.Rectangle((x0, y0), w, h,
                               edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))

# ‑‑‑ base‑64 helper for the Jupyter widget ‑‑‑ -----------------------------------

def encode_image(fp: str | os.PathLike, size=(640, 360)) -> str:
    """
    Resize <fp> for the bbox‑widget and return a base‑64 data URI.
    """
    img = Image.open(fp)
    # store original dims in a global that ui_widget can read
    globals()["orig_size"] = img.size

    img_small = img.resize(size)
    with BytesIO() as buf:
        img_small.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
    return "data:image/jpeg;base64," + b64
