"""
Interactive Jupyter widget harness for CataractSAM‑2.
"""

from __future__ import annotations
import os, numpy as np, ipywidgets as ipw, matplotlib.pyplot as plt
from IPython.display import HTML, display, clear_output
from traitlets import Unicode
from jupyter_bbox_widget import BBoxWidget
from PIL import Image
from .utils import encode_image, show_mask, show_points, show_box
from . import utils

__all__ = ["setup", "Object", "Visualize", "Propagate", "Reset"]

# ────────────────────────── globals ──────────────────────────
predictor       = None          # set by setup()
video_dir       = ""
frame_names: list[str] = []
inference_state = None

THUMB_SIZE = (640, 360)

ann_frame_idx  = 0
ann_obj_id     = 1

positive_points: list[tuple[float, float]] = []
negative_points: list[tuple[float, float]] = []

prompts       : dict[int, tuple[np.ndarray, np.ndarray]] = {}
mask_cache    : dict[tuple[int, int], np.ndarray]        = {}
video_segments: dict[int, dict[int, np.ndarray]]        = {}

# widget & controls (instantiated in setup)
widget      : BBoxWidget
pos_btn     = ipw.Button(description="POS",       layout=ipw.Layout(width="90px"))
neg_btn     = ipw.Button(description="NEG",       layout=ipw.Layout(width="90px"))
vis_btn     = ipw.Button(description="VISUALIZE", layout=ipw.Layout(width="120px"))
banner      = ipw.HTML()
plot_output = ipw.Output()

# ────────────────────────── setup ───────────────────────────
def setup(pred, frames_dir: str):
    """Initialize the predictor and widget for frames in <frames_dir>."""
    global predictor, video_dir, frame_names, inference_state, widget
    predictor  = pred
    video_dir  = frames_dir
    frame_names = sorted(
        [p for p in os.listdir(video_dir) if p.lower().endswith((".jpg", ".jpeg"))],
        key=lambda p: int(os.path.splitext(p)[0]),
    )
    if not frame_names:
        raise FileNotFoundError(f"No .jpg/.jpeg in {video_dir}")

    inference_state = predictor.init_state(video_path=video_dir)

    # Build the interactive BBoxWidget
    widget = BBoxWidget(
        labels=["positive", "negative"],
        show_dropdown=False,
        show_marker=True,
        marker_size=8,
        label_colors={"positive": "lime", "negative": "red"},
        show_toolbar=False,
    )
    widget.add_traits(mode=Unicode("positive"))
    widget.observe(_on_bboxes_changed, names="bboxes")

    # Hook up the buttons
    pos_btn.on_click(lambda _: _set_mode("positive"))
    neg_btn.on_click(lambda _: _set_mode("negative"))
    vis_btn.on_click(_visualize)
    _set_mode("positive")

# ─────────────────────── internal callbacks ───────────────────
def _on_bboxes_changed(change):
    new, old = change.new or [], change.old or []
    if len(new) <= len(old):
        return

    orig_w, orig_h = utils.orig_size
    small_w, small_h = THUMB_SIZE
    scale_x, scale_y = orig_w / small_w, orig_h / small_h

    for box in new[len(old):]:
        x_small, y_small = box["x"], box["y"]
        x_orig, y_orig = x_small * scale_x, y_small * scale_y
        if widget.mode == "positive":
            positive_points.append((x_orig, y_orig))
        else:
            negative_points.append((x_orig, y_orig))

    widget.bboxes = new

def _set_mode(m: str):
    widget.mode = m
    pos_btn.style.button_color = "#28a745" if m == "positive" else None
    neg_btn.style.button_color = "#d9534f" if m == "negative" else None
    vis_btn.style.button_color = None
    banner.value = f'<h3 style="margin:0;color:{"#28a745" if m=="positive" else "#d9534f"}">MODE: {m.upper()}</h3>'

def _visualize(_=None):
    """Internal handler for the VISUALIZE button."""
    with plot_output:
        plot_output.clear_output(wait=True)
        pts_pos = np.array(positive_points, dtype=np.float32) if positive_points else np.zeros((0,2))
        pts_neg = np.array(negative_points, dtype=np.float32) if negative_points else np.zeros((0,2))
        points = np.vstack([pts_pos, pts_neg])
        labels = np.concatenate([np.ones(len(pts_pos)), np.zeros(len(pts_neg))], dtype=int)

        prompts[ann_obj_id] = (points, labels)
        _, oids, logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        for i, oid in enumerate(oids):
            mask_cache[(ann_frame_idx, oid)] = logits[i].detach().cpu()

        # Show full‑res overlay
        plt.figure(figsize=(9,6))
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        plt.axis("off")
        show_points(points, labels, plt.gca())
        for i, oid in enumerate(oids):
            if oid == ann_obj_id:
                show_mask((logits[i] > 0).cpu().numpy(), plt.gca(), obj_id=oid)
        plt.show()

# ─────────────────────── public API ─────────────────────────
def Object(frame_idx: int, obj_id: int):
    """Start annotating object <obj_id> on frame <frame_idx>."""
    global ann_frame_idx, ann_obj_id, positive_points, negative_points
    clear_output(wait=True)
    ann_frame_idx, ann_obj_id = frame_idx, obj_id

    positive_points.clear()
    negative_points.clear()
    widget.bboxes = []
    widget.image = encode_image(
        os.path.join(video_dir, frame_names[frame_idx]), size=THUMB_SIZE
    )
    plot_output.clear_output(wait=True)
    _set_mode("positive")

    display(HTML("""
      <style>#ctlBox{position:fixed;top:12px;right:12px;z-index:9999;
      background:#111;padding:6px 10px;border-radius:8px;box-shadow:0 0 4px #0008;}
      </style><div id="ctlBox"></div>
    """))
    display(ipw.VBox([banner, ipw.HBox([pos_btn, neg_btn, vis_btn])]), target="ctlBox")
    display(widget, plot_output)
    print(f"🆕 Object {obj_id} on frame {frame_idx} — add clicks & press VISUALIZE")

def Visualize(frame_idx: int | None = None):
    """Overlay *all* cached masks on <frame_idx> (defaults to current)."""
    idx = ann_frame_idx if frame_idx is None else frame_idx

    masks = {oid: m for (f, oid), m in mask_cache.items() if f == idx}
    if not masks:
        print("⚠️ No masks found: run VISUALIZE on at least one object first.")
        return

    plt.figure(figsize=(9,6))
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[idx])))
    plt.axis("off")
    for oid, m in masks.items():
        show_mask((m > 0).numpy(), plt.gca(), obj_id=oid)
    plt.show()

def Reset():
    """Clear all annotations and reset the predictor."""
    global inference_state, positive_points, negative_points, mask_cache, prompts
    predictor.reset_state(inference_state)
    positive_points.clear()
    negative_points.clear()
    prompts.clear()
    mask_cache.clear()
    widget.bboxes = []
    plot_output.clear_output(wait=True)
    clear_output(wait=True)
    print("🧹 Workspace reset — ready to annotate a new object.")

def Propagate(vis_frame_stride: int):
    """Propagate your current objects through the entire video."""
    global video_segments
    video_segments = {}
    for f, oids, logits in predictor.propagate_in_video(inference_state):
        video_segments[f] = {oid: (logits[i] > 0).cpu().numpy() for i, oid in enumerate(oids)}

    # Quick preview
    plt.close("all")
    for idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6,4))
        plt.title(f"Frame {idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[idx])))
        plt.axis("off")
        for oid, mask in video_segments.get(idx, {}).items():
            show_mask(mask, plt.gca(), obj_id=oid)
        plt.show()

    print(f"✅ Propagation done – stored masks for {len(video_segments)} frames.")
