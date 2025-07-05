from pathlib import Path
from sam2.build_sam import build_sam2_video_predictor

CFG   = Path(__file__).with_suffix("").parent / "cfg" / "sam2_hiera_l.yaml"

class Predictor:
    """
    Thin wrapper around Meta’s build_sam2_video_predictor.
    Usage
    -----
    >>> pred = Predictor("checkpoints/Cataract-SAM2.pth", device="cuda")
    >>> state = pred.init_state(video_path="data/frames")
    """
    # ----------------------------------------------------
    def __init__(self, weights: str | Path, device: str = "cuda"):
        # Upstream ``build_sam2_video_predictor`` now expects ``config_file``
        # instead of ``model_cfg``. Match the new signature so the wrapper
        # remains compatible with the latest ``sam-2`` package.
        self.pred = build_sam2_video_predictor(
            config_file=str(CFG),
            ckpt=str(weights),
            device=device,
        )

    # ----  surface only the methods the widget / scripts need  ----
    def init_state(self, **kw):                     # pylint: disable=missing-docstring
        return self.pred.init_state(**kw)

    def reset_state(self, *a, **k):
        return self.pred.reset_state(*a, **k)

    def add_new_points_or_box(self, *a, **k):
        return self.pred.add_new_points_or_box(*a, **k)

    def propagate_in_video(self, *a, **k):
        return self.pred.propagate_in_video(*a, **k)
