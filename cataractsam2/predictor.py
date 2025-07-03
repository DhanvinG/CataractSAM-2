from pathlib import Path
from sam2.build_sam import build_sam2_video_predictor   # comes from FB repo

CFG_PATH = Path(__file__).with_suffix("").parent / "cfg" / "sam2_hiera_l.yaml"

class Predictor:
    """Tiny wrapper around Meta’s SAM‑2 video predictor."""
    def __init__(self, weights: str | Path, device: str = "cuda"):
        self._pred = build_sam2_video_predictor(
            model_cfg=str(CFG_PATH),
            ckpt=str(weights),
            device=device,
        )

    # expose only what our UI needs
    def init_state            (self, **k):   return self._pred.init_state(**k)
    def reset_state           (self, *a, **k): return self._pred.reset_state(*a, **k)
    def add_new_points_or_box (self, *a, **k): return self._pred.add_new_points_or_box(*a, **k)
    def propagate_in_video    (self, *a, **k): return self._pred.propagate_in_video(*a, **k)
