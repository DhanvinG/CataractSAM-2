from pathlib import Path
from sam2.build_sam import build_sam2_video_predictor

# Default configuration packaged with the library.  Convert the path to a
# ``file://`` URI so Hydra can load it directly without relying on the search
# path plugin.
CFG_PATH = Path(__file__).parent / "cfg" / "sam2_hiera_l.yaml"
CFG = CFG_PATH.resolve().as_uri()

class Predictor:
    """
    Thin wrapper around Meta’s ``build_sam2_video_predictor``.

    Parameters
    ----------
    weights : str | Path
        Path to the model checkpoint.
    config_file : str | Path, optional
        YAML configuration used to construct the predictor. Defaults to the
        bundled ``sam2_hiera_l.yaml``.
    device : str, optional
        Target device for inference.

    Usage
    -----
    >>> pred = Predictor("checkpoints/Cataract-SAM2.pth", device="cuda")
    >>> state = pred.init_state(video_path="data/frames")
    """
    # ----------------------------------------------------
    def __init__(
        self,
        weights: str | Path,
        config_file: str | Path = CFG,
        device: str = "cuda",
    ):
        """Wrap ``build_sam2_video_predictor`` with sane defaults."""
        config_file = str(config_file)
        self.pred = build_sam2_video_predictor(
            config_file=config_file,
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
