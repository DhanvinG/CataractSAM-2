from pathlib import Path
from sam2.build_sam import build_sam2_video_predictor

CFG_PATH = Path(__file__).parent / "cfg" / "sam2_hiera_l.yaml"
CFG = CFG_PATH.resolve().as_uri()

class Predictor:
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
            ckpt_path=str(weights), 
            device=device,
        )

    def init_state(self, **kw):    
        return self.pred.init_state(**kw)

    def reset_state(self, *a, **k):
        return self.pred.reset_state(*a, **k)

    def add_new_points_or_box(self, *a, **k):
        return self.pred.add_new_points_or_box(*a, **k)

    def propagate_in_video(self, *a, **k):
        return self.pred.propagate_in_video(*a, **k)
