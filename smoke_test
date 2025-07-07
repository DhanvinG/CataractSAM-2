import os
from pathlib import Path

# 1) Check your bundled YAML
from cataractsam2.predictor import CFG_PATH
print("Wrapper YAML exists:", os.path.exists(CFG_PATH))

# 2) Try your Predictor
from cataractsam2 import Predictor
pred = Predictor("checkpoints/Cataract-SAM2.pth")
print("✅ cataractsam2.Predictor loaded")

# 3) Try the Colab‑style CLI
from sam2.build_sam import build_sam2_video_predictor
pred2 = build_sam2_video_predictor(
    config_file="sam2_hiera_l.yaml",
    ckpt="checkpoints/Cataract-SAM2.pth",
    device="cpu"
)
print("✅ sam2.build_sam2_video_predictor loaded")
