"""Download SAM-2 weights from Hugging Face.

This script fetches the pretrained checkpoint from the public
"DhanvinG/Cataract-SAM2" repository and places it in ``checkpoints/``.
It mirrors the behaviour of the original download script but uses
``huggingface_hub`` instead of a direct HTTP request.
"""

from pathlib import Path
from huggingface_hub import hf_hub_download


dest = Path(__file__).resolve().parents[1] / "checkpoints"
dest.mkdir(parents=True, exist_ok=True)

print("⇣ huggingface")
hf_hub_download(
    repo_id="DhanvinG/Cataract-SAM2",
    filename="Cataract-SAM2.pth",
    local_dir=dest,
)
print("✔ downloaded →", dest / "Cataract-SAM2.pth")
