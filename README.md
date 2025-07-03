# CataractSAM‑2

Point‑click interactive segmentation for cataract‑surgery videos on top of Meta’s **SAM‑2**.

```bash
git clone https://github.com/YOUR‑LAB/cataract-sam2.git && cd cataract-sam2
pip install -e .
python examples/download_checkpoints.py     # downloads SAM‑2 weights
jupyter lab notebooks/00_quick_start.ipynb
