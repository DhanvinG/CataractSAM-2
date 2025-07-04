# CataractSAM‑2

Point‑click interactive segmentation for cataract‑surgery videos on top of Meta’s **SAM‑2**.

```bash
git clone https://github.com/DhanvinG/Cataract-SAM2.git && cd Cataract-SAM2
pip install -e .
python examples/download_checkpoints.py     # downloads SAM‑2 weights
```

Start annotating in Python:

```python
from cataractsam2 import Predictor
from cataractsam2.ui_widget import setup, Object

pred = Predictor("checkpoints/sam2_hiera_large.pt")
setup(pred, "data/frames")
Object(0, 1)
```

After propagating masks you can export them as PNG files:

```python
from cataractsam2 import Masks
Masks("./masks")  # writes one PNG per frame/object
```
