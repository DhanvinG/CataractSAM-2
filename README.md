# CataractSAM-2

CataractSAM-2 adapts Meta's Segment Anything Model 2 (SAM-2) for the
precise, point-click segmentation of cataract-surgery videos.  The project
provides a thin wrapper around SAM-2 along with a custom Jupyter widget
that makes annotating medical video frames straightforward.  Masks can then
be propagated throughout an entire sequence and exported for further
analysis.

## Installation

```bash
# clone the repository and install in editable mode
git clone https://github.com/DhanvinG/Cataract-SAM2.git
cd Cataract-SAM2
pip install -e .

# download the pretrained SAM-2 weights (~1.1 GB)
python examples/download_checkpoints.py
```

The environment requires Python 3.10+ and the packages listed in
`requirements.txt`.  The weight download script fetches the public SAM-2
checkpoint from the `DhanvinG/Cataract-SAM2` repository on Hugging Face.
It is stored in `checkpoints/` and is
needed before using the library.

## Quick start

Place your video frames as numbered JPEG files under a directory
(e.g. `data/frames/000.jpg`, `001.jpg`, …).  Then launch a Python session:

```python
from cataractsam2 import Predictor
from cataractsam2.ui_widget import setup, Object

pred = Predictor("checkpoints/sam2_hiera_large.pt")
setup(pred, "data/frames")
Object(0, 1)  # start annotating object 1 on frame 0
```

Click positive/negative points or draw bounding boxes to guide the model.
You can visualise intermediate masks with:

```python
from cataractsam2.ui_widget import Visualize
Visualize()
```

When satisfied with a single frame, propagate your objects through the
sequence:

```python
from cataractsam2.ui_widget import Propagate
Propagate(vis_frame_stride=10)  # show every 10th frame for a quick check
```

Finally export masks for all frames and objects:

```python
from cataractsam2 import Masks
Masks("./masks")  # one PNG per frame/object
```

## Project structure

- `cataractsam2/` – library code wrapping SAM-2 and the widget interface.
- `examples/download_checkpoints.py` – helper script to obtain SAM-2
  weights from Hugging Face.
- `data/` – place your frame sequences here (not included).

## Acknowledgements

CataractSAM-2 builds upon Meta's Segment Anything Model 2.  The code is
licensed under the Apache License 2.0; see the `LICENSE` file for details.
