# CataractSAM-2

CataractSAM-2 provides an interactive interface for video object segmentation built on
Meta's **SAM-2** model. The library exposes a small API for point-based annotation and
mask propagation tailored to cataract surgery footage.

## Installation

Clone the repository and install the package in editable mode. Then download the SAM-2
checkpoint (approx. 3GB) using the helper script. A CUDA-capable GPU with PyTorch is
required for real-time performance.

```bash
git clone https://github.com/DhanvinG/Cataract-SAM2.git && cd Cataract-SAM2
pip install -e .
python examples/download_checkpoints.py
```

## Quick start

Prepare a directory of JPEG frames extracted from a video. The following example shows
how to launch the annotation widget and begin marking objects.

```python
from cataractsam2 import Predictor
from cataractsam2.ui_widget import setup, Object, Propagate, Reset

pred = Predictor("checkpoints/sam2_hiera_large.pt")
setup(pred, "data/frames")  # directory containing frame_000.jpg, frame_001.jpg, ...
Object(0, 1)  # start annotating object id 1 on the first frame
```

After adding points press **VISUALIZE** to create a mask. Repeat for additional frames
or objects. When satisfied, propagate all objects through the video and save the masks:

```python
Propagate(vis_frame_stride=5)      # generates masks for the whole clip
from cataractsam2 import Masks
Masks("./masks")                   # writes frame_000_obj_001.png etc.
```

## Notes

- The model weights and this code are provided for research use. They are not intended
  for clinical decision making.
- The example widget is designed for Jupyter environments and assumes reasonably sized
  input frames (e.g. 1280×720).
- `cataractsam2/cfg/sam2_hiera_l.yaml` configures the network architecture; advanced
  users may experiment with these settings.

For questions or contributions please open an issue or pull request.
