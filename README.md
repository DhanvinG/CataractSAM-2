# CataractSAM-2 Enhancing Transferability and Real-Time Ophthalmic Surgery Segmentation Through Automated Ground-Truth Generation

We introduce CataractSAM‑2, a domain-adapted extension of [SAM‑2](https://github.com/facebookresearch/sam2) optimized for high-precision segmentation in cataract and related ophthalmic surgeries. To preserve generalizable visual priors, we freeze the SAM‑2 image encoder and fine-tune only the prompt encoder and mask decoder on the [Cataract‑1K dataset](https://github.com/Negin-Ghamsarian/Cataract-1K). To address the time-consuming nature of manual frame-by-frame annotation, we develop a human-in-the-loop interactive annotation framework built on the `SAM2VideoPredictor`, significantly accelerating ground-truth generation.

<img width="960" height="720" alt="Image" src="https://github.com/user-attachments/assets/12f5fbfa-b462-4ffc-acda-532f003bb25f" />

## Overview

- **CataractSAM‑2 Model**  
  A fine-tuned, domain-adapted variant of Meta’s SAM‑2, trained specifically for ophthalmic surgery segmentation. It achieves **90–96% mean IoU** and runs in real time at **15 FPS** across surgical videos.

- **Interactive Ground-Truth Annotation Framework**  
  A lightweight, point-guided annotation system leveraging the `SAM2VideoPredictor`. Users provide sparse point-based prompts, and the model propagates accurate masks through the video, cutting annotation time by **over 80%**.

- **Open-Source Toolkit**  
  This repo includes:
  - ✅ Pretrained weights (`.pth`)
  - ✅ Interactive inference widgets and scripts
  - ✅ Training notebooks and fine-tuning code  

## Load from 🤗 Hugging Face

We released our pretrain weight [here](https://huggingface.co/DhanvinG/Cataract-SAM2/tree/main)


## Installation

```bash
# clone the repository and install in editable mode
git clone --recurse-submodules https://github.com/DhanvinG/Cataract-SAM2.git
cd Cataract-SAM2
git submodule update --init --recursive  # fetch Meta's SAM-2 code
pip install -e ./segment_anything_2
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
from cataractsam2 import Predictor, setup, Object

# the YAML configuration is bundled with the package
pred = Predictor("checkpoints/Cataract-SAM2.pth")
setup(pred, "data/frames")
Object(0, 1)  # start annotating object 1 on frame 0
```

`Predictor` automatically loads the bundled configuration
`cataractsam2/cfg/sam2_hiera_l.yaml`.  If you need to override this file,
provide `config_file=PATH` when instantiating the class.

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

**Workflow summary:** clone the repository with submodules, install `sam2`
and `cataractsam2` in editable mode, download `Cataract-SAM2.pth` with the
helper script, then open a Python session and
create a `Predictor` with your checkpoint.  Initialise the widget on your
frame directory, refine masks interactively and propagate them through the
video, finally exporting the results with `Masks("./masks")`.

### Using the vendored SAM-2 (Colab way)

This project ships Meta's original SAM-2 repository as a git submodule
under `sam2/`.  Installing it in editable mode enables the exact CLI
exposed by the upstream code.

```bash
git submodule update --init --recursive
pip install -e ./segment_anything_2
```

Then build the predictor directly:

```python
from sam2.build_sam import build_sam2_video_predictor
pred = build_sam2_video_predictor(model_cfg, "checkpoints/Cataract-SAM2.pth", device="cuda")
```

`sam2` already adds its `configs/` directory to Hydra's search path, so the
configuration file can be referenced without extra setup.  This mirrors the
workflow typically used in Google Colab notebooks.

## Project structure

- `cataractsam2/` – library code wrapping SAM-2 and the widget interface.
- `examples/download_checkpoints.py` – helper script to obtain SAM-2
  weights from Hugging Face.
- `data/` – place your frame sequences here (not included).

## Hydra configuration

```ini
[options.entry_points]
hydra.searchpath =
## Acknowledgements

CataractSAM-2 builds upon Meta's Segment Anything Model 2.  The code is
licensed under the Apache License 2.0; see the `LICENSE` file for details.
