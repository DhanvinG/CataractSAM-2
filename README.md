# CataractSAM-2: Enhancing Transferability and Real-Time Ophthalmic Surgery Segmentation Through Automated Ground-Truth Generation

We introduce CataractSAM‑2, a domain-adapted extension of [SAM‑2](https://github.com/facebookresearch/sam2) optimized for high-precision segmentation in cataract and related ophthalmic surgeries. To preserve generalizable visual priors, we freeze the SAM‑2 image encoder and fine-tune only the prompt encoder and mask decoder on the [Cataract‑1K dataset](https://github.com/Negin-Ghamsarian/Cataract-1K). To address the time-consuming nature of manual frame-by-frame annotation, we develop a human-in-the-loop interactive annotation framework built on the `SAM2VideoPredictor`, significantly accelerating ground-truth generation.

<img width="1920" height="1440" alt="Image" src="https://github.com/user-attachments/assets/1812e863-dee8-4d38-b7f6-faf46663687d" />


## Overview

- **CataractSAM‑2 Model**  
  A fine-tuned, domain-adapted variant of Meta’s SAM‑2, trained specifically for ophthalmic surgery segmentation. It achieves **90–95% mean IoU** and runs in real time at **15 FPS** across surgical videos.

- **Interactive Ground-Truth Annotation Framework**  
  A lightweight, point-guided annotation system leveraging the `SAM2VideoPredictor`. Users provide sparse point-based prompts, and the model propagates accurate masks through the video, cutting annotation time by **over 80%**.

- **Open-Source Toolkit**  
  This repo includes:
  - ✅ Pretrained weights (`.pth`)
  - ✅ Interactive inference widgets
  - ✅ Demo notebook

## Tutorial

https://github.com/user-attachments/assets/daf076f8-af5a-44ce-a793-1e422428ef33

## Load from 🤗 Hugging Face

We released our pretrain weight [here](https://huggingface.co/DhanvinG/Cataract-SAM2/tree/main)


## Installation

Draft-Will be updated:

-create and activate environment
-install SAM2
-install Cataract-SAM2
-install jupyter notebook
-download train model

We tried the python 3.12, SAM-2 1.0, Jupuyterlab 7.4.4 and cuda 12.2

------------------------------

This project ships Meta's original SAM-2 repository as a git submodule under `sam2/`. Installing it in editable mode enables the exact CLI exposed by the upstream code.


```bash
# clone the repository and install in editable mode
git clone --recurse-submodules https://github.com/DhanvinG/Cataract-SAM2.git
cd Cataract-SAM2
git submodule update --init --recursive
pip install -e ./segment_anything_2
pip install -e .
```
> [!WARNING]
> Restart your Python session or runtime to ensure imports work.
> This is required for Hydra and editable installs to be registered correctly


```
# download the pretrained SAM-2 weights from Hugging Face
python examples/download_checkpoints.py
```

The environment requires Python 3.10+ and the packages listed in
`requirements.txt`.  The weight download script fetches the public SAM-2
checkpoint from the `DhanvinG/Cataract-SAM2` repository on Hugging Face.
It is stored in `checkpoints/` and is
needed before using the library.

## Quick start

Place your video frames as numbered JPEG files under the `data` directory
(e.g. `data/frames/000.jpg`, `001.jpg`, …). Then build the predictor directly:

```python
from sam2.build_sam import build_sam2_video_predictor
pred = build_sam2_video_predictor(model_cfg, "checkpoints/Cataract-SAM2.pth", device="cuda")
setup(pred, "data")
Object(0, 1)  # start annotating object 1 on frame 0
```

Click positive/negative points to guide the model segmentation. 

You can visualize intermediate masks by pressing the `VISUALIZE` button in the notebook UI.


```python
from cataractsam2.ui_widget import Visualize
Visualize()
```

When satisfied with a single frame, propagate your objects through the
sequence:

```python
from cataractsam2.ui_widget import Propagate
Propagate(10)  #e.g. show every 10th frame for a quick check
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
- `data/` – place your frame sequences here (example frames included).
- `notebooks/` – Contains an end-to-end demo notebook for using CataractSAM2 on video frames.

CataractSAM-2 builds upon Meta's Segment Anything Model 2.  The code is
licensed under the Apache License 2.0; see the `LICENSE` file for details.
