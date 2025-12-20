# B2F: End-to-End Body-to-Face Motion Generation with Style Reference

Official implementation of [B2F: End-to-End Body-to-Face Motion Generation with Style Reference](https://diglib.eg.org/items/93f81df9-1fa7-4ce4-b1f9-916c145d81e6). Generates facial motions from full-body motions (Motion-X). `CharacterAnimationTools` is vendored (lightly modified). Blender visualization scripts are kept local (ignored via `.gitignore`) and are not part of this public guide.

## Repository Layout
- `inference/`: Runtime controller, Blender viewer scripts, and helper utilities.
- `scripts/training/`: Entry point to train the main Model_Ours network.
- `scripts/dataset/`: Dataset conversion scripts (NPZ/BVH helpers, dataset creation).
- `training/`: Model architectures and training logic.
- `process_dataset/`: Dataset constants and feature utilities.
- `CharacterAnimationTools/`: External library (cloned and locally tweaked for this project).
- `Model/`: Place pretrained or newly trained checkpoints here (e.g., `Model/Model_Ours/model_299epoch.pth`).

## Requirements
- OS: Linux/macOS (tested locally on Linux).
- Python: 3.10+ with `pip`.
- Blender: 3.6+ (uses Blender’s Python to run the viewer scripts).
- GPU: CUDA-capable GPU recommended for training/inference.

### Requirements / Setup (Python 3.10+)
`CharacterAnimationTools` requires Python 3.10+ (uses `match`).
```bash
python3.10 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install --upgrade pip
# GPU build (change cu118 if needed) or CPU build without index-url
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
# If torch libs fail to load in conda:
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
```

#### Option B: System Python
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Blender ships its own Python; to add deps there, run Blender’s bundled Python:
```bash
/path/to/blender/python/bin/python -m ensurepip
/path/to/blender/python/bin/pip install -r requirements.txt
```
`bpy`/`mathutils` come with Blender; do not install them via pip.

## Data
This project expects the Motion-X dataset. The dataset is **not** included in this repo.
- Obtain Motion-X separately (respecting its license) and point `PATH_DB_ORIGIN` (and related paths) in `process_dataset/Constant.py` to your local dataset location. Replace any hard-coded sample paths in the Blender scripts with your own paths.
- `PATH_MODEL` should point to the directory containing your checkpoints (`Model/` by default).

## Setup
1) Clone this repository (including the included `CharacterAnimationTools` folder with local tweaks).
2) Adjust paths in `process_dataset/Constant.py`:
   - `PATH_DB_ORIGIN`, `PATH_DB`, `PATH_MODEL`, and any absolute paths in the Blender scripts if your environment differs.
3) Ensure your pretrained checkpoint exists, e.g. `Model/Model_Ours/model_299epoch.pth`.

## Training
Train the main “ours” model (arguments required):
```bash
python scripts/training/train_ours.py \
  --dataset_dir /path/to/processed_dataset_dir \  # required
  --save_dir Model/Model_Ours \                   # required
  --epochs 300 \                                 # optional
  --batch_size 64 \                              # optional
  --save_step 50 \                               # optional
  --device cuda                                  # optional
```
The entry script wraps `training/createTrainedNetwork.py`, which uses Motion-X (`*_clipping_random_big`) and saves checkpoints under `Model/Model_Ours/`.

## Inference
### CLI (no Blender)
Option 1: processed dataset (pickle) + trained checkpoint
```bash
python scripts/inference/generate_facial_motion.py \
  --dataset_dir /path/to/processed_dataset_dir \
  --model Model/Model_Ours/model_299epoch.pth \
  --index 0 \
  --output outputs/facial_motion.npz
```
This loads one sample, runs Model_Ours, and saves `jaw` and `expression` arrays to the given npz.

Option 2: raw Motion-X files (no processed dataset)
```bash
python scripts/inference/generate_facial_motion.py \
  --body_npy /home/jbok6825/dataset_MotionX/perform/subset_0002/Throw_Stone.npy \
  --style_npy /home/jbok6825/dataset_MotionX/dance/subset_0000/A_Han_And_Tang_Dance_That_You_Will_Never_Get_Tired_Of_clip_1.npy \
  --model Model/Model_Ours/model_299epoch.pth \
  --output outputs/facial_motion.npz
# or use --body_bvh for a retargeted BVH with the required joints
```

### Blender viewers (local only)
Blender visualization/retargeting scripts are not shipped in the public repo and are ignored by `.gitignore`. If you maintain local copies under `inference/`, adjust any hard-coded paths to your dataset/checkpoints before running them inside Blender’s Python.

## CharacterAnimationTools
This repository contains a copied and locally modified version of [CharacterAnimationTools](https://github.com/KosukeFukazawa/CharacterAnimationTools.git). Keep this folder when pushing/sharing, since the Blender viewers rely on its BVH utilities.

## Typical Run Checklist
1) Verify `process_dataset/Constant.py` paths (dataset + model).
2) Place trained weights in `Model/Model_Ours/`.
3) Build processed dataset (if needed):
```bash
python scripts/dataset/build_dataset.py \
  --data_root /path/to/MotionX_npy \
  --style_root /path/to/MotionX_style_txts \
  --bvh_root /path/to/MotionX_bvh \
  --output_dir /path/to/output_dataset    # optional; defaults to <repo>/dataset_clipping_random_big
```
4) Train/retrain: `python scripts/training/train_ours.py ...`.
5) CLI inference: `python scripts/inference/generate_facial_motion.py ...`.
6) Blender (optional): run the viewer scripts if you need visualization/retargeting.

### Quick local inference (scriptable)
If you want a minimal, non-Blender smoke test from Python after setting paths and weights:
```bash
python - <<'PY'
from inference.SystemController import SystemController
from process_dataset.Constant import PATH_MODEL
import numpy as np, torch
import process_dataset.utils as utils

# dummy example: load motion/pose arrays as needed
# global_position, global_orientation, global_velocity, character_local_coordinate = ...
# style_motion_path = "/path/to/style.npy"

# sc = SystemController(
#     torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
#     path_model=f\"{PATH_MODEL}/Model_Ours/model_299epoch.pth\",
#     path_style_motion=style_motion_path,
#     global_position=global_position,
#     global_orientation=global_orientation,
#     global_velocity=global_velocity,
#     character_local_coordinate=character_local_coordinate,
# )
# jaw, expr = sc.create_total_facial_motion()
# print(jaw.shape, expr.shape)
PY
```

## Support / Notes
- All scripts assume the Motion-X directory structure; missing files will raise errors on load.
- If you need to install Python deps into Blender’s Python, run Blender’s bundled `python -m ensurepip` and then `pip install` as needed.

## Citation
If you use this work, please cite:
```
@article{https://doi.org/10.2312/pg.20251256,
  doi = {10.2312/PG.20251256},
  url = {https://diglib.eg.org/handle/10.2312/pg20251256},
  author = {Jang, Bokyung and Jung, Eunho and Lee, Yoonsang},
  title = {B2F: End-to-End Body-to-Face Motion Generation with Style Reference},
  journal = {Pacific Graphics Conference Papers, Posters, and Demos},
  publisher = {The Eurographics Association},
  year = {2025},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
