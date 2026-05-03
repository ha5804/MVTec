"""
Notebook-friendly experiment config.

Edit this file, then reload it in Jupyter:

    import importlib
    import config.config as cfg
    importlib.reload(cfg)

No argparse is needed. Change DATASET/CATEGORY and rerun the notebook cells.
"""

# "mvtec" or "visa"
DATASET = "mvtec"

# MVTec examples:
# bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut,
# pill, screw, tile, toothbrush, transistor, wood, zipper
#
# VisA examples:
# candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2,
# pcb1, pcb2, pcb3, pcb4, pipe_fryum
CATEGORY = "bottle"

# PatchCore settings
K = 100
BACKBONE = "resnet18"

# Small limits are useful while checking heatmaps in Jupyter.
# Use None when you want the full category.
TRAIN_LIMIT = 50
TEST_LIMIT = 30
TEST_LIMIT_PER_CLASS = 3
TEST_INDEX = 0
BATCH_SIZE = 1

# Image preprocessing
IMAGE_SIZE = 256

# "auto", "mps", "cuda", "cpu"
DEVICE = "auto"

# Reproducible coreset sampling.
SEED = 0
