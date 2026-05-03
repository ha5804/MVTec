import random

import numpy as np
import torch

from datasets.anomaly import AnomalyDataset
from models.backbone import get_backbone
from models.patchcore import PatchCore


def get_device(device="auto"):
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_datasets(cfg):
    train_data = AnomalyDataset(
        dataset=cfg.DATASET,
        category=cfg.CATEGORY,
        phase="train",
        image_size=cfg.IMAGE_SIZE,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        limit=cfg.TRAIN_LIMIT,
    )
    test_data = AnomalyDataset(
        dataset=cfg.DATASET,
        category=cfg.CATEGORY,
        phase="test",
        image_size=cfg.IMAGE_SIZE,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        limit=cfg.TEST_LIMIT,
        limit_per_class=cfg.TEST_LIMIT_PER_CLASS,
    )
    return train_data, test_data


def build_patchcore(cfg):
    device = get_device(cfg.DEVICE)
    backbone = get_backbone(cfg.BACKBONE)
    return PatchCore(backbone, k=cfg.K, device=device), device


def run_experiment(cfg, show_index=None, compute_auc=True):
    set_seed(cfg.SEED)
    train_data, test_data = build_datasets(cfg)
    model, device = build_patchcore(cfg)
    model.fit(train_data)

    result = {
        "dataset": cfg.DATASET,
        "category": cfg.CATEGORY,
        "device": device,
        "train_size": len(train_data),
        "test_size": len(test_data),
        "test_classes": test_data.get_classes(),
        "model": model,
        "train_data": train_data,
        "test_data": test_data,
    }

    if compute_auc:
        from utils.metrics import get_image_auc

        result["image_auc"] = get_image_auc(test_data.get_loader(), model)

    if show_index is not None:
        from utils.visualization import show_result

        result["shown_score"], result["shown_label"] = show_result(
            test_data,
            model,
            index=show_index,
        )

    return result


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
