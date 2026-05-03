import torch

import config
from src.backbone import get_backbone
from datasets.data import MyData
from src.models import PatchCore


def get_device():
    if config.DEVICE != "auto":
        return config.DEVICE
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    device = get_device()

    train_data = MyData(
        config.CLASS_NAME,
        phase="train",
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        limit=config.TRAIN_LIMIT,
    )
    test_data = MyData(
        config.CLASS_NAME,
        phase="test",
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        limit=config.TEST_LIMIT,
        limit_per_class=config.TEST_LIMIT_PER_CLASS,
    )

    backbone = get_backbone()
    patchcore = PatchCore(backbone, k=config.K, device=device)

    patchcore.fit(train_data)

    img, label = test_data[config.TEST_INDEX]
    score, heatmap = patchcore.predict(img)

    print(f"class: {config.CLASS_NAME}")
    print(f"device: {device}")
    print(f"test label id: {label} ({test_data.get_classes()[label]})")
    print(f"anomaly score: {score.item():.4f}")
    print(f"heatmap shape: {tuple(heatmap.shape)}")


if __name__ == "__main__":
    main()
