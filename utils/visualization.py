import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def denormalize(img):
    """
    정규화된 이미지를 원래 픽셀 값으로 복원
    args:
        img(Tensor) : (C, H, W)
    returns:
        Tensor: (C, H, W) 형태 복원 이미지 (0~1)
    """
    img = img.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return img.clamp(0, 1)


def show_result(dataset, model, index=0):
    img, label = dataset[index]
    score, heatmap = model.predict(img)

    image = denormalize(img).permute(1, 2, 0).numpy()

    heatmap = heatmap.detach().cpu()
    heatmap = F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0),
        size=image.shape[:2],
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_np = heatmap.numpy()
    label_name = dataset.get_label_name(label)
    image_path = dataset.get_path(index) if hasattr(dataset, "get_path") else None

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    title = f"image: {label_name}"
    if image_path is not None:
        title += f"\n{image_path.name}"
    axes[0].set_title(title)
    axes[0].axis("off")

    axes[1].imshow(heatmap_np, cmap="jet")
    axes[1].set_title(f"heatmap\nscore={score.item():.4f}")
    axes[1].axis("off")

    axes[2].imshow(image)
    axes[2].imshow(heatmap_np, cmap="jet", alpha=0.45)
    axes[2].set_title("overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    return score.item(), label_name
