from torchvision import models


def get_backbone(name="resnet18"):
    if name != "resnet18":
        raise ValueError(f"Unsupported backbone: {name}")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    return model

def get_clip(device):
    import clip

    model, preprocess = clip.load("ViT-B/16", device = device)
    model.eval()
    return model
