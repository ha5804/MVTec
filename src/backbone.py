from torchvision import models
import clip
import torch

def get_backbone():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    return model

def get_clip(device):
    model, preprocess = clip.load("ViT-B/16", device = device)
    model.eval()
    return model
