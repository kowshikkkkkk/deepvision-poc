import torch.nn as nn
from torchvision import models
from typing import Literal

def build_resnet18(num_classes=10, mode="finetune", pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model   = models.resnet18(weights=weights)
    model.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    if mode == "feature_extract":
        for p in model.parameters(): p.requires_grad = False
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, num_classes))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] ResNet-18 | Trainable: {trainable:,} | Mode: {mode}")
    return model

def build_vgg16(num_classes=10, mode="finetune", pretrained=True):
    weights = models.VGG16_Weights.DEFAULT if pretrained else None
    model   = models.vgg16(weights=weights)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    if mode == "feature_extract":
        for p in model.features.parameters(): p.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes)
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] VGG-16 | Trainable: {trainable:,} | Mode: {mode}")
    return model

def get_model(name, num_classes=10, mode="finetune", pretrained=True):
    if name == "resnet18": return build_resnet18(num_classes, mode, pretrained)
    if name == "vgg16":    return build_vgg16(num_classes, mode, pretrained)
    raise ValueError(f"Unknown model: {name}")