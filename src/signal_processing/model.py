import torch.nn as nn
from torchvision import models

class SignalCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    def forward(self, x): return self.classifier(self.features(x))

def get_signal_model(name="signalcnn", num_classes=10, mode="finetune", pretrained=True):
    if name == "signalcnn":
        model = SignalCNN(num_classes)
        print(f"[SignalModel] SignalCNN | Params: {sum(p.numel() for p in model.parameters()):,}")
        return model
    elif name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model   = models.resnet18(weights=weights)
        model.conv1   = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        if mode == "feature_extract":
            for p in model.parameters(): p.requires_grad = False
        model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, num_classes))
        print(f"[SignalModel] ResNet-18 | Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        return model
    raise ValueError(f"Unknown model: {name}")