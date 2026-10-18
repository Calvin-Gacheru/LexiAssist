
import torch, torch.nn as nn
from torchvision import models

class SmallCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 128))
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                        nn.Dropout(0.4), nn.Linear(128, num_classes))
    def forward(self, x):
        return self.classifier(self.features(x))

def build_model(arch="resnet18", num_classes=2, pretrained=False):
    if arch == "resnet18":
        w = models.ResNet18_Weights.DEFAULT if pretrained else None
        m = models.resnet18(weights=w)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
        return m
    if arch == "smallcnn":
        return SmallCNN(num_classes)
    raise ValueError(arch)

def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["arch"], num_classes=len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta
