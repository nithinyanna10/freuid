from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_model(backbone: str = "resnet18", pretrained: bool = True) -> nn.Module:
    weights = "DEFAULT" if pretrained else None
    if backbone == "resnet18":
        net = models.resnet18(weights=weights)
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, 1)
        return net
    if backbone == "efficientnet_b0":
        net = models.efficientnet_b0(weights=weights)
        in_features = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_features, 1)
        return net
    raise ValueError(f"Unknown backbone: {backbone}")
