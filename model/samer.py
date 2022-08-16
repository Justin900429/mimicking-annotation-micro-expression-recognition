import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from .resnet import resnet18


class BaseLayer(nn.Module):
    def __init__(self, pretrained=True):
        super(BaseLayer, self).__init__()

        model = resnet18(pretrained=pretrained)
        self.layers = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            nn.Dropout(0.2),
            model.layer2,
            nn.Dropout(0.2),
        )
        del model

    def forward(self, x):
        return self.layers(x)


class BranchLayer(nn.Module):
    def __init__(self, pretrained=True):
        super(BranchLayer, self).__init__()

        model = resnet18(pretrained=pretrained)
        self.layers = nn.Sequential(
            model.layer3,
            nn.Dropout(0.2),
            model.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
        )
        del model

    def forward(self, x):
        return F.normalize(self.layers(x), dim=1)


class SAMER(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(SAMER, self).__init__()

        # Base layer for all the area
        self.base_layers = BaseLayer(pretrained=pretrained)

        # Separate the area with different branches
        self.eyes_branch = BranchLayer(pretrained=pretrained)
        self.mouth_branch = BranchLayer(pretrained=pretrained)

        self.area_weight = nn.Sequential(
            nn.Linear(512 * 2, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.Sigmoid(),
        )

        self.eyes = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        self.mouth = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def freeze_layers(self):
        self.base_layers.requires_grad_(False)
        self.eyes_branch.requires_grad_(False)
        self.mouth_branch.requires_grad_(False)

    def get_features(self, eyes, mouth):
        eyes_features = self.eyes_branch(self.base_layers(eyes))
        mouth_features = self.mouth_branch(self.base_layers(mouth))
        return eyes_features, mouth_features

    def fusing(self, eyes_features, mouth_features):
        combine_features = torch.cat([eyes_features, mouth_features], dim=-1)
        area_weight = self.area_weight(combine_features.detach())
        combine_features = combine_features * area_weight

        eyes_features = combine_features[:, :512]
        mouth_features = combine_features[:, 512:]
        combine_features = F.normalize(combine_features, dim=-1)

        if self.training:
            combine_features = F.dropout(combine_features, 0.1)

        return eyes_features, mouth_features, combine_features

    def forward(self, eyes, mouth):
        eyes_features, mouth_features = self.get_features(eyes, mouth)
        eyes_features, mouth_features, combine_features = self.fusing(
            eyes_features, mouth_features
        )
        eyes_features = self.eyes(eyes_features)
        mouth_features = self.mouth(mouth_features)
        return eyes_features, mouth_features, combine_features

    def predict(self, eyes, mouth, get_weight=False):
        if self.training:
            warnings.warn(
                "Evaluation is detected but still in training mode", RuntimeWarning
            )

        eyes_features, mouth_features = self.get_features(eyes, mouth)
        eyes_features, mouth_features, combine_features = self.fusing(
            eyes_features, mouth_features
        )
        eyes_features = self.eyes(eyes_features)
        mouth_features = self.mouth(mouth_features)

        if get_weight:
            return eyes_features + mouth_features, combine_features
        else:
            return eyes_features + mouth_features
