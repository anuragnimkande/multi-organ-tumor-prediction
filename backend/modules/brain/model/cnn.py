"""
CNN Feature Extractor — ResNet18 (Pretrained)
=============================================
Accepts single-channel (grayscale) MRI images via weight-averaging of the
pretrained first conv layer. Removes the classification head and adds a
Linear(512 → 128) projector with ReLU + Dropout.

Input shape:  (batch, 1, 224, 224)
Output shape: (batch, 128)

Also includes ClassicalCNNClassifier for benchmarking (no quantum layer).
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CNNFeatureExtractor(nn.Module):
    """
    ResNet18-based feature extractor adapted for grayscale MRI input.

    Key modifications:
    - First conv layer changed from 3→1 channel (weights mean-collapsed)
    - Global average pool output flattened
    - FC head replaced by Linear(512→128) + ReLU + Dropout(0.3)
    """

    def __init__(self, pretrained: bool = True, output_dim: int = 128):
        """
        Args:
            pretrained:  Load ImageNet weights (True for transfer learning)
            output_dim:  Dimension of output feature vector (default 128)
        """
        super().__init__()

        # ── Load pretrained ResNet18 ──────────────────────────────────────
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # ── Adapt first conv for 1-channel grayscale input ────────────────
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        orig_conv = resnet.conv1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False,
        )

        if pretrained:
            # Collapse pretrained 3-channel weights → 1-channel by averaging
            # This preserves learned spatial filters rather than random init
            new_conv.weight.data = orig_conv.weight.data.mean(dim=1, keepdim=True)

        resnet.conv1 = new_conv

        # ── Remove final FC layer — keep everything up to avgpool ─────────
        # children(): conv1, bn1, relu, maxpool, layer1–4, avgpool, fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # Output: (batch, 512, 1, 1)

        # -- Projector: 512 -> output_dim ------------------------------------
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, 224, 224) — normalized grayscale MRI tensor

        Returns:
            features: (batch, output_dim)
        """
        x = self.backbone(x)      # (batch, 512, 1, 1)
        x = self.projector(x)     # (batch, 128)
        return x


class ClassicalCNNClassifier(nn.Module):
    """
    Pure classical CNN classifier — same backbone as HybridModel but with
    a fully classical Dense head. Used for benchmarking against the hybrid.

    Input:  (batch, 1, 224, 224)
    Output: (batch, 1) — sigmoid probability
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.feature_extractor = CNNFeatureExtractor(
            pretrained=pretrained, output_dim=128
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)   # (batch, 128)
        return self.classifier(features)        # (batch, 1)
