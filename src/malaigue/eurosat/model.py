"""The two models: a from-scratch SmallCNN and a frozen ResNet18 backbone.

`SmallCNN` is the centerpiece: a compact convolutional network trained from
scratch on EuroSAT's native 64x64 RGB. Four conv blocks, batch-norm, max-pool,
global average pool, dropout, one linear classifier. About 0.39M parameters,
sized for a CPU epoch budget.

`pretrained_backbone` is the transfer baseline's feature extractor: ResNet18
with ImageNet weights, fully frozen. It is used to extract features once; a
linear probe is then trained on top. This is a baseline for comparison, not
"training a CNN".
"""
import torch.nn as nn


def _conv_block(in_ch, out_ch):
    """conv 3x3 (pad 1) -> batch-norm -> ReLU -> 2x2 max-pool (halves H, W)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """From-scratch CNN for 64x64 RGB EuroSAT chips.

    64 -> 32 -> 16 -> 8 -> 4 over four blocks, then global average pool to a
    256-d vector, dropout, and a linear classifier. Batch-norm and dropout are
    the overfitting controls.
    """

    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),     # 64 -> 32
            _conv_block(32, 64),    # 32 -> 16
            _conv_block(64, 128),   # 16 -> 8
            _conv_block(128, 256),  # 8 -> 4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


def pretrained_backbone():
    """Frozen ResNet18 (ImageNet) as a feature extractor.

    Returns `(backbone, feature_dim, preprocess)`. The final fc layer is
    replaced by identity so the backbone outputs 512-d features; all parameters
    are frozen and the module is in eval mode. `preprocess` is the standard
    ImageNet transform (resize/crop to 224, ImageNet normalization) that the
    backbone expects.
    """
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1
    net = resnet18(weights=weights)
    net.fc = nn.Identity()
    for p in net.parameters():
        p.requires_grad = False
    net.eval()
    return net, 512, weights.transforms()
