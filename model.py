"""
Backbone models with penultimate-layer feature extraction.

The paper's featuriser F' is defined as the penultimate (pre-logit) layer
of a deep neural network.  All distances and confidence measures are computed
over this representation or over the softmax output.

Supported architectures
-----------------------
* ResNet-18 / 34 / 50 / 101 / 152  (He et al., 2016)
* VGG-11 / 13 / 16 / 19            (Simonyan & Zisserman, 2014)
* DenseNet-121 / 169 / 201         (Huang et al., 2017)
* ViT-B/16                          (Dosovitskiy et al., 2021) – optional
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as tv_models


# ── Supported backbone names ─────────────────────────────────────────────────

BACKBONE_REGISTRY = {
    # ResNets
    "resnet18":   (tv_models.resnet18,   512),
    "resnet34":   (tv_models.resnet34,   512),
    "resnet50":   (tv_models.resnet50,  2048),
    "resnet101":  (tv_models.resnet101, 2048),
    "resnet152":  (tv_models.resnet152, 2048),
    # VGGs
    "vgg11":      (tv_models.vgg11_bn,  4096),
    "vgg13":      (tv_models.vgg13_bn,  4096),
    "vgg16":      (tv_models.vgg16_bn,  4096),
    "vgg19":      (tv_models.vgg19_bn,  4096),
    # DenseNets
    "densenet121": (tv_models.densenet121, 1024),
    "densenet169": (tv_models.densenet169, 1664),
    "densenet201": (tv_models.densenet201, 1920),
}


# ── Core model wrapper ───────────────────────────────────────────────────────

class FeatureExtractorModel(nn.Module):
    """
    A backbone classifier that also exposes penultimate features.

    Forward pass returns ``(logits, features)`` where:
    * ``logits``   – raw (un-normalised) class scores, shape (N, C)
    * ``features`` – penultimate-layer activations, shape (N, D)

    Parameters
    ----------
    backbone_name : architecture key from ``BACKBONE_REGISTRY``
    num_classes   : number of target classes
    pretrained    : initialise weights from ImageNet pre-training
    dropout_rate  : dropout applied before the final linear layer
    freeze_bn     : if True, batch-norm statistics are frozen (useful for
                    very small target batches or few-shot scenarios)
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_classes: int   = 7,
        pretrained: bool   = True,
        dropout_rate: float = 0.0,
        freeze_bn: bool    = False,
    ) -> None:
        super().__init__()

        if backbone_name not in BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Choose from: {list(BACKBONE_REGISTRY.keys())}"
            )

        factory, feat_dim  = BACKBONE_REGISTRY[backbone_name]
        self.feat_dim      = feat_dim
        self.num_classes   = num_classes
        self.backbone_name = backbone_name

        # ── Build backbone ─────────────────────────────────────────────────
        weights = "IMAGENET1K_V1" if pretrained else None
        base    = factory(weights=weights)

        if "resnet" in backbone_name:
            self.features = nn.Sequential(*list(base.children())[:-1])   # up to avgpool
            self.dropout  = nn.Dropout(p=dropout_rate)
            self.classifier = nn.Linear(feat_dim, num_classes)

        elif "vgg" in backbone_name:
            self.features = base.features
            self.avgpool  = base.avgpool
            # Replace the original classifier's final layer
            classifier_layers = list(base.classifier.children())[:-1]
            self.intermediate = nn.Sequential(*classifier_layers)
            self.dropout      = nn.Dropout(p=dropout_rate)
            self.classifier   = nn.Linear(4096, num_classes)

        elif "densenet" in backbone_name:
            self.features   = base.features
            self.pool       = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout    = nn.Dropout(p=dropout_rate)
            self.classifier = nn.Linear(feat_dim, num_classes)

        else:
            raise ValueError(f"Architecture family not handled: {backbone_name}")

        if freeze_bn:
            self._freeze_bn()

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self._extract_features(x)                     # (N, D)
        feats_drop = self.dropout(feats)
        logits = self.classifier(feats_drop)                  # (N, C)
        return logits, feats

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if "resnet" in self.backbone_name:
            h = self.features(x)                              # (N, D, 1, 1)
            return h.flatten(1)                               # (N, D)

        elif "vgg" in self.backbone_name:
            h = self.features(x)
            h = self.avgpool(h)
            h = h.flatten(1)
            return self.intermediate(h)

        elif "densenet" in self.backbone_name:
            h = self.features(x)
            h = torch.relu(h)
            h = self.pool(h)
            return h.flatten(1)

        raise RuntimeError("Unhandled backbone family")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _freeze_bn(self):
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def predict_proba(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (softmax_probs, features).
        ``temperature`` can be set > 1 to soften, < 1 to sharpen predictions.
        """
        with torch.no_grad():
            logits, feats = self.forward(x)
        probs = torch.softmax(logits / temperature, dim=-1)
        return probs, feats

    def feature_dim(self) -> int:
        return self.feat_dim


# ── Convenience factory ───────────────────────────────────────────────────────

def build_model(
    backbone: str      = "resnet50",
    num_classes: int   = 7,
    pretrained: bool   = True,
    dropout: float     = 0.1,
    device: Optional[str] = None,
) -> FeatureExtractorModel:
    """
    Construct and return a ``FeatureExtractorModel`` on the requested device.

    Example
    -------
    >>> model = build_model("resnet50", num_classes=7, pretrained=True)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FeatureExtractorModel(
        backbone_name=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout,
    )
    return model.to(device)


# ── Temperature-scaled wrapper ────────────────────────────────────────────────

class TemperatureScaledModel(nn.Module):
    """
    Post-hoc temperature scaling (Guo et al., 2017).

    Wraps any ``FeatureExtractorModel`` and learns a single scalar temperature
    T to improve calibration on a held-out validation set.

    Usage::

        ts_model = TemperatureScaledModel(model)
        ts_model.calibrate(val_loader, device)
        probs = ts_model.predict_proba(x)
    """

    def __init__(self, model: FeatureExtractorModel) -> None:
        super().__init__()
        self.model       = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, feats = self.model(x)
        return logits / self.temperature, feats

    def predict_proba(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits, feats = self.forward(x)
        return torch.softmax(logits, dim=-1), feats

    @torch.no_grad()
    def calibrate(self, val_loader, device: str = "cpu", lr: float = 0.01,
                  max_iter: int = 50) -> float:
        """
        Optimise T by minimising NLL on ``val_loader``.
        Returns the optimal temperature value.
        """
        self.model.eval()
        all_logits, all_labels = [], []
        for batch in val_loader:
            imgs, labels = batch[0].to(device), batch[1].to(device)
            logits, _ = self.model(imgs)
            all_logits.append(logits.detach())
            all_labels.append(labels)
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        nll = nn.CrossEntropyLoss()

        def eval_fn():
            optimizer.zero_grad()
            loss = nll(all_logits / self.temperature, all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_fn)
        return self.temperature.item()
