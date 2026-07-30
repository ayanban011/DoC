"""
Data transforms for domain-generalisation experiments.

All images are resized to 224 × 224 (ImageNet convention) so that
pre-trained ImageNet backbones can be used without modification.
"""

from torchvision import transforms

# ── ImageNet normalisation constants ────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def get_train_transform(img_size: int = 224) -> transforms.Compose:
    """
    Standard ImageNet-style train transform with random crop + horizontal flip.
    Identical to the augmentation used in most DG benchmarks (DomainBed, etc.).
    """
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),   # slight over-size
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def get_eval_transform(img_size: int = 224) -> transforms.Compose:
    """
    Deterministic centre-crop transform used during evaluation and feature
    extraction.  No data augmentation is applied.
    """
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def get_corruption_transforms(img_size: int = 224, severity: int = 3):
    """
    Lightweight synthetic corruptions used to generate calibration shifts
    without requiring the full ImageNet-C pipeline.

    Returns a *dict* mapping corruption name → transform.
    ``severity`` controls corruption intensity (1–5, matching ImageNet-C).
    """
    import numpy as np
    from PIL import Image, ImageFilter

    base = get_eval_transform(img_size)

    def gaussian_noise(pil_img, sev: int = severity):
        std = [0.04, 0.06, 0.08, 0.10, 0.12][sev - 1]
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        arr = np.clip(arr + np.random.randn(*arr.shape) * std, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))

    def gaussian_blur(pil_img, sev: int = severity):
        radius = [0.4, 0.7, 1.0, 1.5, 2.0][sev - 1]
        return pil_img.filter(ImageFilter.GaussianBlur(radius=radius))

    def contrast(pil_img, sev: int = severity):
        factor = [0.75, 0.60, 0.45, 0.30, 0.15][sev - 1]
        enhancer = transforms.functional.adjust_contrast
        # torchvision functional operates on tensors, so do it on PIL manually
        arr = np.array(pil_img, dtype=np.float32)
        mean_val = arr.mean()
        arr = np.clip(mean_val + factor * (arr - mean_val), 0, 255)
        return Image.fromarray(arr.astype(np.uint8))

    def jpeg_compression(pil_img, sev: int = severity):
        import io
        quality = [80, 65, 50, 35, 20][sev - 1]
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()

    def make_transform(fn):
        return transforms.Compose([
            transforms.Lambda(fn),
            *base.transforms,          # append normalisation
        ])

    return {
        "gaussian_noise":    make_transform(gaussian_noise),
        "gaussian_blur":     make_transform(gaussian_blur),
        "contrast":          make_transform(contrast),
        "jpeg_compression":  make_transform(jpeg_compression),
    }
