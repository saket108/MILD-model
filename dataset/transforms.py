from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(image_size: int = 640, train: bool = True) -> A.Compose:
    """Build Albumentations transforms."""
    if train:
        transforms = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ]
    else:
        transforms = [
            A.Resize(image_size, image_size),
            A.Normalize(),
            ToTensorV2(),
        ]

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.0),
    )
