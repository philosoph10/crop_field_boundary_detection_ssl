import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transform(image_size: tuple[int, int]):
    return A.Compose(
        [
            # Data augmentations
            A.HorizontalFlip(p=0.5),  # Random horizontal flip with 50% probability
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=30, border_mode=0, p=0.3
            ),  # Shifting, scaling, rotating
            A.OneOf(
                [
                    A.Compose(
                        [
                            A.ColorJitter(
                                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.8, p=0.3
                            ),  # Color augmentation
                            A.GaussianBlur(blur_limit=(3, 7), p=0.6),  # Optional blur to simulate sensor noise
                            A.GridDistortion(p=0.1),  # Grid distortion for elastic effects
                        ],
                        p=0.5,
                    ),
                    A.Compose(
                        [
                            A.OneOf(
                                [
                                    A.RandomSunFlare(flare_roi=(0, 0, 1, 1), src_radius=200, p=0.7),
                                    A.RandomShadow(shadow_roi=(0, 0, 1, 1), shadow_intensity_range=(0.4, 0.6), p=0.3),
                                ],
                                p=0.6,
                            ),
                            A.OneOf([A.ISONoise(p=0.7), A.AdditiveNoise(p=0.3)], p=0.5),
                        ],
                        p=0.5,
                    ),
                ],
                p=0.7,
            ),
            A.Resize(*image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Standard normalization
            ToTensorV2(),  # Convert to PyTorch tensor
        ]
    )


def get_val_transform(image_size: tuple[int, int]):
    return A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.Resize(*image_size),
            ToTensorV2(),
        ]
    )
