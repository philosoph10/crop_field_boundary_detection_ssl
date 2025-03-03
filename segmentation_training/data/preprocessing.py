import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transform(image_size: tuple[int, int]):
    return A.Compose(
        [
            # Data augmentations
            A.HorizontalFlip(p=0.5),  # Random horizontal flip with 50% probability
            A.VerticalFlip(p=0.5),  # Random vertical flip with 50% probability
            A.RandomRotate90(),  # Random 90 degree rotation

            A.Resize(*image_size),
        ]
    )


def get_val_transform(image_size: tuple[int, int]):
    return A.Compose(
        [
            A.Resize(*image_size),
        ]
    )


def get_final_transform():
    return A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406, 0.5), std=(0.229, 0.224, 0.225, 0.5)),  # Standard normalization
            ToTensorV2(),  # Convert to PyTorch tensor
        ]
    )
