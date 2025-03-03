from pathlib import Path

import lightning as L
from torch.utils.data import Dataset, DataLoader
from utils.read import read_raster

from .preprocessing import (
    get_transform,
    get_val_transform,
    get_final_transform,
)
from .misc import resolve_yolo_paths
from .viz import visualize_dataset


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, final_transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = sorted(Path(images_dir).glob("*.tif"))
        self.label_files = sorted(Path(labels_dir).glob("*.tif"))
        self.transform = transform
        self.final_transform = final_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and label
        image_path = self.images_dir / self.image_files[idx]
        image = read_raster(image_path.as_posix())

        label_path = self.labels_dir / self.label_files[idx]
        mask = read_raster(label_path.as_posix(), band=3) # The 3rd band is the boundary mask

        # Apply transformations
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        
        # Apply final transformation(s)
        if self.final_transform is not None:
            augmented = self.final_transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # print(f"Image shape: {image.shape}")
        # print(f"Mask shape: {mask.shape}")

        return image, mask


class SegmentationDataModule(L.LightningDataModule):
    def __init__(self, train_path, validation_path, image_size, batch_size, logger=None):
        super().__init__()
        train_path = Path(train_path)
        validation_path = Path(validation_path)
        train_images_dir, train_labels_dir, val_images_dir, val_labels_dir = resolve_yolo_paths(
            train_path, validation_path
        )
        self.train_images_dir = train_images_dir
        self.train_labels_dir = train_labels_dir
        self.val_images_dir = val_images_dir
        self.val_labels_dir = val_labels_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.logger = logger

    def setup(self, stage=None):
        self.train_dataset = SegmentationDataset(
            images_dir=self.train_images_dir,
            labels_dir=self.train_labels_dir,
            transform=get_transform(self.image_size),
            final_transform=get_final_transform(),
        )
        self.val_dataset = SegmentationDataset(
            images_dir=self.val_images_dir,
            labels_dir=self.val_labels_dir,
            transform=get_val_transform(self.image_size),
            final_transform=get_final_transform(),
        )

        # 🔥 Log a sample of images to TensorBoard before training
        visualize_dataset(self.train_dataset, self.logger, num_images=32)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
