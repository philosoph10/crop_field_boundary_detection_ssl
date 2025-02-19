from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset
from utils.read import read_rgb


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, labels_dir, mode="binary", class_dict=None, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = sorted(Path(images_dir).glob("*.jpg"))
        self.label_folders = sorted(labels_dir.iterdir())
        self.transform = transform
        self.class_dict = class_dict
        self.mode = mode

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and label
        image_path = self.images_dir / self.image_files[idx]
        image = read_rgb(image_path.as_posix())
        assert image.shape

        label_folder = self.labels_dir / self.label_folders[idx]
        label_paths = list(label_folder.glob("*.png"))
        if self.mode == "multiclass":
            labels = {
                label_path.stem: cv2.imread(label_path.as_posix())[:, :, 0].astype(np.uint8) // 255
                for label_path in label_paths
            }
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for class_name, class_mask in labels.items():
                mask[class_mask > 0] = self.class_dict[class_name]

            # Apply transformations
            if self.transform is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
        elif self.mode == "multilabel":
            labels = [cv2.imread(label_path.as_posix())[:, :, 0].astype(np.uint8) // 255 for label_path in label_paths]

            # Apply transformations
            if self.transform is not None:
                augmented = self.transform(image=image, masks=labels)
                image = augmented["image"]
                mask = augmented["masks"]

        elif self.mode == "binary":
            labels = {
                label_path.stem: cv2.imread(label_path.as_posix())[:, :, 0].astype(np.uint8) // 255
                for label_path in label_paths
            }
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for _, class_mask in labels.items():
                mask[class_mask > 0] = 1

            # Apply transformations
            if self.transform is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

        # print(f"Image shape: {image.shape}")
        # print(f"Mask shape: {mask.shape}")

        return image, mask
