from pathlib import Path


def resolve_yolo_paths(train_dir: Path, val_dir: Path) -> tuple[Path, Path, Path, Path]:
    """
    Resolves training and validation image/label directories based on YOLO dataset structure.
    
    Args:
        train_dir (Path): Path to the training directory.
        val_dir (Path): Path to the validation directory.
        
    Returns:
        tuple[Path, Path, Path, Path]:
            - train_images_dir
            - train_labels_dir
            - val_images_dir
            - val_labels_dir
    """
    
    if (train_dir / "images").exists() and (train_dir / "labels").exists():
        # Structure: train/images, train/labels, val/images, val/labels
        train_images_dir = train_dir / "images"
        train_labels_dir = train_dir / "labels"
        val_images_dir = val_dir / "images"
        val_labels_dir = val_dir / "labels"
    
    elif (train_dir.parent.parent / "images" / train_dir.name).exists() and (train_dir.parent.parent / "labels" / train_dir.name).exists():
        # Structure: images/train, images/val, labels/train, labels/val
        base_train = train_dir.parent.parent
        train_images_dir = base_train / "images" / train_dir.name
        train_labels_dir = base_train / "labels" / train_dir.name
        base_val = val_dir.parent.parent
        val_images_dir = base_val / "images" / val_dir.name
        val_labels_dir = base_val / "labels" / val_dir.name
    
    else:
        raise ValueError("Dataset structure is not recognized. Ensure it follows YOLO conventions.")
    
    return train_images_dir, train_labels_dir, val_images_dir, val_labels_dir
