import argparse
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from data.dataset import SegmentationDataset
from data.preprocessing import get_transform, get_val_transform
from pipeline.training import train
from torch.utils.data import DataLoader
from utils.viz import plot_losses_and_scores


def get_args():
    """
    Parse command-line arguments for the PyTorch script.

    Returns:
        argparse.Namespace: Parsed arguments as an object.
    """
    parser = argparse.ArgumentParser(
        description="Parse arguments for training and validation using PyTorch and SMP models."
    )

    # Required arguments
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training data split (required).")
    parser.add_argument(
        "--validation_path", type=str, required=True, help="Path to the validation data split (required)."
    )
    parser.add_argument(
        "--results_path", type=str, required=True, help="Path to the results, preferably an empty directory (required)."
    )
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs (required).")
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=(640, 640),
        help="Image size for the model: height width (default: 640 640).",
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--architecture", type=str, default="Unet", help="Model architecture supported by SMP (default: Unet)."
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="resnet18",
        help="Model backbone supported by SMP and compatible with the architecture (default: resnet18).",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training.")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Learning rate for training."
    )
    parser.add_argument(
        "--validation_metric",
        type=str,
        default="f1_score",
        help="Metric for validating the model. Should be supported by `segmentation_models_pytorch`"
        + " (all supported metrics are listed in smp.metrics)",
    )

    return parser.parse_args()


def main(args):
    train_images_dir = Path(args.train_path) / "images"
    train_labels_dir = Path(args.train_path) / "labels"

    val_images_dir = Path(args.validation_path) / "images"
    val_labels_dir = Path(args.validation_path) / "labels"

    train_dataset_binary = SegmentationDataset(
        images_dir=train_images_dir,
        labels_dir=train_labels_dir,
        mode="binary",
        transform=get_transform(args.image_size),
    )
    val_dataset_binary = SegmentationDataset(
        images_dir=val_images_dir,
        labels_dir=val_labels_dir,
        mode="binary",
        transform=get_val_transform(args.image_size),
    )

    train_dataloader_binary = DataLoader(train_dataset_binary, batch_size=args.batch_size, shuffle=True)
    val_dataloader_binary = DataLoader(val_dataset_binary, batch_size=args.batch_size, shuffle=False)

    model_binary = smp.create_model(
        arch=args.architecture, encoder_name=args.encoder, in_channels=3, classes=1, activation="sigmoid"
    )

    optimizer_binary = torch.optim.AdamW(model_binary.parameters(), lr=args.lr)

    save_path = Path(args.results_path)
    save_path.mkdir(exist_ok=True, parents=True)
    _, metrics = train(
        model=model_binary,
        epochs=args.epochs,
        optimizer=optimizer_binary,
        train_loader=train_dataloader_binary,
        val_loader=val_dataloader_binary,
        save_path=save_path.as_posix(),
        verbose=True,
        device=torch.device(args.device),
        mode="binary",
    )

    plot_losses_and_scores(
        metrics["val_losses"], metrics["val_metrics"], output_file=save_path / "validation_plots.png"
    )


if __name__ == "__main__":
    main(get_args())
