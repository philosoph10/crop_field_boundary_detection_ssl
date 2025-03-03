import argparse
from pathlib import Path
import shutil

import yaml
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from pipeline.training import SegmentationModel
from data.dataset import SegmentationDataModule
from utils.exports import export_model


def load_config(config_path):
    """Loads training configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def get_args():
    parser = argparse.ArgumentParser(description="Lightning-based Training Script for Semantic Segmentation")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML config file.")
    return parser.parse_args()


def main(args):
    # ✅ Load config from YAML
    config = load_config(args.config_path)

    save_path = Path(config["results_path"])
    save_path.mkdir(exist_ok=True, parents=True)
    shutil.copy2(args.config_path, save_path / "config.yaml")
    export_path = save_path / "exports"

    # ✅ TensorBoard Logger
    logger = TensorBoardLogger(save_dir=save_path, name="tensorboard_logs")

    model = SegmentationModel(
        architecture=config["architecture"],
        encoder=config["encoder"],
        lr=config["lr"],
        validation_metric=config["validation_metric"],
    )

    data_module = SegmentationDataModule(
        train_path=config["train_path"],
        validation_path=config["validation_path"],
        image_size=tuple(config["image_size"]),
        batch_size=config["batch_size"],
        logger=logger,
    )

    # ✅ Model Checkpoint Callbacks
    best_model_callback = ModelCheckpoint(
        dirpath=save_path, filename="best", save_top_k=1, monitor="val_loss", mode="min"
    )
    last_model_callback = ModelCheckpoint(
        dirpath=save_path, filename="last", save_top_k=1, save_last=True
    )

    trainer = L.Trainer(
        max_epochs=config["epochs"],
        accelerator=config["device"] if config["device"] != "auto" else "gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[best_model_callback, last_model_callback],
        logger=logger,
    )

    trainer.fit(model, datamodule=data_module)

    # ✅ Export Models (after training)
    best_model_path = save_path / "best.ckpt"
    last_model_path = save_path / "last.ckpt"

    if best_model_path.exists():
        export_model(best_model_path, export_path, input_shape=(1, 4, *config["image_size"]))
    if last_model_path.exists():
        export_model(last_model_path, export_path, input_shape=(1, 4, *config["image_size"]))


if __name__ == "__main__":
    main(get_args())
