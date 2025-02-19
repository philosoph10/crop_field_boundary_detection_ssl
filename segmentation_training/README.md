# Semantic Segmentation Training and Validation with PyTorch and SMP

This project trains and validates a semantic segmentation model using PyTorch and [Segmentation Models PyTorch (SMP)](https://github.com/qubvel/segmentation_models.pytorch). The training and validation pipelines support various model architectures and encoders from SMP and allow configuring multiple parameters.

## Features
- Train and validate a semantic segmentation model with configurable architecture and encoder.
- Uses PyTorch DataLoader for efficient batch processing.
- Supports binary segmentation (merging all masks into a single binary mask).
- Configurable training parameters such as batch size, learning rate, number of epochs, etc.
- Saves trained model and validation plots.
- Supports sliding window inference for validation.
- Computes IoU and F1 scores, both cumulative and per image.
- Allows saving visualizations of model performance during validation.

## Dataset Structure
The dataset is structured into different splits (e.g., `train`, `val`). Each split contains:

```
train/
    images/
        im_0001.jpg
        im_0002.jpg
        ...
    labels/
        im_0001/
            mask1.png
            mask2.png
        im_0002/
            mask1.png
```
- **Images**: Contain the input images in `.jpg` format.
- **Labels**: Contain segmentation masks in `.png` format, grouped in subdirectories corresponding to image filenames.
- Multiple masks per image are combined into a single binary mask for training.

## Installation
```sh
pip install -r requirements.txt
```

## Training Usage
Run the training script with the required parameters:
```sh
python train.py --train_path /path/to/train \
                --validation_path /path/to/val \
                --results_path /path/to/results \
                --epochs 50 \
                --architecture Unet \
                --encoder resnet18 \
                --batch_size 4 \
                --lr 1e-4
```

### Important Configurable Parameters
- **Train and Validation Paths**: Paths to the dataset splits (`--train_path`, `--validation_path`).
- **Results Path**: Where to save model checkpoints and plots (`--results_path`).
- **Epochs**: Number of training epochs (`--epochs`).
- **Architecture**: Model architecture from SMP (`--architecture`, e.g., `Unet`).
- **Encoder**: Backbone model for feature extraction (`--encoder`, e.g., `resnet18`).
- **Batch Size**: Number of samples per batch (`--batch_size`).
- **Learning Rate**: Initial learning rate for training (`--lr`).

## Validation Usage
Run the validation script with the required parameters:
```sh
python validate.py --images_path /path/to/images \
                   --labels_path /path/to/labels \
                   --model_path /path/to/model.pt \
                   --results_path /path/to/results \
                   --tile_size 640 640 \
                   --tile_step 480 480 \
                   --conf 0.5
```

### Important Configurable Parameters
- **Images and Labels Paths**: Paths to the images and labels for validation (`--images_path`, `--labels_path`).
- **Model Path**: Path to the trained (traced) model for validation (`--model_path`).
- **Results Path**: Where to save evaluation results (`--results_path`).
- **Tile Size**: Sliding window tile size (`--tile_size`, default: 640x640).
- **Tile Step**: Sliding window step size (`--tile_step`, default: 480x480).
- **Confidence Threshold**: Confidence threshold to binarize masks (`--conf`, default: 0.5).
- **Disable Visualization**: Use `--no_viz` to disable saving visualizations.

## Output
- **Metrics**: The script outputs IoU and F1 scores, both cumulative and per image, saved in a JSON file.
- **Visualization**: If enabled, the model's performance is saved as images in the `visualization/` directory.

## TODO
1. Add automatic tracing for the best and last models.
2. Add support for multiclass/multilabel segmentation.
