# Semantic Segmentation Training and Validation with PyTorch and SMP

This project trains and validates a binary segmentation model using PyTorch and [Segmentation Models PyTorch (SMP)](https://github.com/qubvel/segmentation_models.pytorch). The training and validation pipelines support various model architectures and encoders from SMP and allow configuring multiple parameters.

## Features
- Train and validate a semantic segmentation model with configurable architecture and encoder.
- Uses PyTorch DataLoader for efficient batch processing.
- Supports (only) binary segmentation (merging all masks into a single binary mask).
- Configurable training parameters such as batch size, learning rate, number of epochs, etc.
- Logs the model's training stats to tensorboard.
- Supports sliding window inference for validation.
- Computes IoU and F1 scores, both cumulative and per image.

## Dataset Structure
The dataset is structured into different splits (e.g., `train`, `val`). Each split contains:

```
train/
    images/
        im_0001.tif
        im_0002.tif
        ...
    labels/
        im_0001.tif
        im_0002.tif
```
- **Images**: Contain the input images.
- **Labels**: Contain segmentation masks.

Images/labels and splits directories might be swapped. Also, the names of "images" and "labels" directories are not fixed.

## Installation
```sh
pip install -r requirements.txt
```

## Training Usage
Run the training script with the required parameters:
```sh
python train.py --config_path config/main.yaml
```

### Important Configurable Parameters
- **Train and Validation Paths**: Paths to the dataset splits.
- **Results Path**: Where to save model checkpoints and plots.
- **Epochs**: Number of training epochs.
- **Architecture**: Model architecture from SMP, e.g. `Unet`.
- **Encoder**: Backbone model for feature extraction, e.g.`resnet18`.
- **Batch Size**: Number of samples per batch.
- **Learning Rate**: Initial learning rate for training.

## Validation Usage
Run the validation script with the required parameters:
```sh
python validate.py --images_path /path/to/images \
                   --labels_path /path/to/labels \
                   --model_path /path/to/model.pt \
                   --results_path /path/to/results \
                   --tile_size 256 256 \
                   --tile_step 192 192 \
                   --conf 0.5
```

### Important Configurable Parameters
- **Images and Labels Paths**: Paths to the images and labels for validation (`--images_path`, `--labels_path`).
- **Model Path**: Path to the trained (traced) model for validation (`--model_path`).
- **Results Path**: Directory (preferabely empty) for saving evaluation results (`--results_path`).
- **Tile Size**: Sliding window tile size (`--tile_size`, default: 640x640).
- **Tile Step**: Sliding window step size (`--tile_step`, default: 480x480).
- **Confidence Threshold**: Confidence threshold to binarize masks (`--conf`, default: 0.5).
- **Disable Visualization**: Use `--no_viz` to disable saving visualizations.

## Output
- **Metrics**: The script outputs IoU and F1 scores, both cumulative and per image, saved in a JSON file.
- **Visualization**: If enabled, the model's performance is saved as images in the `visualization/` directory.
