import argparse
import json
from pathlib import Path

import torch
from pipeline.validation import validate_model


def get_args():
    """
    Parse command-line arguments for the PyTorch script.

    Returns:
        argparse.Namespace: Parsed arguments as an object.
    """
    parser = argparse.ArgumentParser(
        description="Parse arguments for testing a semantic segmentation model using PyTorch and SMP models."
    )

    # Required arguments
    parser.add_argument("--images_path", type=str, required=True, help="Path to the folder with images (required).")
    parser.add_argument("--labels_path", type=str, required=True, help="Path to the folder with labels (required).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the traced model (required).")
    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Path to the results forlder, preferably an empty directory (required).",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        nargs=2,
        default=(640, 640),
        help="Tile size for the model: height width (default: 640 640).",
    )
    parser.add_argument(
        "--tile_step",
        type=int,
        nargs=2,
        default=(480, 480),
        help="Tile step for the model: height width (default: 480 480).",
    )

    # Optional arguments with defaults
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold to binarize the masks.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device.",
    )
    parser.add_argument(
        "--no_viz",
        action="store_true",
        help="A flag to disable saving the results as images.",
    )

    return parser.parse_args()


def main(args):
    save_dir = Path(args.results_path)
    save_dir.mkdir(exist_ok=True, parents=True)

    metrics_path = save_dir / "metrics.json"
    if metrics_path.exists():
        raise ValueError(f"Metrics path {metrics_path.as_posix()} already exist! Please choose a different folder.")

    images_list = list(Path(args.images_path).glob("*.jpg"))
    labels_list = []
    for image_path in images_list:
        label_dir = list(Path(args.labels_path).glob(image_path.stem))[0]
        labels_list.append(label_dir)

    model = torch.jit.load(args.model_path)

    if not args.no_viz:
        viz_dir = save_dir / "visualization"
        viz_dir.mkdir(exist_ok=True)
    else:
        viz_dir = None

    metrics = validate_model(
        model=model,
        image_paths=images_list,
        label_paths=labels_list,
        tile_size=args.tile_size,
        tile_step=args.tile_step,
        conf=args.conf,
        viz_dir=viz_dir,
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main(get_args())
