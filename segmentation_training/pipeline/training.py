from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from tqdm import tqdm


def train(
    model,
    epochs,
    optimizer,
    train_loader,
    val_loader,
    validation_metric="f1_score",
    save_path=None,
    scheduler=None,
    verbose=False,
    device=None,
    mode="binary",
):
    # Define loss function
    dice_loss = smp.losses.DiceLoss(mode=mode)

    # Metrics storage
    best_metric = 0.0
    metrics = {"train_losses": [], "val_losses": [], "val_metrics": []}

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    metric_fn = getattr(smp.metrics, validation_metric)

    for epoch in range(epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        epoch_progress = tqdm(train_loader, leave=True, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for images, labels in epoch_progress:
            images = images.to(device)
            if mode != "multilabel":
                labels = labels.to(device)
            else:
                labels = [label.to(device) for label in labels]

            optimizer.zero_grad()
            outputs = model(images)
            loss = dice_loss(outputs.contiguous(), labels.long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # Update progress bar with current average loss
            epoch_progress.set_postfix({"Train Loss": train_loss / (len(metrics["train_losses"]) + 1)})

        # Scheduler step (if scheduler is defined)
        if scheduler:
            scheduler.step()

        # Save epoch's training loss
        metrics["train_losses"].append(train_loss / len(train_loader))

        # Validation loop
        model.eval()
        val_loss = 0.0
        metric_score = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, leave=False, desc="Validation"):
                images = images.to(device)
                if mode != "multilabel":
                    labels = labels.to(device)
                else:
                    labels = [label.to(device) for label in labels]

                outputs = model(images)

                loss = dice_loss(outputs.contiguous(), labels.long())
                val_loss += loss.item()

                # Calculate metric score for each batch
                if mode == "multiclass":
                    tp, fp, fn, tn = smp.metrics.get_stats(
                        outputs.contiguous().argmax(dim=1), labels, mode=mode, num_classes=5
                    )
                    score_batch = metric_fn(tp, fp, fn, tn, reduction="micro")
                elif mode == "binary":
                    hard_res = (outputs.contiguous() > 0.5).to(torch.int)[:, 0]
                    tp, fp, fn, tn = smp.metrics.get_stats(hard_res, labels, mode=mode)
                    score_batch = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                    if 0 <= score_batch and score_batch <= 1:
                        metric_score += score_batch
                    else:
                        assert torch.sum(hard_res) == 0 or torch.sum(labels) == 0, "Unclear why score is invalid!"
                else:
                    raise ValueError(f"Mode {mode} is not supported.")

        # Calculate mean IoU across validation set
        mean_metric = metric_score / len(val_loader)

        # Save validation metrics
        val_agg_loss = val_loss / len(val_loader)
        metrics["val_losses"].append(float(val_agg_loss))
        metrics["val_metrics"].append(float(mean_metric))

        # Check if the current model is the best so far, and save it if it is
        if mean_metric > best_metric:
            best_metric = mean_metric
            if save_path:
                Path(save_path).mkdir(exist_ok=True, parents=True)
                save_file = Path(save_path) / "best.pt"
                torch.save(model.state_dict(), save_file.as_posix())  # Save best model

        if epoch == epochs - 1 and save_path:
            Path(save_path).mkdir(exist_ok=True, parents=True)
            save_file = Path(save_path) / "last.pt"
            torch.save(model.state_dict(), save_file.as_posix())

        if verbose:
            print(
                f"Epoch {epoch+1}/{epochs}, Train Loss: {metrics['train_losses'][-1]:.4f}, "
                f"Val Loss: {metrics['val_losses'][-1]:.4f}, {validation_metric}: {metrics['val_metrics'][-1]:.4f}"
            )

    return best_metric, metrics
