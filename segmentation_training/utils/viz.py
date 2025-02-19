from pathlib import Path

import matplotlib.pyplot as plt


def plot_losses_and_scores(losses: list[float], scores: list[float], output_file: str | Path = "plot.png"):
    """
    Plots the losses and scores on the same plot with a legend and saves it to a file.

    Args:
        losses (list): List of loss values.
        scores (list): List of score values.
        output_file (str): File name or path to save the plot. Default is 'plot.png'.
    """

    # Ensure lists are valid
    if not losses or not scores:
        raise ValueError("Both 'losses' and 'scores' must be non-empty lists.")
    if len(losses) != len(scores):
        raise ValueError("The lengths of 'losses' and 'scores' must be the same.")

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Losses", color="red", linestyle="--")
    plt.plot(scores, label="Scores", color="blue", linestyle="-")

    # Add labels, title, and legend
    plt.xlabel("Epochs")
    plt.ylabel("Values")
    plt.title("Losses and Scores over Epochs")
    plt.legend()
    plt.grid(True)

    # Save the plot to the specified file
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_images(images, title=None, save_path=None):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(15, 5), dpi=300)
    if n == 1:
        axes.imshow(images[0])  # Directly use axes as it's a single object
        axes.axis("off")
    else:
        for i in range(n):
            axes[i].imshow(images[i])
            axes[i].axis("off")
    if title is not None:
        fig.suptitle(title, y=0.75)
    if save_path is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()
