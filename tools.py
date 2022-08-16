from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, recall_score
import torch


def plot_confusion_matrix(
    data,
    label_mapping: dict,
    fig_name: str,
):
    type_mapping = label_mapping.values()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.matshow(data, cmap="Blues")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Target")
    ax.set_xticks(np.arange(len(type_mapping)))
    ax.set_xticklabels(list(type_mapping))
    ax.yaxis.set_label_position("right")
    ax.set_yticks(np.arange(len(type_mapping)))
    ax.set_yticklabels(list(type_mapping))

    for (i, j), z in np.ndenumerate(data):
        ax.text(j, i, z, ha="center", va="center")
    plt.savefig(fig_name, dpi=1200)


def evaluate(test_loader, model, device):
    # Set into evaluation mode
    model.eval()
    test_accuracy = 0.0
    test_f1_score = 0.0

    total_prediction = []
    total_true = []

    with torch.no_grad():
        for images, labels in test_loader:
            eyes_img, mouth_img = images
            labels = labels
            eyes_img = eyes_img.to(device)
            mouth_img = mouth_img.to(device)
            # images = images.to(device)
            labels = labels.to(device)

            output = model.predict(eyes_img, mouth_img)

            # Compute the accuracy
            prediction = output.argmax(-1) == labels
            test_accuracy += prediction.sum().item() / labels.size(0)
            test_f1_score += f1_score(
                labels.view(-1).tolist(),
                output.argmax(-1).view(-1).tolist(),
                average="macro",
            )

            total_prediction.extend(output.argmax(-1).view(-1).tolist())
            total_true.extend(labels.view(-1).tolist())

    return (
        test_accuracy / len(test_loader),
        test_f1_score / len(test_loader),
        total_prediction,
        total_true,
    )
