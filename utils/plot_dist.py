import argparse
import os
import sys

import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import SAMER
from dataset import make_loader, LOSO_sequence_generate

plt.style.use("ggplot")


def transform_one_hot_to_str(one_hots):
    one_hot_list = []

    for one_hot in one_hots:
        non_zero_idx = np.nonzero(one_hot.reshape(-1))[0]
        non_zero_idx = np.sort(non_zero_idx).astype("str").tolist()
        one_hot_list.append("+".join(non_zero_idx))

    return one_hot_list


def plot_dist(args, device):
    # Load in data and generate the data loader
    csv_file = pd.read_csv(args.csv_file, dtype={"Subject": str})
    train_loader, _ = make_loader(
        img_root=args.img_root,
        file_name=csv_file,
        mode="tt",
        batch_size=16,
        micro_catego=args.catego,
    )

    # Set up model and train the model
    model = SAMER(num_classes=args.num_classes).eval().to(device)
    model.load_state_dict(torch.load(args.weight_path))

    au_true = []
    au_features = []

    for images, labels in tqdm(train_loader):
        eyes_img, mouth_img = images
        img_label, au_labels = labels
        eyes_img = eyes_img.to(device)
        mouth_img = mouth_img.to(device)
        img_label = img_label.to(device)

        *_, au_feature = model(eyes_img, mouth_img)

        au_true.extend(transform_one_hot_to_str(au_labels.numpy()))
        au_features.extend(au_feature.cpu().tolist())

    au_true = np.array(au_true)
    au_unique = sorted(np.unique(au_true), key=lambda x: int(x.split("+")[0]))

    au_2d = TSNE(n_components=2, init="pca", learning_rate="auto").fit_transform(
        np.array(au_features)
    )

    plt.figure(figsize=(18, 8))
    linspace = np.linspace(0.0, 1.0, len(au_unique))
    au_color = cm.get_cmap(plt.get_cmap("nipy_spectral"))(linspace)
    for idx, au_cls in enumerate(au_unique):
        features = au_2d[au_true == au_cls]
        plt.scatter(features[:, 0], features[:, 1], color=au_color[idx], label=au_cls)

    if args.catego.lower() == "casme":
        matplotlib.rcParams.update({"legend.labelspacing": 0.25})

    elif args.catego.lower() == "samm":
        matplotlib.rcParams.update({"legend.labelspacing": 0.66})

    lgd = plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        fancybox=True,
        ncol=6,
        fontsize=17,
        handletextpad=0.3,
        facecolor="white",
        edgecolor="black",
        framealpha=1,
    )
    lgd.get_frame().set_linewidth(1.2)
    plt.tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,
    )
    plt.grid(visible=None)
    plt.savefig(args.save_fig, bbox_extra_artists=(lgd,), bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_root",
        type=str,
        required=True,
        help="Path for the micro expression image",
    )
    parser.add_argument(
        "--csv_file", type=str, required=True, help="Path for the csv file"
    )
    parser.add_argument(
        "--catego", type=str, required=True, help="SAMM or CASME dataset"
    )
    parser.add_argument(
        "--weight_path", type=str, required=True, help="Path for the checkpoints file"
    )
    parser.add_argument(
        "--num_classes", type=int, default=5, help="Classes to be trained"
    )
    parser.add_argument("--save_fig", type=str, help="Save fig path")
    args = parser.parse_args()
    device = "cuda:5" if torch.cuda.is_available() else "cpu"

    plot_dist(args, device)
