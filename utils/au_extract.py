import re
import json
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F


eyes_AU = list(range(1, 10)) + list(range(41, 45)) + list(range(61, 65))
eyes_AU_cls_map = {au: idx for idx, au in enumerate(eyes_AU)}
mouths_AU = list(range(10, 29)) + [38]
mouth_AU_cls_map = {au: idx for idx, au in enumerate(mouths_AU)}


def plot_confusion_matrix(
    data,
    type_mapping,
    fig_name: str,
):
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.matshow(data, cmap="Blues")
    ax.set_xticks(np.arange(len(type_mapping)))
    ax.set_xticklabels(list(type_mapping), rotation=60)
    ax.yaxis.set_label_position("right")
    ax.set_yticks(np.arange(len(type_mapping)))
    ax.set_yticklabels(list(type_mapping))

    for (i, j), z in np.ndenumerate(data):
        ax.text(j, i, f"{z:.1f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(fig_name)


def compute_AU_categories_weight(data_info):
    eyes_labels = defaultdict(int)
    eyes_labels[0] = 0
    mouth_labels = defaultdict(int)
    mouth_labels[0] = 0

    eyes_mapping = {"": 0}
    count_eyes = 1
    mouth_mapping = {"": 0}
    count_mouth = 1

    for idx in range(len(data_info)):
        EYES_FLAG = False
        MOUTH_FLAG = False
        au_labels = list(re.findall(r"\d+", data_info.loc[idx, "Action Units"]))
        for au_label in au_labels:
            if int(au_label) in eyes_AU:
                if au_label not in eyes_mapping:
                    eyes_mapping[au_label] = count_eyes
                    count_eyes += 1
                eyes_labels[eyes_mapping[au_label]] += 1
                EYES_FLAG = True
            if int(au_label) in mouths_AU:
                if au_label not in mouth_mapping:
                    mouth_mapping[au_label] = count_mouth
                    count_mouth += 1
                mouth_labels[mouth_mapping[au_label]] += 1
                MOUTH_FLAG = True

        if not EYES_FLAG:
            eyes_labels[0] += 1
        if not MOUTH_FLAG:
            mouth_labels[0] += 1

    # Make class
    eyes_weight = max(eyes_labels.values()) / torch.tensor(list(eyes_labels.values()))
    mouth_weight = max(mouth_labels.values()) / torch.tensor(
        list(mouth_labels.values())
    )

    return eyes_weight, mouth_weight, eyes_mapping, mouth_mapping


def get_AU_one_hot(AU_labels, eyes_mapping, mouth_mapping):
    eyes_labels = []
    mouth_labels = []

    au_labels = list(re.findall(r"\d+", AU_labels))
    for au_label in au_labels:
        if au_label in eyes_mapping:
            eyes_labels.append(eyes_mapping[au_label])
        if au_label in mouth_mapping:
            mouth_labels.append(mouth_mapping[au_label])

    if len(eyes_labels) == 0:
        eyes_labels.append(0)
    if len(mouth_labels) == 0:
        mouth_labels.append(0)

    eyes_labels = torch.LongTensor(eyes_labels)
    mouth_labels = torch.LongTensor(mouth_labels)
    eyes_one_hot = F.one_hot(eyes_labels, num_classes=len(eyes_mapping)).sum(dim=0)
    mouth_one_hot = F.one_hot(mouth_labels, num_classes=len(mouth_mapping)).sum(dim=0)

    return eyes_one_hot.float(), mouth_one_hot.float()


def extract_au(au):
    return np.array((re.findall(r"\d+", au))).astype("int")


def get_AU_one_hot_with_labels(ori_au, new_au):
    AU_list = np.zeros((100,))

    eyes_aus = extract_au(ori_au)
    mouth_aus = extract_au(new_au)

    for eyes_au in eyes_aus:
        AU_list[eyes_au] = 1
    for mouth_au in mouth_aus:
        AU_list[mouth_au] = 1
    return torch.tensor(AU_list)


def separate_au(df, save_file_path, eyes_conf, mouth_conf, catego):
    assert isinstance(
        df, (str, pd.DataFrame)
    ), f"Type '{type(df)}' not supported. Support only str and pd.DataFrame"
    if isinstance(df, str):
        # Read in data
        df = pd.read_csv(df, dtype={"Subject": str})

    # Record for new facial labels
    mouth_labels = []
    eyes_labels = []

    data = df.loc[:, "Action Units"]
    total_AU = set()
    # Split the action list
    for idx, unit in enumerate(data):
        # Find only the digit
        au_list = set(np.array(re.findall(r"\d+", unit)).astype("int"))
        total_AU = total_AU | au_list

        eyes = []
        mouth = []
        for au in au_list:
            # For eyes
            if au in eyes_AU:
                eyes.append(au)

            # For mouth
            if au in mouths_AU:
                mouth.append(au)

        eyes = sorted(eyes)
        eyes_str = "+".join([str(ins) for ins in eyes])
        eyes_labels.append(eyes_str)

        mouth = sorted(mouth)
        mouth_str = "+".join([str(ins) for ins in mouth])
        mouth_labels.append(mouth_str)

    # Process the class
    eyes_unique = sorted(list(set(eyes_labels)))
    eyes_keys = {label: idx for idx, label in enumerate(eyes_unique)}
    mouth_unique = sorted(list(set(mouth_labels)))
    mouth_keys = {label: idx for idx, label in enumerate(mouth_unique)}

    df["eyes_au"] = [eyes_keys[label] for label in eyes_labels]
    df["mouth_au"] = [mouth_keys[label] for label in mouth_labels]

    df.to_csv(save_file_path, index=False)

    compute_AU_distance(eyes_keys, eyes_conf, catego)
    compute_AU_distance(mouth_keys, mouth_conf, catego)


def compute_AU_distance(distintct_au, save_place, catego):
    with open(f"{save_place}_{catego}.json", "w") as file:
        json.dump(distintct_au, file, indent=4)

    num_labels = len(distintct_au)
    reverse_au = {value: key for key, value in distintct_au.items()}
    weight_matrix = [[0 for _ in range(num_labels)] for _ in range(num_labels)]

    for row in range(num_labels):
        for col in range(num_labels):
            row_keys = reverse_au[row].split("+")
            col_keys = reverse_au[col].split("+")

            diff_keys_legnth = len(set(row_keys) & set(col_keys)) / len(
                set(row_keys) | set(col_keys)
            )
            weight_matrix[row][col] = diff_keys_legnth

    np.save(f"{save_place}_{catego}.npy", np.array(weight_matrix))
    plot_confusion_matrix(
        np.array(weight_matrix), distintct_au.keys(), f"{save_place}_{catego}.pdf"
    )


if __name__ == "__main__":
    # Argument parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True, help="Load filename")
    parser.add_argument(
        "--save_path_file", type=str, required=True, help="Save filename"
    )
    parser.add_argument("--catego", type=str, required=True, help="Category of dataset")
    parser.add_argument(
        "--eyes_conf",
        type=str,
        default="npy_file/eyes",
        help="Save place for eyes-weight matrix",
    )
    parser.add_argument(
        "--mouth_conf",
        type=str,
        default="npy_file/mouth",
        help="Save place for mouth-weight matrix",
    )
    args = parser.parse_args()

    separate_au(
        args.csv_file,
        args.save_path_file,
        args.eyes_conf,
        args.mouth_conf,
        args.catego.lower(),
    )
