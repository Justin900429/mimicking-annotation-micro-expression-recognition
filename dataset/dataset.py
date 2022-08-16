import random
import functools

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms

from utils import get_AU_one_hot_with_labels


def LOSO_sequence_generate(data: pd.DataFrame, sub_column: str) -> tuple:
    """Generate train and test data using leave-one-subject-out for
    specific subject

    Parameters
    ----------
    data : pd.DataFrame
        Original DataFrame
    sub_column : str
        Subject column in DataFrame

    Returns
    -------
    tuple
        Return training and testing list DataFrame
    """
    # Save the training and testing list for all subject
    train_list = []
    test_list = []

    # Unique subject in `sub_column`
    subjects = np.unique(data[sub_column])

    for subject in subjects:
        # Mask for the training
        mask = data["Subject"].isin([subject])

        # Masking for the specific data
        train_data = data[~mask].reset_index(drop=True)
        test_data = data[mask].reset_index(drop=True)

        train_list.append(train_data)
        test_list.append(test_data)

    try:
        subjects = subjects.astype("int").tolist()
    except ValueError:
        pass

    return train_list, test_list, subjects


def patch_masking(img, base_len=10):
    channel, height, width = img.shape
    choice = [0, 1, 2]

    for row in range(0, height, base_len * 2):
        for col in range(0, width, base_len * 2):
            if ((row + base_len) < height) and ((col + base_len) < width):
                mask_type = random.choice(choice)

                if mask_type == 0:
                    v = torch.empty(
                        [channel, base_len, base_len], dtype=torch.float32
                    ).normal_()
                elif mask_type == 1:
                    v = 0
                else:
                    v = 1
                img[..., row : row + base_len, col : col + base_len] = v

    return img


class CropAreaMask:
    def __init__(self, sep_num=2):
        self.sep_num = sep_num
        self.area_list = list(range(sep_num))
        self.choice = list(range(3))

    def get_param(self, img):
        height_idx = random.choice(self.area_list)
        width_idx = random.choice(self.area_list)
        channel, height, width = img.shape

        height_uint = height // self.sep_num
        width_unit = width // self.sep_num

        start_h = height_uint * height_idx
        end_h = height_uint * (height_idx + 1)

        start_w = width_unit * width_idx
        end_w = width_unit * (width_idx + 1)

        fill_type = random.choice(self.choice)

        if fill_type == 0:
            v = torch.empty(
                [channel, height_uint, width_unit], dtype=torch.float32
            ).normal_()
        elif fill_type == 1:
            v = 0.0
        else:
            v = 1.0

        return start_h, end_h, start_w, end_w, v

    def __call__(self, img):
        start_h, end_h, start_w, end_w, v = self.get_param(img)
        img[..., start_h:end_h, start_w:end_w] = v

        return img


class Identity:
    def __call__(self, x):
        return x


class MicroDataset(Dataset):
    def __init__(self, img_root, file, catego, mode):
        if isinstance(file, str):
            self.data_info = pd.read_csv(file, dtype={"Subject": str})
        else:
            self.data_info = file

        self.img_root = img_root
        self.mode = mode
        self.catego = catego.lower()

        if catego == "samm":
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                # mean=[0.5118, 0.5237, 0.4511],
                # std=[0.2177, 0.1817, 0.1652]
            )
        elif catego == "casme":
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                # mean=[0.3775, 0.4642, 0.3105],
                # std=[0.1494, 0.1431, 0.1461]
            )
        else:
            raise ValueError(f"No {catego} category found")

        self.resize = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((224, 224))]
        )
        if mode == "training_con":
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomErasing(scale=(0.02, 0.3), value="random"),
                ]
            )

    def __len__(self):
        return len(self.data_info)

    def compute_data_weight(self, column):
        value = torch.tensor(
            self.data_info.groupby(column).count().sort_index().iloc[:, 0].values
        )
        return torch.div(torch.full_like(value, torch.max(value)), value)

    def open_and_preprocess(self, path):
        optical_frame = Image.open(path).convert("RGB")
        optical_frame = self.resize(optical_frame)

        if "training" in self.mode:
            optical_frame = self.transform(optical_frame)

        optical_frame = self.normalize(optical_frame)
        return optical_frame

    def random_exchange(self, emo_label):
        same_label_cls = self.data_info[self.data_info.loc[:, "emo_label"] == emo_label]
        get_sample = same_label_cls.sample().reset_index(drop=True)
        subject = get_sample.loc[0, "Subject"]
        catego = get_sample.loc[0, "catego"]

        if catego == "casme":
            base_path = f"sub{subject}/{get_sample.loc[0, 'Filename']}"
        elif catego == "samm":
            base_path = f"{subject}/{get_sample.loc[0, 'Filename']}"
        elif catego == "smic":
            base_path = f"{subject}/micro/{get_sample.loc[0, 'label']}/{get_sample.loc[0, 'Filename']}"
        else:
            raise ValueError("Wrong categories {}", catego)

        optical_frame = self.open_and_preprocess(
            f"{self.img_root}/{base_path}/optical.jpg"
        )
        _, h, _ = optical_frame.shape
        mouth_frame = optical_frame[:, h // 2 :, :]

        return mouth_frame, get_sample.loc[0, "Action Units"]

    def __getitem__(self, idx: int):
        # Label for the image
        AUs = self.data_info.loc[idx, "Action Units"]
        label = self.data_info.loc[idx, "emo_label"]
        subject = self.data_info.loc[idx, "Subject"]
        catego = self.data_info.loc[idx, "catego"]

        # Load in the matrix that already been preprocessed
        if catego == "casme":
            base_path = f"sub{subject}/{self.data_info.loc[idx, 'Filename']}"
        elif catego == "samm":
            base_path = f"{subject}/{self.data_info.loc[idx, 'Filename']}"
        elif catego == "smic":
            base_path = f"{subject}/micro/{self.data_info.loc[idx, 'label']}/{self.data_info.loc[idx, 'Filename']}"
        else:
            raise ValueError("Wrong categories {}", catego)

        optical_frame = self.open_and_preprocess(
            f"{self.img_root}/{base_path}/optical.jpg"
        )

        _, h, _ = optical_frame.shape
        eyes_frame = optical_frame[:, : h // 2, :]

        if ("training" in self.mode) and (random.random() > 0.5):
            mouth_frame, new_AU = self.random_exchange(label)
        else:
            mouth_frame = optical_frame[:, h // 2 :, :]
            new_AU = AUs

        if ("training" in self.mode) or ("tt" in self.mode):
            # Obtain one hot AU
            au_one_hot = get_AU_one_hot_with_labels(AUs, new_AU)

        if ("training" in self.mode) or ("tt" in self.mode):
            return (eyes_frame, mouth_frame), (label, au_one_hot)
        else:
            return (eyes_frame, mouth_frame), label


def make_loader(
    img_root,
    file_name,
    batch_size,
    micro_catego,
    mode="training",
    num_workers=0,
    drop_last=False,
    pin_memory=False,
):
    assert mode in [
        "training_con",
        "training_cls",
        "testing",
        "tt",
    ], f"{mode} not supports."
    dataset = MicroDataset(img_root, file_name, mode=mode, catego=micro_catego)

    if "training" in mode:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            pin_memory=pin_memory,  # sampler=ImbalancedDatasetSampler(dataset),
            drop_last=drop_last,
            num_workers=num_workers,
            shuffle=True,
        )
    else:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    return dataloader, dataset
