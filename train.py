import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, recall_score
from tqdm import tqdm

import torch

from model import SAMER
from loss import WeightedSupCoL
from dataset import make_loader, LOSO_sequence_generate
from tools import plot_confusion_matrix, evaluate


def get_au_similarity(au):
    # (and) / (or)
    bool_au = torch.clone(au.bool())
    bool_au = bool_au.unsqueeze(1).repeat(1, au.shape[0], 1)
    union_au = torch.logical_or(bool_au, bool_au.permute(1, 0, 2)).sum(dim=-1)
    diff_au = torch.logical_and(bool_au, bool_au.permute(1, 0, 2)).sum(dim=-1)
    similarity = diff_au / union_au
    return similarity


def train_cls(model, train_loader, test_loader, device, loso_idx, weight, args):
    # Define criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1)
    au_criterion = WeightedSupCoL()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.cls_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    max_f1 = -1
    max_acc = -1
    # Start training
    for epoch in range(args.cls_epochs):
        # Set model in training mode
        model.train()
        # Training Metrics
        train_loss = 0.0

        for batch_idx, (images, labels) in enumerate(
            tqdm(train_loader, desc=f"Cls Epochs {epoch + 1}")
        ):
            eyes_img, mouth_img = images
            img_label, au_labels = labels
            eyes_img = eyes_img.to(device)
            mouth_img = mouth_img.to(device)
            img_label = img_label.to(device)
            au_labels = au_labels.to(device)

            eyes_features, mouth_features, au_features = model(eyes_img, mouth_img)
            eye_cls_loss = criterion(eyes_features, img_label)
            mouth_cls_loss = criterion(mouth_features, img_label)

            # Filter the au labels
            au_labels_similarity = get_au_similarity(au_labels)
            au_loss = au_criterion(au_features, au_labels_similarity)

            lambda_ = 0.3

            if au_loss is not None:
                total_loss = (1 - lambda_) * (
                    eye_cls_loss + mouth_cls_loss
                ) + lambda_ * (au_loss)
            else:
                total_loss = eye_cls_loss + mouth_cls_loss

            optimizer.zero_grad()
            total_loss.backward()
            train_loss += total_loss.item()

            # Update the parameters
            optimizer.step()

        scheduler.step()

        train_loss /= len(train_loader)
        test_accuracy, test_f1, *_ = evaluate(test_loader, model, device)

        print(f"Train loss: {train_loss:.4f}")
        print(f"Test accuracy, f1: {test_accuracy:.4f}, {test_f1:.4f}")
        # adjust_learning_rate(optimizer, epoch, args)

        if (test_accuracy > max_acc) or (
            (test_accuracy == max_acc) and (test_f1 > max_f1)
        ):
            max_f1 = test_f1
            max_acc = test_accuracy
            torch.save(
                model.state_dict(), f"{args.weight_save_path}/model_best_{loso_idx}.pt"
            )


def LOSO_train(
    data: pd.DataFrame, sub_column: str, args, label_mapping: dict, device: torch.device
):
    log_file = open(f"{args.weight_save_path}/weight.log", "w")

    # Create different DataFrame for each subject
    train_list, test_list, loso_list = LOSO_sequence_generate(data, sub_column)
    test_accuracy = 0.0
    test_f1_score = 0.0

    save_prediction = []
    save_total_true = []

    for LOSO_idx in range(len(train_list)):
        print(f"=================LOSO {loso_list[LOSO_idx]}=====================")

        # Create dataset and dataloader
        train_cls_loader, cls_dataset = make_loader(
            img_root=args.img_root,
            file_name=train_list[LOSO_idx],
            mode="training_con",
            batch_size=args.batch_size,
            micro_catego=args.catego,
            num_workers=4,
            pin_memory=True,
        )
        test_loader, _ = make_loader(
            img_root=args.img_root,
            file_name=test_list[LOSO_idx],
            mode="testing",
            batch_size=len(test_list[LOSO_idx]),
            micro_catego=args.catego,
        )

        # Train the data
        label_weight = cls_dataset.compute_data_weight(column="emo_label").to(device)

        # Read in the model
        model = SAMER(num_classes=args.num_classes).to(device)
        train_cls(
            model=model,
            train_loader=train_cls_loader,
            test_loader=test_loader,
            device=device,
            loso_idx=loso_list[LOSO_idx],
            weight=label_weight,
            args=args,
        )
        model.load_state_dict(
            torch.load(
                f"{args.weight_save_path}/model_best_{loso_list[LOSO_idx]}.pt",
                map_location=device,
            )
        )
        temp_test_accuracy, temp_f1_score, temp_prediction, temp_true = evaluate(
            test_loader=test_loader, model=model, device=device
        )
        save_prediction.extend(temp_prediction)
        save_total_true.extend(temp_true)

        print(
            f"In LOSO {loso_list[LOSO_idx]}, test accuracy: {temp_test_accuracy:.4f}, F1-Score: {temp_f1_score:.4f}"
        )
        log_file.write(
            f"LOSO {loso_list[LOSO_idx]}: Accuracy: {temp_test_accuracy:.4f}, F1-Score: {temp_f1_score:.4f}\n"
        )
        test_accuracy += temp_test_accuracy
        test_f1_score += temp_f1_score

    confusion_matrix_compute = confusion_matrix(save_total_true, save_prediction)
    plot_confusion_matrix(
        confusion_matrix_compute,
        label_mapping=label_mapping,
        fig_name=f"{args.weight_save_path}/{args.svg_name}",
    )

    count_correct = 0
    count_total = 0
    for idx in range(len(save_prediction)):
        if save_prediction[idx] == save_total_true[idx]:
            count_correct += 1
        count_total += 1
    double_total_f1_score = f1_score(save_total_true, save_prediction, average="macro")
    UAR = recall_score(save_total_true, save_prediction, average="macro")

    print(
        f"Mean LOSO accuracy: {test_accuracy / len(train_list):.4f}, f1-score: {test_f1_score / len(train_list):.4f}"
    )
    print(
        f"Unweighted LOSO accuracy: {count_correct / count_total:.4f}, f1-score: {double_total_f1_score:.4f}"
    )
    print(f"UAR: {UAR:.4f}")
    log_file.write(
        f"Mean LOSO accuracy: {test_accuracy / len(train_list):.4f}, F1-Score: {test_f1_score / len(train_list):.4f}\n"
    )
    log_file.write(
        f"Unweighted LOSO accuracy: {count_correct / count_total:.4f}, f1-score: {double_total_f1_score:.4f}\n"
    )
    log_file.write(f"UAR: {UAR:.4f}")
    log_file.close()


if __name__ == "__main__":
    # Argument parse
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
        "--num_classes", type=int, default=5, help="Classes to be trained"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Training batch size"
    )
    parser.add_argument(
        "--weight_save_path",
        type=str,
        default="weight",
        help="Path for the saving weight",
    )
    parser.add_argument(
        "--svg_name", type=str, default="test.pdf", help="Confusion matrix name"
    )
    parser.add_argument(
        "--cls_epochs", type=int, default=20, help="Epochs for training classifier"
    )
    parser.add_argument(
        "--cls_lr", type=float, default=3e-3, help="Learning rate for classification"
    )
    args = parser.parse_args()

    # Training device
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

    # Create folders for the saving weight
    os.makedirs(args.weight_save_path, exist_ok=True)

    data = pd.read_csv(args.csv_file, dtype={"Subject": str})
    label_mapping = {
        idx: emotion
        for idx, emotion in zip(
            np.unique(data.loc[:, "emo_label"]), np.unique(data.loc[:, "Label"])
        )
    }
    label_mapping = dict(sorted(label_mapping.items()))

    # Train the model
    LOSO_train(
        data=data,
        sub_column="Subject",
        label_mapping=label_mapping,
        args=args,
        device=device,
    )
