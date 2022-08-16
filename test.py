import argparse
import pandas as pd
from sklearn.metrics import f1_score, recall_score

import torch

from model import SAMER
from dataset import make_loader, LOSO_sequence_generate
from tools import evaluate


def test(args):
    data = pd.read_csv(args.csv_file, dtype={"Subject": str})
    train_list, test_list, loso_list = LOSO_sequence_generate(data, "Subject")

    device = "cuda:5"

    save_prediction = []
    save_total_true = []

    for LOSO_idx in range(len(train_list)):
        print(f"=================LOSO {loso_list[LOSO_idx]}=====================")

        test_loader, _ = make_loader(
            img_root=args.img_root,
            file_name=test_list[LOSO_idx],
            mode="testing",
            batch_size=len(test_list[LOSO_idx]),
            micro_catego=args.catego.lower(),
        )

        model = SAMER(num_classes=args.num_classes).to(device)
        model.load_state_dict(
            torch.load(
                f"{args.weight_root}/model_best_{loso_list[LOSO_idx]}.pt",
                map_location=device,
            )
        )
        model.eval()

        *_, temp_prediction, temp_true = evaluate(
            test_loader=test_loader, model=model, device=device
        )

        save_prediction.extend(temp_prediction)
        save_total_true.extend(temp_true)

        count_correct = 0
        count_total = 0

    for idx in range(len(save_prediction)):
        if save_prediction[idx] == save_total_true[idx]:
            count_correct += 1
        count_total += 1
    double_total_f1_score = f1_score(save_total_true, save_prediction, average="macro")
    UAR = recall_score(save_total_true, save_prediction, average="macro")
    print(
        f"Unweighted LOSO accuracy: {count_correct / count_total:.4f}, f1-score: {double_total_f1_score:.4f}"
    )
    print(f"UAR: {UAR:.4f}")


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
        "--weight_root", required=True, type=str, help="Classes to be trained"
    )
    parser.add_argument(
        "--catego", type=str, required=True, help="SAMM or CASME dataset"
    )
    parser.add_argument(
        "--num_classes", type=int, default=5, help="Classes to be trained"
    )
    args = parser.parse_args()

    test(args)
