import argparse
import os
import glob

from PIL import Image
import numpy as np
from tqdm import tqdm

from landmarks import detect_landmarks


def crop_region(img):
    height, width, _ = img.shape
    landmarks = np.array(detect_landmarks(img))

    min_x = max(0, np.min(landmarks[:, 1]))
    max_x = min(np.max(landmarks[:, 1]), height)

    min_y = max(0, np.min(landmarks[:, 0]))
    max_y = min(np.max(landmarks[:, 0]), width)

    return min_x, max_x, min_y, max_y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_path", type=str, required=True)
    parser.add_argument("--ori_path", type=str, required=True)
    args = parser.parse_args()

    img_root = args.ori_path
    new_root = args.new_path

    subjects = os.listdir(img_root)
    for subject in tqdm(subjects):
        type_folders = os.listdir(f"{img_root}/{subject}")
        for type_folder in type_folders:
            image_list = [
                os.path.basename(x)
                for x in glob.glob(f"{img_root}/{subject}/{type_folder}/0*.jpg")
            ]
            sorted_image_list = sorted(image_list, key=lambda x: int(x[4:-4]))

            first_img = np.array(
                Image.open(
                    f"{img_root}/{subject}/{type_folder}/{sorted_image_list[0]}"
                ).convert("RGB")
            )
            top_left, bottom_left, top_right, bottom_right = crop_region(first_img)
            os.makedirs(f"{new_root}/{subject}/{type_folder}", exist_ok=True)
            for image_path in sorted_image_list:
                img = np.array(
                    Image.open(
                        f"{img_root}/{subject}/{type_folder}/{image_path}"
                    ).convert("RGB")
                )
                crop_img = img[top_left:bottom_left, top_right:bottom_right]
                Image.fromarray(crop_img).save(
                    f"{new_root}/{subject}/{type_folder}/{image_path}"
                )
