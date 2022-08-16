import argparse

import pandas as pd
from tqdm import tqdm
from PIL import Image

from optical_related import *

save_area_1 = ["left", "right", "mouth"]
save_area_2 = ["eye", "mouth"]


def compute_features(onset_frame: np.array, apex_frame: np.array):
    # Compute flow
    flow = TVL1_optical_flow(prev_frame=onset_frame, next_frame=apex_frame)
    # Compute magnitude and strain
    flow_mag = TVL1_magnitude(flow)
    strain_mag = optical_strain(flow)

    return flow_mag, strain_mag


def crop_from_dataset(file_name, catego, img_root):
    data = pd.read_csv(file_name, dtype={"Subject": str})

    for data_idx in tqdm(range(len(data))):
        # Open the image
        subject = data.loc[data_idx, "Subject"]

        # Load in the matrix that already been preprocessed
        if catego == "casme":
            base_path = f"sub{subject}/{data.loc[data_idx, 'Filename']}"
            onset = f"{img_root}/{base_path}/img{data.loc[data_idx, 'Onset']}.jpg"
            apex = f"{img_root}/{base_path}/img{data.loc[data_idx, 'Apex']}.jpg"
            offset = f"{img_root}/{base_path}/img{data.loc[data_idx, 'Offset']}.jpg"
        elif catego == "samm":
            base_path = f"{subject}/{data.loc[data_idx, 'Filename']}"
            onset = f"{img_root}/{base_path}/{subject}_{data.loc[data_idx, 'Onset']:05d}.jpg"
            apex = (
                f"{img_root}/{base_path}/{subject}_{data.loc[data_idx, 'Apex']:05d}.jpg"
            )
            offset = f"{img_root}/{base_path}/{subject}_{data.loc[data_idx, 'Offset']:05d}.jpg"
        else:
            raise ValueError(f"Not supported category: {catego}")

        amplify_path = f"{img_root}/{base_path}/amplify_le.jpg"

        # Obtain the faces area
        onset_frame = np.array(Image.open(onset).convert("RGB"))
        apex_frame = np.array(Image.open(apex).convert("RGB"))
        offset_frame = np.array(Image.open(offset).convert("RGB"))
        amplify_frame = np.array(
            Image.open(amplify_path).convert("RGB").resize(onset_frame.shape[:2][::-1])
        )

        if catego == "casme":
            target_frame = apex_frame
        else:
            target_frame = amplify_frame

        on_mag, _ = compute_features(onset_frame, target_frame)
        off_mag, _ = compute_features(offset_frame, target_frame)

        # Restrict the noise by the mean threshold
        on_mag_thresh = np.mean(on_mag) * 1.5
        on_mag[on_mag < on_mag_thresh] = 0
        off_mag_thresh = np.mean(off_mag) * 1.5
        off_mag[off_mag < off_mag_thresh] = 0

        combine_img = np.array(
            [cv2.cvtColor(amplify_frame, cv2.COLOR_RGB2GRAY), on_mag, off_mag]
        ).transpose(1, 2, 0)
        Image.fromarray(combine_img).save(f"{img_root}/{base_path}/optical.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, required=True)
    parser.add_argument("--catego", type=str)
    parser.add_argument("--img_root", type=str, required=True)
    args = parser.parse_args()

    crop_from_dataset(
        img_root=args.img_root, file_name=args.file_name, catego=args.catego.lower()
    )
