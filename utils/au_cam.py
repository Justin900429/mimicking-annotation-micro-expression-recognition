import sys
import os
import argparse

import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from matplotlib import cm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import SAMER


class GradCAM:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name

        # Save the forward and backward features
        self.module_list = []
        self.features_list = dict()
        self.gradient_list = dict()

        # Handlers list
        self.handlers = []
        self._register_hook()

        self.img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def activation_name(self, name, hook_type):
        def _get_hook(module, input, output):
            if hook_type == "forward":
                self.module_list.append(name)
                self.features_list[name] = output
            elif hook_type == "backward":
                self.gradient_list[name] = output[0]
            else:
                raise ValueError(f"Nor supported hook type: {hook_type}")

        return _get_hook

    def _clear_list(self):
        self.module_list = []
        self.features_list = dict()
        self.gradient_list = dict()

    def _register_hook(self):
        self.handlers = []
        for (name, module) in self.model.named_modules():
            if name in self.layer_name:
                self.handlers.append(
                    module.register_forward_hook(self.activation_name(name, "forward"))
                )
                self.handlers.append(
                    module.register_full_backward_hook(
                        self.activation_name(name, "backward")
                    )
                )

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, ori_img, in_features, device):
        # Clear the list
        self._clear_list()
        ori_img = cv2.resize(ori_img, (224, 224))
        eyes_img = np.array(ori_img)[:112, ...]
        mouth_img = np.array(ori_img)[112:, ...]

        in_features = self.img_transform(in_features).to(device)
        _, f_h, _ = in_features.shape
        eyes = in_features[:, : f_h // 2, :].unsqueeze(0)
        mouth = in_features[:, f_h // 2 :, :].unsqueeze(0)

        res, features = self.model.predict(eyes, mouth, get_weight=True)
        res = res.view(-1)
        features = features.view(-1)

        self.model.zero_grad()
        eyes = features[:512].argmax(dim=-1).item()
        eyes_au_target = features[:512][eyes]
        eyes_au_target.backward(retain_graph=True)

        mouth = features[512:].argmax(dim=-1).item()
        mouth_au_target = features[512:][mouth]
        mouth_au_target.backward(retain_graph=True)

        origin_image = [eyes_img, mouth_img]
        cams = []
        for idx, module in enumerate(self.module_list):
            gradient = self.gradient_list[module].data[0]
            weight = torch.mean(gradient, dim=(1, 2), keepdim=True)

            feature = self.features_list[module].data[0]
            temp_cam = (feature * weight).sum(dim=0).relu()

            # Normalize the value
            temp_cam -= torch.min(temp_cam)
            temp_cam /= torch.max(temp_cam)

            temp_cam = cv2.resize(
                temp_cam.detach().cpu().numpy(), origin_image[idx].shape[:2][::-1]
            )
            heatmap = (cm.jet(temp_cam)[..., :3] * 255).astype("uint8")
            temp_cam = (origin_image[idx] * 0.5 + heatmap * 0.5).astype("uint8")
            cams.append(temp_cam)

        res = np.vstack(cams)
        return res


def save_grad_cam(file_name, img_root, weight_file, num_classes, save_path, catego):
    data = pd.read_csv(file_name, dtype={"Subject": str})
    device = "cuda:5"

    # Define model and grad cam instance
    model = SAMER(num_classes=num_classes).eval().to(device)
    model.load_state_dict(torch.load(weight_file))
    grad_cam = GradCAM(
        model, ["eyes_branch.layers.2.1.sa", "mouth_branch.layers.2.1.sa"]
    )

    for idx in tqdm(range(len(data))):
        # Open the image
        subject = data.loc[idx, "Subject"]

        # Load in the matrix that already been preprocessed
        if catego == "casme":
            base_path = f"sub{subject}/{data.loc[idx, 'Filename']}"
        elif catego == "samm":
            base_path = f"{subject}/{data.loc[idx, 'Filename']}"
        elif catego == "smic":
            base_path = (
                f"{subject}/micro/{data.loc[idx, 'label']}/{data.loc[idx, 'Filename']}"
            )
        else:
            raise ValueError(f"Not supported category: {catego}")

        # Get image paths
        ori_img = Image.open(f"{img_root}/{base_path}/amplify_le.jpg").convert("RGB")
        optical_img = Image.open(f"{img_root}/{base_path}/optical.jpg").convert("RGB")

        # Get grad cam image and save it
        res = grad_cam(np.array(ori_img), optical_img, device)
        Image.fromarray(res).save(
            f"{save_path}/{subject}_{data.loc[idx, 'Filename']}.jpg"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--img_root", type=str, required=True)
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--catego", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--save_path", type=str, default="utils/new_au")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    save_grad_cam(
        file_name=args.csv_file,
        img_root=args.img_root,
        weight_file=args.weight_path,
        save_path=args.save_path,
        num_classes=args.num_classes,
        catego=args.catego.lower(),
    )

    # model = SAMER()
    # for name, module in model.named_modules():
    #     print(name)
