import argparse
import pandas as pd
import tensorflow as tf
import setproctitle
from configobj import ConfigObj
from validate import Validator
from magnet import MagNet3Frames
from utils import load_train_data, mkdir, imread, save_images, to_numpy
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root", type=str, required=True)
    parser.add_argument("--csv_file", type=str, required=True)
    args = parser.parse_args()

    configspec = ConfigObj("configs/configspec.conf", raise_errors=True)
    config = ConfigObj(
        "configs/o3f_hmhm2_bg_qnoise_mix4_nl_n_t_ds3.conf",
        configspec=configspec,
        raise_errors=True,
        file_error=True,
    )
    config.validate(Validator())
    network_type = config["architecture"]["network_arch"]
    exp_name = config["exp_name"]
    setproctitle.setproctitle("{}_{}_{}".format("run", network_type, exp_name))
    tfconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    data_info = pd.read_csv(args.csv_file, dtype={"Subject": str})
    img_root = args.img_root
    with tf.Session(config=tfconfig) as sess:
        model = MagNet3Frames(sess, exp_name, config["architecture"])
        checkpoint = config["training"]["checkpoint_dir"]
        model.setup_for_inference(checkpoint, 384, 384)

        for idx in tqdm(range(len(data_info))):
            subject = data_info.loc[idx, "Subject"]
            base_path = f"{subject}/{data_info.loc[idx, 'Filename']}"
            onset = f"{img_root}/{base_path}/img{data_info.loc[idx, 'Onset']}.jpg"
            apex = f"{img_root}/{base_path}/img{data_info.loc[idx, 'Apex']}.jpg"

            save_path = f"{img_root}/{base_path}/amplify_le.jpg"
            out_amp = model.inference(onset, apex, 6)
            save_images(out_amp, [1, 1], save_path)
