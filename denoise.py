# coding: utf-8
# Author：Wangzixiang
# Date ：2025/6/25 20:41
import glob
import os

import soundfile as sf
import torch
import yaml
import json
import argparse
from tqdm import tqdm

from model import DCCRN
from BaseModel import load_best_param

parser = argparse.ArgumentParser()
parser.add_argument(
    "--denoise_path", type=str, default="~/DNS-Challenge-master/datasets/xx.wav",required=False, help="Directory containing wav files, or file path"
)
parser.add_argument(
    "--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution"
)
parser.add_argument("--exp_dir", default="logs", help="Experiment root")


def main(conf):
    # Get best trained model
    model = DCCRN()
    model = load_best_param(conf["exp_dir"],model)

    if conf["use_gpu"]:
        model = model.cuda()
    model_device = next(model.parameters()).device
    print(model_device)

    # Get a list of wav files (or single wav file)
    save_folder = os.path.join(conf["exp_dir"], "denoise")
    os.makedirs(save_folder, exist_ok=True)

    conf["denoise_path"] = os.path.expanduser(conf["denoise_path"])

    # 获取文件列表
    if os.path.isfile(conf["denoise_path"]):
        all_wavs = [conf["denoise_path"]]
    else:
        # 规范化目录名称
        base_name = os.path.basename(conf["denoise_path"].rstrip('/'))
        save_folder = os.path.join(save_folder, base_name)
        os.makedirs(save_folder, exist_ok=True)
        
        # 修复路径拼接
        search_path = os.path.join(conf["denoise_path"], "*.wav")
        all_wavs = glob.glob(search_path)
    
    # 添加文件检查
    if not all_wavs:
        print(f"Error: No WAV files found at {conf['denoise_path']}")
        return
    
    print(f"Processing {len(all_wavs)} files...")

    print(all_wavs)
    for wav_path in tqdm(all_wavs):
        print(wav_path)
        mix, fs = sf.read(wav_path, dtype="float32")
        with torch.no_grad():
            net_inp = torch.tensor(mix)[None].to(model_device)
            estimate = model.forward(net_inp).squeeze().cpu().data.numpy()
        # Save the estimate speech
        wav_name = os.path.basename(wav_path)
        sf.write(os.path.join(save_folder, wav_name), estimate, fs)


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    main(arg_dic)