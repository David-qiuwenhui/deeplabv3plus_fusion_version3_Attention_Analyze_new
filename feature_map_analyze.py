"""
@author: qiuwenhui
@Software: VSCode
@Time: 2023-03-02 11:41:45
"""
import math
import os
import time
import cv2
import numpy as np
from PIL import Image
from utils.utils import time_synchronized
import torch
from nets.deeplabv3_plus import DeepLab
from torch import nn
from utils.utils import cvtColor, resize_image, preprocess_input
import matplotlib.pyplot as plt
from rich.progress import track


pred_cfg = dict(
    # ---------- 深度卷积神经网络模型的超参数 ----------
    model_path="./logs/deeplabv3plus_fusion/02_deeplabv3plus_fusion_version3_MobileVit_d4_Normal_sknet_500epochs_bs16_adam/ep485-loss0.235-val_loss0.426.pth",
    backbone="deeplabv3plus_fusion",
    input_shape=[512, 512],
    downsample_factor=4,
    deploy=True,
    num_classes=7,
    name_classes=[
        "Background_waterbody",
        "Human_divers",
        "Wrecks_and_ruins",
        "Robots",
        "Reefs_and_invertebrates",
        "Fish_and_vertebrates",
        "sea_floor_and_rocks",
    ],
    aux_branch=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    # ---------- 文件夹路径 ----------
    img_dir="../version3_img",
    save_root="./version3_img_feature",
)


def main(pred_cfg):
    # ******************** 计算设备类型和保存路径 ********************
    device_type = pred_cfg["device"]
    device = torch.device(device_type)
    print(f"\033[1;36;46m 🔌🔌🔌🔌Use {device} for predicting \033[0m")
    save_root = pred_cfg["save_root"]
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    # ******************** 实例化深度卷积模型 ********************
    model = DeepLab(
        num_classes=pred_cfg["num_classes"],
        backbone=pred_cfg["backbone"],
        downsample_factor=pred_cfg["downsample_factor"],
        aux_branch=pred_cfg["aux_branch"],
        analyze=True,
    )
    model_weight_path = pred_cfg["model_path"]
    model.load_state_dict(
        torch.load(model_weight_path, map_location=device), strict=False
    )
    if pred_cfg["deploy"]:
        model.switch_to_deploy()
    print(f"{model_weight_path} model, and classes loaded.")
    model = model.to(device)

    img_dir = pred_cfg["img_dir"]
    for num, img in enumerate(os.listdir(img_dir)):
        if not img.endswith(".jpg"):
            continue
        print(f"🔦🔦🔦 analyze {num+1}/{len(os.listdir(img_dir))} image")
        img_save_root = os.path.join(save_root, img.split(".")[0])
        if not os.path.exists(img_save_root):
            os.mkdir(img_save_root)
        # ******************** 预处理输入图片 ********************
        file_path = os.path.join(img_dir, img)
        input_shape = pred_cfg["input_shape"]
        image = Image.open(file_path)
        image = cvtColor(image)
        # 给图像增加灰条，实现不失真的resize
        image_data, nw, nh = resize_image(image, (input_shape[1], input_shape[0]))
        # 添加上batch_size维度
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)),
            0,
        )

        model.eval()
        with torch.no_grad():
            images = torch.from_numpy(image_data).to(device)
            result = model(images)

        layers_list = [
            "conv1",
            "conv2",
            "stage1",
            "stage2_mid0",
            "stage2_mid1",
            "stage2_0",
            "stage2_1",
            "stage3_mid0",
            "stage3_mid1",
            "stage3_mid2",
            "stage3_0",
            "stage3_1",
            "stage3_2",
            "stage4_mid0",
            "stage4_mid1",
            "stage4_mid2",
            "stage4_mid3",
            "stage4",
            "aspp",
            "concat1",
            "concat3",
        ]
        show_num = 16
        cmap_type = "jet"
        index = 0

        for index, layer in enumerate(track(layers_list)):
            feature = eval("result." + layer)
            feature = np.squeeze(feature.detach().cpu().numpy())
            save_path = os.path.join(img_save_root, f"{index}_{layer}")
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            for c in range(math.ceil(feature.shape[0] / show_num)):
                plt.figure(figsize=(12, 9), dpi=100)
                start_index = show_num * c
                end_index = show_num * (c + 1)
                for i in range(start_index, end_index):
                    # 创建单个子图
                    plt.subplot(
                        int(show_num**0.5), int(show_num**0.5), i % show_num + 1
                    )
                    plt.imshow(
                        feature[i, :, :],
                        cmap=cmap_type,
                    )
                # plt.subplots_adjust(wspace=0.1, hspace=0.1)
                plt.savefig(
                    os.path.join(save_path, layer + f"_{c}.jpg"),
                    dpi=100,
                    bbox_inches="tight",
                )
                # plt.show()
                plt.close()


if __name__ == "__main__":
    main(pred_cfg)
