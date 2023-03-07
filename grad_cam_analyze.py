"""
@author: qiuwenhui
@Software: VSCode
@Time: 2023-03-03 10:10:13
"""
import os
import warnings
from utils.utils import cvtColor, preprocess_input, resize_image

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from nets.deeplabv3_plus import DeepLab
from pytorch_grad_cam import GradCAM
from rich.progress import track


pred_cfg = dict(
    # ---------- 深度卷积神经网络模型的超参数 ----------
    model_path="./logs/deeplabv3plus_fusion/02_deeplabv3plus_fusion_version3_MobileVit_d4_Normal_sknet_500epochs_bs16_adam/ep485-loss0.235-val_loss0.426.pth",
    backbone="deeplabv3plus_fusion",
    input_shape=[512, 512],
    downsample_factor=4,
    deploy=True,
    num_classes=7,
    classes_name=[
        "BW",
        "HD",
        "WR",
        "RO",
        "RI",
        "FV",
        "SR",
    ],
    aux_branch=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
    # ---------- 文件夹路径 ----------
    mode="single",  # single, multi
    dir_path="./img_analyze",
    save_root="./img_analyze_cam",
    # dir_path="../version3_img_new",
    # save_root="./version3_img_new_cam",
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
        analyze=False,
    )
    model_weight_path = pred_cfg["model_path"]
    model.load_state_dict(
        torch.load(model_weight_path, map_location=device), strict=False
    )
    if pred_cfg["deploy"]:
        model.switch_to_deploy()
    print(f"{model_weight_path} model, and classes loaded.")
    model = model.to(device)

    img_root = pred_cfg["dir_path"]
    img_list = os.listdir(img_root)
    for img_index, img in enumerate(img_list):
        print(
            f"******************** 🔦🔦🔦 analyze {img_index+1}/{len(img_list)} image ********************"
        )
        if not img.endswith(".jpg"):
            continue
        img_path = os.path.join(pred_cfg["dir_path"], img)
        # ******************** 预处理输入图片 ********************
        input_shape = pred_cfg["input_shape"]
        image = Image.open(img_path)
        image = cvtColor(image)  # 将图像转换成RGB图像
        # 给图像增加灰条，实现不失真的resize
        image_data, nw, nh = resize_image(image, (input_shape[1], input_shape[0]))
        origin_image = np.array(image_data)
        rgb_img = np.float32(origin_image) / 255
        # 添加上batch_size维度
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)),
            0,
        )

        # ******************** 获取模型预测掩码 ********************
        model.eval()
        with torch.no_grad():
            images = torch.from_numpy(image_data).to(device)
            result = model(images).main
            print(result.shape)
        result = F.softmax(result, dim=1).cpu()
        sem_classes = pred_cfg["classes_name"]

        # ******************** 逐个类别进行分析 ********************
        for num, class_name in enumerate(sem_classes):
            print(
                f"📝📝📝 analyze {class_name} classes, {num+1}/{len(sem_classes)} classes"
            )
            sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
            target_category = sem_class_to_idx[class_name]
            target_mask = result[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
            target_mask_uint8 = 255 * np.uint8(target_mask == target_category)
            target_mask_float = np.float32(target_mask == target_category)
            both_images = np.hstack(
                (
                    origin_image,
                    np.repeat(target_mask_uint8[:, :, None], 3, axis=-1),
                )
            )

            # ******************** 保存真实图片和目标掩码的对比 ********************
            img_dir_path = os.path.join(save_root, img.split(".")[0])
            if not os.path.exists(img_dir_path):
                os.mkdir(img_dir_path)
            class_save_path = os.path.join(img_dir_path, f"{num}_" + class_name)
            if not os.path.exists(class_save_path):
                os.mkdir(class_save_path)
            compare_img = Image.fromarray(both_images)
            compare_img.save(
                os.path.join(class_save_path, f"{class_name}_mask.jpg"), quality=95
            )
            origin_image_save = Image.fromarray(origin_image)
            origin_image_save.save(
                os.path.join(class_save_path, f"{class_name}_input.jpg"), quality=95
            )

            class SemanticSegmentationTarget:
                def __init__(self, category, mask):
                    self.category = category
                    self.mask = torch.from_numpy(mask)
                    if torch.cuda.is_available():
                        self.mask = self.mask.cuda()

                def __call__(self, model_output):
                    return (model_output[self.category, :, :] * self.mask).sum()

            layer_name_multi = [
                # "model.backbone.conv1",
                # "model.backbone.conv2",
                # "model.backbone.stage1",
                # "model.backbone.transition1[0]",
                # "model.backbone.transition1[1]",
                # "model.backbone.stage2[0].fuse_layers[0][0]",
                # "model.backbone.stage2[0].fuse_layers[1][0]",
                # "model.backbone.transition2[0]",
                # "model.backbone.transition2[1]",
                # "model.backbone.transition2[2]",
                # "model.backbone.stage3[0].fuse_layers[0][0]",
                # "model.backbone.stage3[0].fuse_layers[1][0]",
                # "model.backbone.stage3[0].fuse_layers[2][0]",
                # "model.backbone.transition3[0]",
                # "model.backbone.transition3[1]",
                # "model.backbone.transition3[2]",
                # "model.backbone.transition3[3]",
                # ******************** Backbone Stage3 ********************
                # # branch
                # "model.backbone.stage3[0].branches[0][0]",
                # "model.backbone.stage3[0].branches[1][0]",
                # "model.backbone.stage3[0].branches[2][0]",
                # "model.backbone.stage3[0].branches[0][1]",
                # "model.backbone.stage3[0].branches[1][1]",
                # "model.backbone.stage3[0].branches[2][1]",
                # # transition3
                # "model.backbone.transition3[0]",
                # "model.backbone.transition3[1]",
                # "model.backbone.transition3[2]",
                # "model.backbone.transition3[3]",
                # ReLU Layer
                "model.backbone.stage2[0].relu",
                "model.backbone.stage3[0].relu",
                # ******************** Backbone Stage4 ********************
                # branch
                # "model.backbone.stage4[0].branches[0][0]",
                # "model.backbone.stage4[0].branches[1][0]",
                # "model.backbone.stage4[0].branches[2][0]",
                # "model.backbone.stage4[0].branches[3][0]",
                # "model.backbone.stage4[0].branches[0][1]",
                # "model.backbone.stage4[0].branches[1][1]",
                # "model.backbone.stage4[0].branches[2][1]",
                # "model.backbone.stage4[0].branches[3][1]",
                # stage4 fuse_layer
                "model.backbone.stage4[0].fuse_layers[0][0]",
                "model.backbone.stage4[0].fuse_layers[0][1]",
                "model.backbone.stage4[0].fuse_layers[0][2]",
                "model.backbone.stage4[0].fuse_layers[0][3]",
                "model.backbone.stage4[0].relu",
                # ******************** ASPP Module ********************
                "model.aspp.branch1",
                "model.aspp.branch2",
                "model.aspp.branch3",
                "model.aspp.branch4",
                "model.aspp.branch5_relu",
                "model.aspp",
                # ******************** Concat Module ********************
                "model.cat_conv1",
                "model.cat_conv3",
                "model.cls_conv",
                # ******************** Shortcut Module ********************
                # "model.conv1_shortcut",
                # "model.stage1_shortcut",
                # "model.stage2_shortcut",
                # "model.stage3_shortcut",
                # "model.stage4_shortcut",
            ]

            layer_name_sigle = [
                # ******************** Backbone Stage3 ********************
                "model.backbone.stage2[0].relu",
                "model.backbone.stage3[0].relu",
                # ******************** Backbone Stage4 ********************
                # stage4 fuse_layer
                "model.backbone.stage4[0].fuse_layers[0][0]",
                "model.backbone.stage4[0].fuse_layers[0][1]",
                "model.backbone.stage4[0].fuse_layers[0][2]",
                "model.backbone.stage4[0].fuse_layers[0][3]",
                "model.backbone.stage4[0].relu",
                # ******************** ASPP Module ********************
                "model.aspp.branch1",
                "model.aspp.branch2",
                "model.aspp.branch3",
                "model.aspp.branch4",
                "model.aspp.branch5_relu",
                "model.aspp",
                # ******************** Concat Module ********************
                "model.cat_conv1",
                "model.cat_conv3",
                "model.cls_conv",
            ]
            if pred_cfg["mode"] == "multi":
                layer_name = layer_name_multi
            elif pred_cfg["mode"] == "single":
                layer_name = layer_name_sigle

            for index, name in enumerate(track(layer_name)):
                target_layers = [eval(name)]
                # target_layers = [model.aspp]

                targets = [
                    SemanticSegmentationTarget(target_category, target_mask_float)
                ]
                cam = GradCAM(
                    model=model,
                    target_layers=target_layers,
                    use_cuda=torch.cuda.is_available(),
                )
                grayscale_cam = cam(
                    input_tensor=images,
                    targets=targets,
                    # aug_smooth=True,
                    # eigen_smooth=True,
                )[0, :]

                # fusion rgb image
                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                cam_img = Image.fromarray(cam_image)
                save_name = f"{index}_" + name.replace(".", "_") + ".png"
                cam_img.save(os.path.join(class_save_path, save_name), quality=95)

                # rgb cam
                rgb_cam = show_cam_on_image(
                    rgb_img, grayscale_cam, use_rgb=True, image_weight=0.0
                )
                rgb_cam = Image.fromarray(rgb_cam)
                rgb_cam_name = f"{index}_" + name.replace(".", "_") + "_rgbcam" + ".png"
                rgb_cam.save(os.path.join(class_save_path, rgb_cam_name), quality=95)

                # grayscale cam
                gray_cam = Image.fromarray((np.uint8(255 * grayscale_cam))).convert(
                    "RGB"
                )
                gray_cam_name = (
                    f"{index}_" + name.replace(".", "_") + "_graycam" + ".png"
                )
                gray_cam.save(os.path.join(class_save_path, gray_cam_name), quality=95)


if __name__ == "__main__":
    main(pred_cfg)
