"""
@author: qiuwenhui
@Software: VSCode
@Time: 2023-03-03 15:24:38
"""
from nets.deeplabv3_plus import DeepLab


def main():
    model = DeepLab(
        num_classes=7,
        backbone="deeplabv3plus_fusion",
        downsample_factor=4,
        aux_branch=False,
        analyze=False,
    )

    for names, layers in model.named_children():
        print("******************** " + names + " ********************")
        for name, layer in layers.named_children():
            print("^^^^ " + name + " ^^^^")
            for n, l in layer.named_children():
                print("((( " + n + " ))))")


if __name__ == "__main__":
    main()
