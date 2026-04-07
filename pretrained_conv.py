## Dana Tran
## CS 5330 Project 5
## 04/06/26

import sys

import torch
import matplotlib.pyplot as plt

from torchvision.models import resnet18, ResNet18_Weights

def main(argv):
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.eval()
    print(model)  

    #first conv
    w = model.conv1.weight.detach().cpu()
    print("conv1.weight shape:", tuple(w.shape))
    #second conv
    w2 = model.layer1[0].conv1.weight.detach().cpu()
    print("layer1[0].conv1.weight shape:", tuple(w2.shape))

    n_show = 16 #show 16 filters
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    axes = axes.flatten()
    for i in range(n_show):
        filt = w[i].mean(dim=0).numpy()  # [7, 7]
        ax = axes[i]
        ax.imshow(filt, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(str(i), fontsize=8)
    plt.tight_layout()
    plt.savefig("pretrained_resnet18_conv1_filters.png", dpi=150)
    plt.close()
    print("saved pretrained_resnet18_conv1_filters.png")
    return 0


if __name__=="__main__":
    raise SystemExit(main(sys.argv))