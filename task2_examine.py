## Dana Tran
## CS 5330 Project 5
## 04/06/26

import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

from task1_network import MyNetwork

def load_model(path: str = "mnist_cnn.pt"):
    model = MyNetwork()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def mnist_train_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


#first training image as tensor
def first_train_image_tensor(data_dir: str = "mnist_data"):
    ds = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=mnist_train_transform()
    )
    img, label = ds[0]
    return img, int(label)


def main(argv):
    model = load_model()
    print(model)
    with torch.no_grad():
        w = model.conv1.weight
        print("conv1.weight shape:", tuple(w.shape))
        print(w)

    #10 filters 3x4 grid
    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    axes = axes.flatten()
    with torch.no_grad():
        kernels = w.cpu().numpy()
    for i in range(10):
        ax = axes[i]
        filt = kernels[i, 0]
        im = ax.imshow(filt, cmap="gray")
        ax.set_title(f"filter {i}")
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(10, 12):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig("conv1_filters.png", dpi=150)
    plt.close(fig)
    print("saved conv1_filters.png")

    img_t, y0 = first_train_image_tensor()
    print("first train label:", y0)

    img_np = img_t.squeeze(0).numpy().astype(np.float32)
    with torch.no_grad():
        kernels = model.conv1.weight.cpu().numpy()
    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    axes = axes.flatten()
    for i in range(10):
        k = kernels[i, 0].astype(np.float32)
        resp = cv2.filter2D(img_np, -1, k)
        ax = axes[i]
        ax.imshow(resp, cmap="gray")
        ax.set_title(f"response {i}")
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(10, 12):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig("conv1_filterResults.png", dpi=150)
    plt.close(fig)
    print("saved conv1_filterResults.png")
    return 0

if __name__=="__main__":
    raise SystemExit(main(sys.argv))