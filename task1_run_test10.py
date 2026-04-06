## Dana Tran
## CS 5330 Project 1
## 04/05/26

#load saved cnn and run on first 10 mnist test images

import sys
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from task1_network import MyNetwork

def denormalize_for_display(t_chw):
    mean = 0.1307
    std = 0.3081
    return (t_chw * std + mean).clamp(0.0, 1.0)


def load_model(weights_path: str = "mnist_cnn.pt"):
    model = MyNetwork()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


def make_test_dataset(data_dir: str = "mnist_data"):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    return datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    

def main(argv):
    model = load_model("mnist_cnn.pt")
    test_set = make_test_dataset()
    images = []
    labels = []
    for i in range(10):
        img, lab = test_set[i]
        images.append(img)
        labels.append(lab)
    batch = torch.stack(images, dim=0) 
    with torch.no_grad():
        out = model(batch) 
    print("=== First 10 MNIST test examples (in order) ===")
    for i in range(10):
        vals = out[i].tolist()
        vals_str = ", ".join(f"{v:6.2f}" for v in vals)
        pred = int(out[i].argmax().item())
        true = int(labels[i])
        print(f"example {i}: outputs=[{vals_str}]  argmax={pred}  label={true}")


    #3x3 plot
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    axes = axes.flatten()
    for j in range(9):
        img_disp = denormalize_for_display(images[j]).squeeze(0).numpy()
        pred = int(out[j].argmax().item())
        axes[j].imshow(img_disp, cmap="gray")
        axes[j].set_title(f"pred: {pred}")
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig("plot-predictions.png", dpi=150)
    plt.close()
    print("Saved plot-predictions.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
