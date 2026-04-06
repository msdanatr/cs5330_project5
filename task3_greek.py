## Dana Tran
## CS 5330 Project 5
## 04/06/26

import argparse
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.utils.data import DataLoader
from torchvision import transforms

from task1_network import MyNetwork


#greek data set transform
class GreekTransform:
    def __init__(self):
        pass
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


def build_frozen_greek_model(weights_path: str="mnist_cnn.pt"):
    model = MyNetwork()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    for p in model.parameters():
        p.requires_grad = False
    #replace last linear 10 classes -> 3 Greek letters
    model.fc2 = nn.Linear(50, 3)
    for p in model.fc2.parameters():
        p.requires_grad = True
    return model


def make_greek_loader(training_set_path: str, batch_size: int = 5):
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            GreekTransform(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    ds = torchvision.datasets.ImageFolder(root=training_set_path, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=True), ds


def train_greek(model, loader, epochs: int, lr: float):
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.fc2.parameters(), lr=lr, momentum=0.5)
    losses = []
    errors = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += images.size(0)
        avg_loss = running_loss / total
        err = 1.0 - (correct / total)
        losses.append(avg_loss)
        errors.append(err)
        print(f"epoch {epoch}/{epochs}  loss {avg_loss:.4f}  train err {err:.4f}")
    return losses, errors


    
def main(argv):
    parser = argparse.ArgumentParser(description="Greek letter transfer learning")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to folder containing alpha/, beta/, gamma/ subfolders",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args(argv[1:])
    model = build_frozen_greek_model()
    print(model)
    print("class order:", torchvision.datasets.ImageFolder(root=args.data).classes)
    loader, ds = make_greek_loader(args.data, batch_size=args.batch_size)
    losses, errors = train_greek(model, loader, epochs=args.epochs, lr=args.lr)
    ep = list(range(1, len(errors) + 1))
    plt.figure()
    plt.plot(ep, errors, label="train error", color="tab:blue")
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.legend()
    plt.tight_layout()
    plt.savefig("greek-train-error.png", dpi=150)
    plt.close()
    print("saved greek-train-error.png")
    torch.save(model.state_dict(), "greek_cnn.pt")
    print("saved greek_cnn.pt")
    return 0


if __name__=="__main__":
    raise SystemExit(main(sys.argv))