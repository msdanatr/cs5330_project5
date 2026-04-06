## Dana Tran
## CS 5330 Project 5
## 04/06/26

import csv
import itertools
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#same conv geometry as MyNetwork fc1 width and dropout varies
class ExperimentCNN(nn.Module):
    def __init__(self, fc_hidden: int, dropout_p: float):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(320, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x =F.relu(x)
        x =self.fc2(x)
        x =self.log_softmax(x)
        return x


#mnist standard normalize  - diff from digits(?)
def fashion_loaders(batch_size: int, data_dir: str = "fashion_data"):
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    train_set = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=tfm
    )
    test_set = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=tfm
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers = 0
    )
    return train_loader, test_loader


def train_epochs(model, train_loader, optimizer, criterion, epochs, device):
    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()


#avg loss + accuracy on a loader
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)
            total_loss += criterion(out, labels).item() * images.size(0)
            pred = out.argmax(dim=1)
            total_correct += (pred == labels).sum().item()
            total_samples += images.size(0)
    return total_loss / total_samples, total_correct / total_samples


#main function
def main(argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 3 dimensions, edit these lists later
    dropouts = [0.3, 0.5, 0.7]
    fc_hiddens = [32, 50, 80]
    batch_sizes = [32, 64, 128]
    epochs_list = [3, 5]

    #3 * 3 * 3 * 2 = 54 runs
    runs = list(itertools.product(dropouts, fc_hiddens, batch_sizes, epochs_list))
    print("total runs:", len(runs))

    csv_path = "experiment_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run_id",
                "dropout",
                "fc_hidden",
                "batch_size",
                "epochs",
                "test_acc",
                "test_loss",
                "seconds",
            ]
        )

        for run_id, (dr, fh, bs, ep) in enumerate(runs, start=1):
            train_loader, test_loader = fashion_loaders(bs)
            model = ExperimentCNN(fc_hidden=fh, dropout_p=dr).to(device)
            criterion = nn.NLLLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

            t0 = time.perf_counter()
            train_epochs(model, train_loader, optimizer, criterion, ep, device)
            elapsed = time.perf_counter() - t0

            _, test_acc = evaluate(model, test_loader, device, criterion)
            test_loss, _ = evaluate(model, test_loader, device, criterion)

            w.writerow(
                [run_id, dr, fh, bs, ep, f"{test_acc:.6f}", f"{test_loss:.6f}", f"{elapsed:.3f}"]
            )
            print(
                f"run {run_id}/{len(runs)}  dr={dr} fh={fh} bs={bs} ep={ep}  "
                f"test_acc={test_acc:.4f}  time={elapsed:.1f}s"
            )

    print("wrote", csv_path)
    return 0


if __name__=="__main__":
    raise SystemExit(main(sys.argv))