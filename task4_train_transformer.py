## Dana Tran
## CS 5330 Project 5
## 04/06/26


import sys
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from NetTransformer_template import NetConfig, NetTransformer

#pick a device
def pick_device(config_device: str) -> torch.device:
    if config_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if config_device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

#mnist train + test loaders with same normalize as the CNN task
def make_loaders(batch_size: int, data_dir: str="mnist_data"):
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_set = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=tfm
    )
    test_set = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=tfm
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, test_loader

#one full training epoch
def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

#avg loss and accuracy for log_softmax + NLLLoss
def evaluate(model,loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)
            batch_loss = criterion(out, labels).item()
            total_loss += batch_loss * images.size(0)
            pred = out.argmax(dim=1)
            total_correct += (pred == labels).sum().item()
            total_samples += images.size(0)
    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc

# main function training nettransformer on mnist
def main(argv):
    config = NetConfig()
    torch.manual_seed(config.seed)
    device = pick_device(config.device)
    print("using device:", device)
    train_loader, test_loader = make_loaders(
        batch_size=config.batch_size, data_dir="mnist_data"
    )
    model = NetTransformer(config).to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    epoch_times = []
    for epoch in range(1, config.epochs + 1):
        t0 = time.perf_counter()
        train_one_epoch(model, train_loader, device, criterion, optimizer)
        epoch_times.append(time.perf_counter() - t0)
        tr_loss, tr_acc = evaluate(model, train_loader, device, criterion)
        te_loss, te_acc = evaluate(model, test_loader, device, criterion)
        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        train_accs.append(tr_acc)
        test_accs.append(te_acc)
        print(
            f"epoch {epoch}/{config.epochs}  "
            f"train loss {tr_loss:.4f}  train err {1-tr_acc:.4f}  "
            f"test loss {te_loss:.4f}  test err {1-te_acc:.4f}  "
            f"time {epoch_times[-1]:.1f}s"
        )
    ep = list(range(1, config.epochs + 1))
    plt.figure()
    plt.plot(ep, [1.0 - a for a in train_accs], label="train err", color="tab:blue")
    plt.plot(ep, [1.0 - a for a in test_accs], label="test err", color="tab:orange")
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.legend()
    plt.tight_layout()
    plt.savefig("transformer-train-error.png", dpi=150)
    plt.close()
    plt.figure()
    plt.plot(ep, train_accs, label="train acc", color="tab:blue")
    plt.plot(ep, test_accs, label="test acc", color="tab:orange")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("transformer-train-accuracy.png", dpi=150)
    plt.close()
    torch.save(model.state_dict(), "mnist_transformer.pt")
    print("saved mnist_transformer.pt")
    print(
        f"final test acc: {test_accs[-1]:.4f}  "
        f"mean epoch time: {sum(epoch_times)/len(epoch_times):.1f}s"
    )
    return 0


if __name__=="__main__":
    raise SystemExit(main(sys.argv))