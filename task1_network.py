## Dana Tran
## CS 5330 Project 5
## 04/03/26

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

class MyNetwork(nn.Module):

    def __init__(self): # set up conv / pool / dropout / fc layers for mnist
        super().__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.pool =nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.dropout=nn.Dropout(.5)
        self.fc1=nn.Linear(320,50)
        self.fc2=nn.Linear(50,10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self,x): #conv-> pool+ relu -> conv -> dropout -> pool+ relu -> dense -> log_softmax
        x= self.conv1(x)
        x= self.pool(x)
        x = F.relu(x)
        x =self.conv2(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x

#helper to build train dataloader
def make_train_loader(batch_size: int=64, data_dir: str="mnist_data"):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set=datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    return DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)


# def main(argv):
#     model = MyNetwork()
#     model.eval()
#     batch = 2
#     x = torch.randn(batch, 1, 28, 28)
#     with torch.no_grad():
#         y = model(x)
#     print("input shape:", tuple(x.shape))
#     print("output shape:", tuple(y.shape))
#     assert y.shape == (batch, 10), "expected [N, 10] after log_softmax"
    
#     loader =make_train_loader(batch_size=64)
#     images, labels = next(iter(loader))
#     print("1 MNIST batch images:", tuple(images.shape), "labels:", tuple(labels.shape))
    
#     model=MyNetwork()
#     optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
#     criterion =nn.NLLLoss()

#     loss_val = train_one_batch(model,images, labels, optimizer, criterion)
#     print("1 batch loss:", loss_val)
    
#     return 0

def main(argv): #train 5 epochs, plot curves, save mnist_cnn.pt
    batch_size = 64
    epochs = 5
    data_dir = "mnist_data"

    train_loader = make_train_loader(batch_size=batch_size, data_dir=data_dir)
    test_loader = make_test_loader(batch_size=batch_size, data_dir=data_dir)

    model = MyNetwork()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.NLLLoss()

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, optimizer, criterion)

        tr_loss, tr_acc = evaluate(model, train_loader, criterion)
        te_loss, te_acc = evaluate(model, test_loader, criterion)

        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        train_accs.append(tr_acc)
        test_accs.append(te_acc)

        tr_err = 1.0 - tr_acc
        te_err = 1.0 - te_acc
        print(
            f"epoch {epoch}/{epochs}  "
            f"train loss {tr_loss:.4f}  train err {tr_err:.4f}  "
            f"test loss {te_loss:.4f}  test err {te_err:.4f}"
        )

    #plots for the report (error = 1 - accuracy)
    epoch_ids = list(range(1, epochs + 1))
    plt.figure()
    plt.plot(epoch_ids, [1.0 - a for a in train_accs], label="train error", color="tab:blue")
    plt.plot(epoch_ids, [1.0 - a for a in test_accs], label="test error", color="tab:orange")
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot-training-error.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epoch_ids, train_accs, label="train acc", color="tab:blue")
    plt.plot(epoch_ids, test_accs, label="test acc", color="tab:orange")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot-training-accuracy.png", dpi=150)
    plt.close()

    torch.save(model.state_dict(), "mnist_cnn.pt")
    print("saved weights to mnist_cnn.pt")
    return 0


#one batch... loss, backward, optimizer step
def train_one_batch(model, images, labels, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(images)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

#test loader
def make_test_loader(batch_size: int=64, data_dir: str="mnist_data"):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    test_set=datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

#train pass over all batches
def train_one_epoch(model, train_loader, optimizer, criterion):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()


#evaluate
def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in data_loader:
            out = model(images)
            batch_loss = criterion(out, labels).item()
            total_loss += batch_loss * images.size(0)
            pred = out.argmax(dim=1)
            total_correct += (pred == labels).sum().item()
            total_samples += images.size(0)
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))