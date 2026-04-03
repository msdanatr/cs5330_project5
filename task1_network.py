## Dana Tran
## CS 5330 Project 1
## 04/03/26

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MyNetwork(nn.Module):

    def __init__(self):
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


def main(argv):
    model = MyNetwork()
    model.eval()
    batch = 2
    x = torch.randn(batch, 1, 28, 28)
    with torch.no_grad():
        y = model(x)
    print("input shape:", tuple(x.shape))
    print("output shape:", tuple(y.shape))
    assert y.shape == (batch, 10), "expected [N, 10] after log_softmax"
    
    loader =make_train_loader(batch_size=64)
    images, labels = next(iter(loader))
    print("1 MNIST batch images:", tuple(images.shape), "labels:", tuple(labels.shape))
    
    model=MyNetwork()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion =nn.NLLLoss()

    loss_val = train_one_batch(model,images, labels, optimizer, criterion)
    print("1 batch loss:", loss_val)
    
    return 0


def train_one_batch(model, images, labels, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(images)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))