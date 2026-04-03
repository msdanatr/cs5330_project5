## Dana Tran
## CS 5330 Project 1
## 04/03/26

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))