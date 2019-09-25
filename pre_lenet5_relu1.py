import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import tqdm

from models.lenet5 import LeNet5ReLU1

train_dataset = MNIST(root='./data/',train=True, download=True, transform=transforms.ToTensor())
loader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = MNIST(root='./data/', train=False, download=True, transform=transforms.ToTensor())
loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)


net = LeNet5ReLU1()
net.load_state_dict(torch.load('./weights/mnist_relu6.pkl'))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), 0.01, momentum=0.9)

best = 0

for epoch in range(30):
    print("epoch: ", epoch)
    net.train()
    for (x, y) in tqdm.tqdm(loader_train):
        scores = net(x)
        loss = criterion(scores, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    net.eval()
    num_correct, num_samples = 0, len(loader_test.dataset)
    for x, y in loader_test:
        scores = net(x)
        _, preds = scores.data.max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct) / num_samples

    if best < acc:
        best = acc
        net.cpu()
        if not os.path.exists('./weights'):
             os.mkdir('./weights')
        torch.save(net.state_dict(), './weights/mnist_relu1.pkl')

    print("",end="")#表示が若干変になるので入れた
    print(acc)

print("best accuracy:",best)
