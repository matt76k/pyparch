import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, models, transforms
import numpy as np
import tqdm
import os

from utils.correct import correct
from models.alexnet import AlexNetReLU6

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = AlexNetC()

net = net.to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.1)

epochs = 200

bestnet = 0

for e in range(epochs):
    print("epoch:", e)

    net.train()
    scheduler.step(e)

    for (img, target) in tqdm.tqdm(trainloader):
        img, target = img.to(device), target.to(device)

        out = net(img)
        loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    net.eval()
    correct1, correct5 = 0, 0
    n_passed = 0

    for (img, target) in testloader:
        img, target = img.to(device), target.to(device)

        out = net(img)
        bs = out.size(0)
        n_passed += bs

        corr = correct(out, target, topk=(1, 5))
        correct1 += corr[0]
        correct5 += corr[1]

    if correct1 > bestnet:
        bestnet = correct1
        if not os.path.exists('./weights'):
             os.mkdir('./weights')
        torch.save(net.state_dict(), './weights/cifar10_relu6.pkl')

    print("{}, {}".format(e, correct1 / len(testloader)))
print("best accuracy:",bestnet / len(testloader))
