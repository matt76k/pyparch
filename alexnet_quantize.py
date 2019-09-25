import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, models, transforms
import numpy as np
import tqdm
import os
from collections import OrderedDict

from models.alexnet import AlexNetReLU1
from utils.correct import correct
from parch import *


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

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False, num_workers=4)


device = 'cuda' if torch.cuda.is_available() else 'cpu'



for bits in range(11, 22):

    qnet = AlexNetReLU1()
    qnet.load_state_dict(torch.load('./weights/cifar10_relu1.pkl'))
    qnet = change_model_with_quant(qnet, bits)

    qnet.eval()

    correct1 = 0
    n_passed = 0

    for (img, target) in tqdm.tqdm(testloader):
        img, target = img.to(device), target.to(device)

        qimg = to_fix_point(img, bits)

        out = qnet(qimg)
        bs = out.size(0)
        n_passed += bs

        corr = correct(out, target)
        correct1 += corr[0]
    with open('alexnet_q_result.txt',mode="a") as f:
        f.write(str(bits)+"bit "+str(correct1)+"\n")

    print("{} {}".format(bits, correct1 / len(testloader)), flush=True)
