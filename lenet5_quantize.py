import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import numpy as np
import tqdm
import os
import torch.nn as nn
import math
from collections import OrderedDict
from torchvision.datasets import MNIST

from models.lenet5 import LeNet5ReLU1
from parch import *

test_dataset = MNIST(root='./data/', train=False, download=True, transform=transforms.ToTensor())
loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


for bits in range(4,15):

    net = LeNet5ReLU1()
    net.load_state_dict(torch.load('./weights/mnist_relu1.pkl'))

    # idata_sf = set_sf(net, bits, loader_test, device)
    net = change_model_with_quant(net, bits)
    net.eval()

    num_correct, num_samples = 0, len(loader_test.dataset)
    for x, y in tqdm.tqdm(loader_test):
        # scores = net(to_fix_point(linear_quantize(x, idata_sf, bits), idata_sf,bits))
        scores = net(to_fix_point(x,bits))
        _, preds = scores.data.max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct) / num_samples

    with open('lenet5_q_result.txt',mode="a") as f:
        f.write(str(bits)+"bit "+str(acc)+"\n")
    print(acc)

#0.9919

#0.9923 -> 0.9921
