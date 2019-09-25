import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import OrderedDict

def to_fix_point(input, bits):
    fixed_input = torch.round(input * math.pow(2.0, bits - 1))
    fixed_input = fixed_input.type(torch.LongTensor)

    return fixed_input

def from_fix_to_float(input, bits):
    return input.type(torch.FloatTensor)/math.pow(2, bits - 1)

def cast_output(x, s_i, s_o, bits):
    bound = 2 ** (bits - 1)


    min_val = - bound
    max_val = bound - 1

    if s_i > s_o:
        x += 2**(s_i - s_o - 1)
        a = x.numpy() >> (s_i - s_o)
    else:
        a = x.numpy() << (s_o - s_i)

    return torch.clamp(torch.from_numpy(a), min_val, max_val).type(torch.LongTensor)

def check_overflow_bits(n, bits):
    m = math.floor(math.fabs(n) / math.pow(2.0, bits - 1))

    if m == 0:
        return 0
    else:
        return round(math.log2(m) + 1)

def linear_quantize(input, sf, bits):
    delta = math.pow(2.0, -sf)
    bound = math.pow(2.0, bits - 1)
    min_val = - bound
    max_val = bound - 1
    rounded = torch.floor(input / delta + 0.5)

    clipped_value = torch.clamp(rounded, min_val, max_val) * delta

    return clipped_value

def change_model_with_quant(model, bits):
    if isinstance(model, nn.Sequential):
        l = OrderedDict()
        for k, v in model._modules.items():
            if isinstance(v, nn.Conv2d):
                l[k] = QConv2d(v.weight, v.bias, stride=v.stride[0], padding=v.padding[0], bits=bits)
            elif isinstance(v, nn.Linear):
                l[k] = QLinear(v.weight, v.bias, bits=bits)
            elif isinstance(v, (nn.MaxPool2d)):
                l[k] = QMaxPool2dP(v.kernel_size, bits=bits)
            elif isinstance(v, (ReLU1)):
                l[k] = QReLU1(bits)
            else:
                l[k] = v
        m = nn.Sequential(l)
        return m
    else:
        for k, v in model._modules.items():
            model._modules[k] = change_model_with_quant(v, bits)
        return model
        
def set_sf(net, bits, loader_test, device):

    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            w_max, b_max = m.weight.abs().view(-1).max().item(), m.bias.abs().view(-1).max().item()
            max_v = max(w_max, b_max)
            sf = bits -1 - math.ceil(math.log2(max_v+1e-12))
            m.weight.data = linear_quantize(m.weight.data, sf, bits)
            m.bias.data = linear_quantize(m.bias.data, sf, bits)
            m.wsf = sf
            m.layer_outputs = None
            m.register_forward_hook(lambda m, i, o: exec('m.layer_outputs = o'))
            m.layer_max = float("-inf")

    net = net.to(device)
    net.eval()

    idata_max = float('-inf')

    for x, y in loader_test:
        idata_max = max(idata_max, x.abs().view(-1).max().item())
        net(x.to(device))
        for m in net.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                o_max = m.layer_outputs.abs().view(-1).max().item()
                m.layer_max = max(m.layer_max, o_max)

    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.osf = bits - 1 - math.ceil(math.log2(m.layer_max+1e-12))

    idata_sf = bits -1 - math.ceil(math.log2(idata_max+1e-12))

    sf = idata_sf
    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.isf = sf
            sf = m.osf
        elif isinstance(m, nn.MaxPool2d):
            m.isf = sf
            m.wsf = sf
            m.osf = sf

    return idata_sf

class ReLU1(nn.Hardtanh):
    def __init__(self, inplace=False):
        super().__init__(0, 1, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + '(' \
            + inplace_str + ')'

class QReLU1(nn.Module):
    def __init__(self, bits=8):
        super().__init__()
        self.max_val = math.pow(2, bits - 1)

    def forward(self, x):
        return torch.clamp(x, 0, self.max_val)

class QReLU(nn.Module):
    def forward(self, x):
        return torch.where(x > 0, x, torch.zeros(x.shape, dtype=x.dtype))

class QMaxPool2dP(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, bits=8):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bits = bits

    def forward(self, x):
        x = from_fix_to_float(x, self.bits)
        out = F.max_pool2d(x, self.kernel_size, self.stride,
                            self.padding)

        return to_fix_point(out, self.bits)

    def extra_repr(self):
        return 'kernel_size={}, stride={}, padding={} sf=({}, {}, {})'.format(self.kernel_size, self.stride, self.padding, self.isf, None, self.osf)

class QLinear(nn.Module):
    def __init__(self, W, b, bits=8):
        super().__init__()
        self.weight = to_fix_point(W, bits)

        if b is None:
            self.bias = torch.zeros(W.shape[0], dtype=W.dtype)
        else:
            self.bias = to_fix_point(b, bits)

        self.out_features = self.weight.shape[0]
        self.bits = bits

    def forward(self, x):
        C, R = x.shape


        out = torch.zeros(C, self.out_features, dtype=x.dtype)

        for i in range(C):
            for j in range(self.out_features):
                m = self.weight[j] * x[i]
                converted_m = m / math.pow(2.0, self.bits - 1)
                out[i, j] = torch.sum(converted_m) + self.bias[j]

        return out

class QConv2d(nn.Module):
    def __init__(self, W, b, stride=1, padding=0, bits=8):
        super().__init__()
        self.weight = to_fix_point(W, bits)

        if b is None:
            self.bias = torch.zeros(W.shape[0], dtype=W.dtype)
        else:
            self.bias = to_fix_point(b, bits)

        self.stride = stride
        self.padding = padding
        self.bits = bits

        self.overflow_bits = None
        self.overflow_bits_r = 0



    def forward(self, x):
        # フィルターと出力の形状のセットアップ
        FN, FC, FH, FW = self.weight.shape
        N, C, H, W = x.shape
        OH = (H - FH) // self.stride + 1
        OW = (W - FW) // self.stride + 1

        padding = nn.ConstantPad2d(self.padding, 0.0)

        out = []

        for nx in range(N):
            f_out = []
            for nf in range(FN):
                f_out.append(self.calConv(padding(x[nx]).type(x.dtype), self.weight[nf], self.bias[nf], self.stride))

            out.append(torch.stack(f_out))

        output = torch.stack(out)

        return output

    def calConv(self, input, filter, bias, stride):
        C, H, W = input.shape
        FC, FH, FW = filter.shape

        self.overflow_bits = round(math.log2(FH * FW) + 1)


        OH = (H - FH) // stride + 1
        OW = (W - FW) // stride + 1

        out = torch.zeros(OH, OW, dtype=input.dtype)

        for j in range(OH):
            for i in range(OW):
                ni = stride * i
                nj = stride * j
                m = input[:, ni:ni + FW, nj:nj + FH] * filter

                #mp = torch.where(m > 0, m, torch.zeros(m.shape, dtype=m.dtype)).sum()
                #mn = torch.where(m < 0, m, torch.zeros(m.shape, dtype=m.dtype)).sum()

                #self.overflow_bits_r = max(self.overflow_bits_r, check_overflow_bits(mp, self.bits * 2 - 2), check_overflow_bits(mn, self.bits * 2 - 2))

                converted_m = m / math.pow(2.0, self.bits - 1)

                out[i, j] = torch.sum(converted_m) + bias

        return out

class QMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, bits=8):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
         N, C, H, W = x.shape
         out = []

         for nx in range(N):
            f_out = []
            for nf in range(C):
                f_out.append(self.cal_max(x[nx, nf]))

            out.append(torch.stack(f_out))

         return torch.stack(out)

    def cal_max(self, mat):

        H, W = mat.shape

        OH = H // self.kernel_size
        OW = W // self.kernel_size

        out = torch.zeros(OH, OW, dtype=mat.dtype)

        for j in range(OH):
            for i in range(OW):
                ni = self.kernel_size * i
                nj = self.kernel_size * j
                out[i, j] = mat[ni:ni+self.kernel_size, nj:nj+self.kernel_size].max()

        return out
