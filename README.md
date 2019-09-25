# Deep Learning Framework with Arbitrary Numerical Precision
pyparch is DNNs framework, that allows easy manipulation of quantiztion.

## Dependency
- Python3
- torch
- torchvision
- numpy
- tqdm

## Pretrain

```
python pre_lenet5_relu6.py
python pre_lenet5_relu1.py
```

### AlexNet(CIFAR-10)
```
python pre_alexnet_relu6.py
python pre_alexnet_relu1.py
```

## Quantization
### LeNet5(MNIST)

```
python lenet5_quantize.py
```

### AlexNet(CIFAR-10)

```
python alexnet_quantize.py
```

## Citation
If you use our code in your research, please cite:

@inproceedings{mcsoc2019kiyama,
  title={Deep Learning Framework with Arbitrary Numerical Precision},
  author={Masato KIYAMA and Motoki Amagasaki and Masahiro Iida},
  booktitle={MCSoC 2019},
  year={2019}
}