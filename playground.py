import torch
import numpy as np
import os
import shutil
# shutil.make_archive(output_filename, 'zip', dir_name)

# get mnist dataset
from torchvision import datasets, transforms

mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
print(type(mnist.targets))
# get cifar10
cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

print(type(cifar10.targets))