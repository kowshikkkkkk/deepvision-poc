import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from typing import Tuple
from src.utils.augmentation import (
    get_mnist_train_transforms, get_mnist_val_transforms,
    get_cifar10_train_transforms, get_cifar10_val_transforms,
)

MNIST_CLASSES   = [str(i) for i in range(10)]
CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]

def get_mnist_loaders(data_dir="./data", batch_size=64, num_workers=0):
    train_set = datasets.MNIST(data_dir, train=True,  download=True, transform=get_mnist_train_transforms())
    test_set  = datasets.MNIST(data_dir, train=False, download=True, transform=get_mnist_val_transforms())
    val_size  = int(len(train_set) * 0.1)
    train_set, val_set = random_split(train_set, [len(train_set)-val_size, val_size],
                                      generator=torch.Generator().manual_seed(42))
    return (DataLoader(train_set, batch_size, shuffle=True,  num_workers=num_workers),
            DataLoader(val_set,   batch_size, shuffle=False, num_workers=num_workers),
            DataLoader(test_set,  batch_size, shuffle=False, num_workers=num_workers))

def get_cifar10_loaders(data_dir="./data", batch_size=64, num_workers=0):
    train_set = datasets.CIFAR10(data_dir, train=True,  download=True, transform=get_cifar10_train_transforms())
    test_set  = datasets.CIFAR10(data_dir, train=False, download=True, transform=get_cifar10_val_transforms())
    val_size  = int(len(train_set) * 0.1)
    train_set, val_set = random_split(train_set, [len(train_set)-val_size, val_size],
                                      generator=torch.Generator().manual_seed(42))
    return (DataLoader(train_set, batch_size, shuffle=True,  num_workers=num_workers),
            DataLoader(val_set,   batch_size, shuffle=False, num_workers=num_workers),
            DataLoader(test_set,  batch_size, shuffle=False, num_workers=num_workers))