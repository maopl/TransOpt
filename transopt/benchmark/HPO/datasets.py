# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset, ConcatDataset, Dataset
from torchvision.datasets import MNIST, ImageNet, CIFAR10, CIFAR100

from robustbench.data import load_cifar10c, load_cifar100c, load_imagenetc
from robustbench.utils import clean_accuracy
from robustbench.utils import load_model

from transopt.benchmark.HBOROB.algorithms import ERM

from robustbench.data import load_cifar10


ImageFile.LOAD_TRUNCATED_IMAGES = True



def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class Dataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 1            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class RobMNIST(Dataset):
    def __init__(self, root):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)

        original_images = original_dataset_tr.data

        original_labels = original_dataset_tr.targets

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]
        
        dataset_transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToPILImage(),
            transforms.ToTensor()])
        
        x = torch.zeros(len(original_images), 1, 28, 28)
        for i in range(len(original_images)):
            x[i] = dataset_transform(original_images[i])

        self.input_shape = (1, 28, 28)
        self.num_classes = 10
        self.datasets = TensorDataset(x, original_labels)
        


class RobCifar10(Dataset):
    def __init__(self, root):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = CIFAR10(root, train=True, download=True)
        original_dataset_te = CIFAR10(root, train=False, download=True)

        original_images = original_dataset_tr.data
        original_labels = torch.tensor(original_dataset_tr.targets)

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]
        
        dataset_transform = transforms.Compose([
            # transforms.Resize((32, 32)),
            transforms.ToPILImage(),
            transforms.ToTensor()])
    
        transformed_images = torch.stack([dataset_transform(img) for img in original_images])
        transformed_test_images = torch.stack([dataset_transform(img) for img in original_dataset_te.data])


        self.input_shape = (3, 32, 32)
        self.num_classes = 10
        self.datasets = TensorDataset(transformed_images, original_labels)
        self.corruptions = ['fog']
        x_test_attack, y_test_attack = load_cifar10c(n_examples=1000 ,corruptions=self.corruptions, severity=5)
        self.test_ds = TensorDataset(x_test_attack, y_test_attack)
        
        self.test = TensorDataset(transformed_test_images, torch.tensor(original_dataset_te.targets))


class RobCifar100(Dataset):
    def __init__(self, root):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        # 加载 CIFAR-100 训练数据集
        original_dataset_tr = CIFAR100(root, train=True, download=True)

        # 获取图像数据和标签
        original_images = original_dataset_tr.data  # shape: (N, 32, 32, 3)
        original_labels = torch.tensor(original_dataset_tr.targets)  # 转换为张量

        # 随机打乱数据
        shuffle = torch.randperm(len(original_images))
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        # 数据集转换操作
        dataset_transform = transforms.Compose([
            transforms.ToPILImage(),  # 将 numpy 数组转换为 PIL 图像
            transforms.ToTensor(),  # 将 PIL 图像转换为 Tensor 并缩放到 [0, 1]
        ])

        # 将图像数据转换为 Tensor 并应用转换操作
        transformed_images = torch.stack([dataset_transform(img) for img in original_images])

        # 初始化训练集
        self.input_shape = (3, 32, 32)
        self.num_classes = 100
        self.datasets = TensorDataset(transformed_images, original_labels)  # 使用转换后的图像数据
        
        self.corruptions = ['fog']
        x_test, y_test = load_cifar100c(n_examples=5000, corruptions=self.corruptions, severity=5)
        self.test = TensorDataset(x_test, y_test)
        

class RobImageNet(Dataset):
    def __init__(self, root):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')


        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = ImageNet(root=root, transform=augment_transform)

        self.input_shape = (3, 224, 224)
        self.num_classes = 1000
        self.corruptions = ['fog']
        x_test, y_test = load_imagenetc(n_examples=5000, corruptions=self.corruptions, severity=5)
        self.test = TensorDataset(x_test, y_test)
