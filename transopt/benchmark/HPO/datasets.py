# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import numpy as np
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

class RobCifar10(Dataset):
    def __init__(self, root=None, augment=False):
        super().__init__()
        if root is None:        
            user_home = os.path.expanduser('~')
            root = os.path.join(user_home, 'transopt_tmp/data')

        # Load original CIFAR-10 dataset
        original_dataset_tr = CIFAR10(root, train=True, download=True)
        original_dataset_te = CIFAR10(root, train=False, download=True)

        original_images = original_dataset_tr.data
        original_labels = torch.tensor(original_dataset_tr.targets)

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]
        
        dataset_transform = self.get_transform(augment)
    
        transformed_images = torch.stack([dataset_transform(img) for img in original_images])
        transformed_test_images = torch.stack([self.get_transform(False)(img) for img in original_dataset_te.data])

        self.input_shape = (3, 32, 32)
        self.num_classes = 10
        self.datasets = TensorDataset(transformed_images, original_labels)
        
        # Prepare test sets
        self.test_sets = {}
        
        # Standard test set
        self.test_sets['standard'] = TensorDataset(transformed_test_images, torch.tensor(original_dataset_te.targets))
        
        # Corruption test sets
        corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ]
        for corruption in corruptions:
            x_test_corrupt, y_test_corrupt = load_cifar10c(n_examples=10000, corruptions=[corruption], severity=5, data_dir=root)
            self.test_sets[f'corruption_{corruption}'] = TensorDataset(x_test_corrupt, y_test_corrupt)

        # Load CIFAR-10.1 dataset
        cifar101_path = os.path.join(root, 'cifar10.1_v6_data.npy')
        cifar101_labels_path = os.path.join(root, 'cifar10.1_v6_labels.npy')
        if os.path.exists(cifar101_path) and os.path.exists(cifar101_labels_path):
            cifar101_data = np.load(cifar101_path)
            cifar101_labels = np.load(cifar101_labels_path)
            cifar101_data = torch.from_numpy(cifar101_data).float() / 255.0
            cifar101_labels = torch.from_numpy(cifar101_labels).long()
            self.test_sets['cifar10.1'] = TensorDataset(cifar101_data, cifar101_labels)
        else:
            print("CIFAR-10.1 dataset not found. Please download it to the data directory.")

        # Load CIFAR-10.2 dataset
        cifar102_path = os.path.join(root, 'cifar102_test.npz')
        if os.path.exists(cifar102_path):
            cifar102_data = np.load(cifar102_path)
            cifar102_images = cifar102_data['images']
            cifar102_labels = cifar102_data['labels']
            cifar102_images = torch.from_numpy(cifar102_images).float() / 255.0
            cifar102_labels = torch.from_numpy(cifar102_labels).long()
            self.test_sets['cifar10.2'] = TensorDataset(cifar102_images, cifar102_labels)
        else:
            print("CIFAR-10.2 dataset not found. Please download it to the data directory.")

    def get_transform(self, augment):
        if augment:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    def get_test_set(self, name):
        """
        Get a specific test set by name.
        Available names: 'standard', 'corruption_<corruption_name>', 'cifar10.1', 'cifar10.2'
        """
        return self.test_sets.get(name, None)

    def get_all_test_sets(self):
        """
        Return all available test sets.
        """
        return self.test_sets

class RobCifar100(Dataset):
    def __init__(self, root, augment=False):
        super().__init__()
        if root is None:        
            user_home = os.path.expanduser('~')
            root = os.path.join(user_home, 'transopt_tmp/data')

        original_dataset_tr = CIFAR100(root, train=True, download=True)
        original_dataset_te = CIFAR100(root, train=False, download=True)

        original_images = original_dataset_tr.data
        original_labels = torch.tensor(original_dataset_tr.targets)

        shuffle = torch.randperm(len(original_images))
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        dataset_transform = self.get_transform(augment)

        transformed_images = torch.stack([dataset_transform(img) for img in original_images])

        self.input_shape = (3, 32, 32)
        self.num_classes = 100
        self.datasets = TensorDataset(transformed_images, original_labels)
        
        # Standard test set
        test_images = torch.tensor(original_dataset_te.data).float() / 255.0
        test_labels = torch.tensor(original_dataset_te.targets)
        self.test_sets = {'standard': TensorDataset(test_images, test_labels)}

        # Corruption test sets
        corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ]
        for corruption in corruptions:
            x_test, y_test = load_cifar100c(n_examples=10000, corruptions=[corruption], severity=5, data_dir=root)
            self.test_sets[f'corruption_{corruption}'] = TensorDataset(x_test, y_test)

    def get_transform(self, augment):
        if augment:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])

    def get_test_set(self, name):
        """
        Get a specific test set by name.
        Available names: 'standard', 'corruption_<corruption_name>'
        """
        return self.test_sets.get(name, None)

    def get_all_test_sets(self):
        """
        Return all available test sets.
        """
        return self.test_sets


class RobImageNet(Dataset):
    def __init__(self, root, augment=False):
        super().__init__()
        if root is None:        
            user_home = os.path.expanduser('~')
            root = os.path.join(user_home, 'transopt_tmp/data')

        transform = self.get_transform(augment)

        self.datasets = ImageNet(root=root, split='train', transform=transform)
        self.test_sets = {'standard': ImageNet(root=root, split='val', transform=self.get_transform(False))}

        self.input_shape = (3, 224, 224)
        self.num_classes = 1000

        # Corruption test sets
        corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ]
        for corruption in corruptions:
            x_test, y_test = load_imagenetc(n_examples=5000, corruptions=[corruption], severity=5, data_dir=root)
            self.test_sets[f'corruption_{corruption}'] = TensorDataset(x_test, y_test)

    def get_transform(self, augment):
        if augment:
            return transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def get_test_set(self, name):
        """
        Get a specific test set by name.
        Available names: 'standard', 'corruption_<corruption_name>'
        """
        return self.test_sets.get(name, None)

    def get_all_test_sets(self):
        """
        Return all available test sets.
        """
        return self.test_sets


def test_dataset(dataset_name='cifar10', num_samples=100):
    # Set up the dataset
    if dataset_name.lower() == 'cifar10':
        dataset = RobCifar10(root=None, augment=True)
    elif dataset_name.lower() == 'cifar100':
        dataset = RobCifar100(root=None, augment=True)
    elif dataset_name.lower() == 'imagenet':
        dataset = RobImageNet(root=None, augment=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Test training data
    assert len(dataset.datasets) > 0, "Training dataset is empty"
    print(f"Training dataset size: {len(dataset.datasets)}")
    print(f"Training data shape: {dataset.datasets[0][0].shape}")
    print(f"Training label shape: {dataset.datasets[0][1].shape}")

    # Test standard test set
    standard_test = dataset.get_test_set('standard')
    assert standard_test is not None, "Standard test set not found"
    print(f"Standard test set size: {len(standard_test)}")

    # Test corruption test sets
    corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    for corruption in corruptions[:num_samples]:
        corruption_test = dataset.get_test_set(f'corruption_{corruption}')
        assert corruption_test is not None, f"Corruption test set '{corruption}' not found"
        print(f"Corruption test set '{corruption}' size: {len(corruption_test)}")

    # Test additional test sets (CIFAR-10.1 and CIFAR-10.2 for CIFAR-10)
    if dataset_name.lower() == 'cifar10':
        for additional_test in ['cifar10.1', 'cifar10.2']:
            test_set = dataset.get_test_set(additional_test)
            if test_set is not None:
                print(f"{additional_test.upper()} test set size: {len(test_set)}")
            else:
                print(f"{additional_test.upper()} test set not found")

    print(f"All tests for {dataset_name} passed successfully!")

if __name__ == "__main__":
    # test_dataset('cifar10')
    # test_dataset('cifar100')
    test_dataset('imagenet')
