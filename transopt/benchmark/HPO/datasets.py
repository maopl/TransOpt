# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import TensorDataset, Subset, ConcatDataset, Dataset
from torchvision.datasets import MNIST, ImageNet, CIFAR10, CIFAR100
from torchvision.transforms.functional import rotate

from torchvision.utils import make_grid

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision


from robustbench.data import load_cifar10c, load_cifar100c, load_imagenetc

from transopt.benchmark.HPO.augmentation import ImageNetPolicy, CIFAR10Policy, CIFAR10PolicyGeometric, CIFAR10PolicyPhotometric, Cutout, SamplerPolicy

ImageFile.LOAD_TRUNCATED_IMAGES = True



def data_transform(dataset_name, augmentation_name=None):
    if dataset_name.lower() == 'cifar10' or dataset_name.lower() == 'cifar100':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        size = 32
    elif dataset_name.lower() == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        size = 224
    elif dataset_name.lower() == 'mnist':
        mean = (0.1307,)
        std = (0.3081,)
        size = 28
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # transform_list = [transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean, std)]
    transform_list = [transforms.ToPILImage(), transforms.ToTensor()]

    if augmentation_name:
        if dataset_name.lower() in ['cifar10', 'cifar100']:
            if augmentation_name.lower() == 'cutout':
                transform_list.insert(-1,Cutout(n_holes=1, length=4))
            elif augmentation_name.lower() == 'geometric':
                transform_list.insert(1, CIFAR10PolicyGeometric())
            elif augmentation_name.lower() == 'photometric':
                transform_list.insert(1, CIFAR10PolicyPhotometric())
            elif augmentation_name.lower() == 'autoaugment':
                transform_list.insert(1, CIFAR10Policy())
            elif augmentation_name.lower() == 'testaugment':
                transform_list = [transforms.ToPILImage(), transforms.ToTensor()]
                print("Testaugment should be applied during training, not as part of the transform.")
            elif augmentation_name.lower() == 'ddpm':
                print("DDPM use generated data as training data")
            elif augmentation_name.lower() == 'augmix':
                print("Augmix should be applied during training, not as part of the transform.")
            elif augmentation_name.lower() == 'mixup':
                print("Mixup should be applied during training, not as part of the transform.")
            elif augmentation_name.lower() == 'paraaug':
                print("Parameterized augmentation should be applied during training, not as part of the transform.")
            else:
                raise ValueError(f"Unsupported augmentation strategy for CIFAR: {augmentation_name}")
        elif dataset_name.lower() == 'imagenet':
            if augmentation_name.lower() == 'cutout':
                transform_list.append(Cutout())
            elif augmentation_name.lower() == 'autoaugment':
                transform_list.insert(0, ImageNetPolicy())
            elif augmentation_name.lower() == 'mixup':
                print("Mixup should be applied during training, not as part of the transform.")
            else:
                raise ValueError(f"Unsupported augmentation strategy for ImageNet: {augmentation_name}")
        else:
            raise ValueError(f"Unsupported dataset for augmentation: {dataset_name}")
    print(transform_list)
    return transforms.Compose(transform_list)

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
class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 1            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __init__(self):
        self.datasets = {}

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)
    
class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, shift_function, input_shape,
                 num_classes, augment=None):
        super().__init__()
        if root is None:
            user_home = os.path.expanduser('~')
            root = os.path.join(user_home, 'transopt_tmp/data')
        
        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        train_images = original_dataset_tr.data
        train_labels = original_dataset_tr.targets
        test_images = original_dataset_te.data
        test_labels = original_dataset_te.targets

        train_shifted_images, train_shifted_labels = shift_function(train_images, train_labels, environments[0])
        test_standard_images, test_standard_labels = shift_function(test_images, test_labels, environments[0])
        test_shift_images, test_shift_labels = shift_function(test_images, test_labels, environments[1])

        indices = torch.randperm(len(train_shifted_labels))
        val_size = len(indices) // 10  # 10% 作為驗證集
        
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        train_data = train_shifted_images[train_indices]
        train_labels = train_shifted_labels[train_indices]
        val_data = train_shifted_images[val_indices]
        val_labels = train_shifted_labels[val_indices]

        self.dataset_transform = data_transform('mnist', augment)
        self.normalized_images = data_transform('mnist', None) 

        def transform_dataset(data, transform):
            return torch.stack([transform(img) for img in data])

        train_transformed = transform_dataset(train_data, self.dataset_transform)
        val_transformed = transform_dataset(val_data, self.normalized_images)
        test_standard_transformed = transform_dataset(test_standard_images, self.normalized_images)
        test_shift_transformed = transform_dataset(test_shift_images, self.normalized_images)

        self.input_shape = input_shape
        self.num_classes = num_classes
        
        self.datasets['train'] = TensorDataset(train_transformed, train_labels)
        self.datasets['val'] = TensorDataset(val_transformed, val_labels)
        self.datasets['test_standard'] = TensorDataset(test_standard_transformed, test_standard_labels)
        self.datasets['test_shift'] = TensorDataset(test_shift_transformed, test_shift_labels)

class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    INPUT_SHAPE = (3, 28, 28)

    def __init__(self, root, augment):
        super().__init__(root, [[0.45, 0.45, 0.1], [0.2,0.2,0.6]],
                        self.color_dataset, self.INPUT_SHAPE, 10, augment)

    def color_dataset(self, images, labels, environment):
        # Generate random colors based on Bernoulli probability
        # Generate random numbers between 0 and 1
        rand_nums = torch.rand(len(labels))
        
        # Initialize colors tensor
        colors = torch.zeros(len(labels))
        
        # Assign colors (0,1,2) based on environment probabilities
        # environment is now a list of 3 probabilities that sum to 1
        colors[rand_nums < environment[0]] = 0
        colors[(rand_nums >= environment[0]) & (rand_nums < environment[0] + environment[1])] = 1
        colors[rand_nums >= environment[0] + environment[1]] = 2
        
        # Stack the image into 3 channels
        images = torch.stack([images, images, images], dim=1)
        
        # Zero out channels based on generated colors
        images[torch.tensor(range(len(images))), colors.long(), :, :] *= 0

        # Normalize to [0,1] range
        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return x, y

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()
    
    def get_available_test_set_names(self):
        """
        Return a list of available test set names.
        """
        return list(self.datasets.keys())


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

class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']
    INPUT_SHAPE = (1, 28, 28)

    def __init__(self, root, augment):
        super().__init__(root, [0, 75],
                        self.rotate_dataset, self.INPUT_SHAPE, 10, augment)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return (x, y)


class RobCifar10(Dataset):
    def __init__(self, root=None, augment=False):
        super().__init__()
        if root is None:        
            user_home = os.path.expanduser('~')
            root = os.path.join(user_home, 'transopt_tmp/data')

        self.datasets = {}
        # Load original CIFAR-10 dataset
        original_dataset_tr = CIFAR10(root, train=True, download=True)
        original_dataset_te = CIFAR10(root, train=False, download=True)

        original_images = original_dataset_tr.data
        original_labels = torch.tensor(original_dataset_tr.targets)

        shuffle = torch.randperm(len(original_images))
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]
        
        
        self.dataset_transform = data_transform('cifar10', augment)
        self.normalized_images = data_transform('cifar10', None)
        
        transformed_images = torch.stack([self.dataset_transform(img) for img in original_images])
        # Split into train and validation sets
        val_size = len(transformed_images) // 10
        self.datasets['train'] = TensorDataset(transformed_images[:-val_size], original_labels[:-val_size])
        self.datasets['val'] = TensorDataset(transformed_images[-val_size:], original_labels[-val_size:])
        
        if augment == 'ddpm':
            ddpm_path = os.path.join(root, 'cifar10_ddpm.npz')
            ddpm_data = np.load(ddpm_path)
            ddpm_images = torch.from_numpy(ddpm_data['image']).float()
            ddpm_images = ddpm_images.permute(0, 3, 1, 2)  # Change from (N, 32, 32, 3) to (N, 3, 32, 32)
            ddpm_labels = torch.from_numpy(ddpm_data['label']).long()  # Convert to long tensor
            self.datasets['train'] = TensorDataset(ddpm_images, ddpm_labels)

        standard_test_images = torch.stack([self.normalized_images(img) for img in original_dataset_te.data])

        self.input_shape = (3, 32, 32)
        self.num_classes = 10
        # Standard test set
        self.datasets['test_standard'] = TensorDataset(standard_test_images, torch.tensor(original_dataset_te.targets))
        
        # Corruption test sets
        self.corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ]
        for corruption in self.corruptions:
            x_test_corrupt, y_test_corrupt = load_cifar10c(n_examples=5000, corruptions=[corruption], severity=5, data_dir=root)
            x_test_corrupt = torch.stack([self.normalized_images(img) for img in x_test_corrupt])
            self.datasets[f'test_corruption_{corruption}'] = TensorDataset(x_test_corrupt, y_test_corrupt)

        # Load CIFAR-10.1 dataset
        cifar101_path = os.path.join(root, 'cifar10.1_v6_data.npy')
        cifar101_labels_path = os.path.join(root, 'cifar10.1_v6_labels.npy')
        if os.path.exists(cifar101_path) and os.path.exists(cifar101_labels_path):
            cifar101_data = np.load(cifar101_path)
            cifar101_labels = np.load(cifar101_labels_path)
            cifar101_data = torch.from_numpy(cifar101_data).float() / 255.0
            cifar101_data = cifar101_data.permute(0, 3, 1, 2)  # Change from (N, 32, 32, 3) to (N, 3, 32, 32)
            cifar101_data = torch.stack([self.normalized_images(img) for img in cifar101_data])
            cifar101_labels = torch.from_numpy(cifar101_labels).long()
            self.datasets['test_cifar10.1'] = TensorDataset(cifar101_data, cifar101_labels)
        else:
            print("CIFAR-10.1 dataset not found. Please download it to the data directory.")

        # Load CIFAR-10.2 dataset
        cifar102_path = os.path.join(root, 'cifar102_test.npz')
        if os.path.exists(cifar102_path):
            cifar102_data = np.load(cifar102_path)
            cifar102_images = cifar102_data['images']
            cifar102_labels = cifar102_data['labels']
            cifar102_images = torch.from_numpy(cifar102_images).float() / 255.0
            cifar102_images = cifar102_images.permute(0, 3, 1, 2)  # Change from (N, 32, 32, 3) to (N, 3, 32, 32)
            cifar102_images = torch.stack([self.normalized_images(img) for img in cifar102_images])
            cifar102_labels = torch.from_numpy(cifar102_labels).long()
            self.datasets['test_cifar10.2'] = TensorDataset(cifar102_images, cifar102_labels)
        else:
            print("CIFAR-10.2 dataset not found. Please download it to the data directory.")

    def get_available_test_set_names(self):
        """
        Return a list of available test set names.
        """
        return list(self.datasets.keys())


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
    def __init__(self, root=None, augment=False):
        super().__init__()
        if root is None:        
            user_home = os.path.expanduser('~')
            root = os.path.join(user_home, 'transopt_tmp/data')

        self.datasets = {}
        # Load original CIFAR-100 dataset
        original_dataset_tr = CIFAR100(root, train=True, download=True)
        original_dataset_te = CIFAR100(root, train=False, download=True)

        original_images = original_dataset_tr.data
        original_labels = torch.tensor(original_dataset_tr.targets)

        shuffle = torch.randperm(len(original_images))
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]
        
        self.dataset_transform = data_transform('cifar100', augment)
        self.normalized_images = data_transform('cifar100', None)
        
        transformed_images = torch.stack([self.dataset_transform(img) for img in original_images])
        # Split into train and validation sets
        val_size = len(transformed_images) // 10
        self.datasets['train'] = TensorDataset(transformed_images[:-val_size], original_labels[:-val_size])
        self.datasets['val'] = TensorDataset(transformed_images[-val_size:], original_labels[-val_size:])

        if augment == 'ddpm':
            ddpm_path = os.path.join(root, 'cifar100_ddpm.npz')
            ddpm_data = np.load(ddpm_path)
            ddpm_images = torch.from_numpy(ddpm_data['image']).float()
            ddpm_images = ddpm_images.permute(0, 3, 1, 2)  # Change from (N, 32, 32, 3) to (N, 3, 32, 32)
            ddpm_labels = torch.from_numpy(ddpm_data['label']).long()  # Convert to long tensor
            self.datasets['train'] = TensorDataset(ddpm_images, ddpm_labels)

        standard_test_images = torch.stack([self.normalized_images(img) for img in original_dataset_te.data])

        self.input_shape = (3, 32, 32)
        self.num_classes = 100
        # Standard test set
        self.datasets['test_standard'] = TensorDataset(standard_test_images, torch.tensor(original_dataset_te.targets))
        
        # Corruption test sets
        self.corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ]
        for corruption in self.corruptions:
            x_test_corrupt, y_test_corrupt = load_cifar100c(n_examples=5000, corruptions=[corruption], severity=5, data_dir=root)
            x_test_corrupt = torch.stack([self.normalized_images(img) for img in x_test_corrupt])
            self.datasets[f'test_corruption_{corruption}'] = TensorDataset(x_test_corrupt, y_test_corrupt)

    def get_available_test_set_names(self):
        """
        Return a list of available test set names.
        """
        return list(self.datasets.keys())

    def get_test_set(self, name):
        """
        Get a specific test set by name.
        Available names: 'standard', 'corruption_<corruption_name>'
        """
        return self.datasets.get(name, None)

    def get_all_test_sets(self):
        """
        Return all available test sets.
        """
        return self.datasets

class RobImageNet(Dataset):
    def __init__(self, root, augment=False):
        super().__init__()
        if root is None:
            user_home = os.path.expanduser('~')
            root = os.path.join(user_home, 'transopt_tmp/data')

        self.datasets = {}
        self.corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ]

        self.datasets = {}
        original_dataset_tr = ImageNet(root, split='train', download=True)
        original_dataset_te = ImageNet(root, split='test', download=True)

        original_images = original_dataset_tr.data
        original_labels = torch.tensor(original_dataset_tr.targets)

        shuffle = torch.randperm(len(original_images))
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]
        
        self.dataset_transform = data_transform('imagenet', augment)
        self.normalized_images = data_transform('imagenet', None)
        
        transformed_images = torch.stack([self.dataset_transform(img) for img in original_images])
        # Split into train and validation sets
        val_size = len(transformed_images) // 10
        self.datasets['train'] = TensorDataset(transformed_images[:-val_size], original_labels[:-val_size])
        self.datasets['val'] = TensorDataset(transformed_images[-val_size:], original_labels[-val_size:])

        standard_test_images = torch.stack([self.normalized_images(img) for img in original_dataset_te.data])

        self.input_shape = (3, 224, 224)
        self.num_classes = 1000
        # Standard test set
        self.datasets['test_standard'] = TensorDataset(standard_test_images, torch.tensor(original_dataset_te.targets))
        
        # Corruption test sets
        self.corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ]
        for corruption in self.corruptions:
            x_test_corrupt, y_test_corrupt = load_imagenetc(n_examples=5000, corruptions=[corruption], severity=5, data_dir=root)
            x_test_corrupt = torch.stack([self.normalized_images(img) for img in x_test_corrupt])
            self.datasets[f'test_corruption_{corruption}'] = TensorDataset(x_test_corrupt, y_test_corrupt)

        # Add ImageNet-A dataset
        imagenet_a_path = os.path.join(root, 'imagenet-a')
        if os.path.exists(imagenet_a_path):
            imagenet_a_dataset = ImageNet(imagenet_a_path, split='test')
            imagenet_a_images = torch.stack([self.normalized_images(img) for img in imagenet_a_dataset.data])
            self.datasets['test_imageneta'] = TensorDataset(imagenet_a_images, 
                                                           torch.tensor(imagenet_a_dataset.targets))
        else:
            print("ImageNet-A dataset not found. Please download it to the data directory.")

        # Add ImageNet-O dataset
        imagenet_o_path = os.path.join(root, 'imagenet-o')
        if os.path.exists(imagenet_o_path):
            imagenet_o_dataset = ImageNet(imagenet_o_path, split='test')
            imagenet_o_images = torch.stack([self.normalized_images(img) for img in imagenet_o_dataset.data])
            self.datasets['test_imageneto'] = TensorDataset(imagenet_o_images, 
                                                           torch.tensor(imagenet_o_dataset.targets))
        else:
            print("ImageNet-O dataset not found. Please download it to the data directory.")

        # Add ImageNet-R dataset
        imagenet_r_path = os.path.join(root, 'imagenet-r')
        if os.path.exists(imagenet_r_path):
            imagenet_r_dataset = ImageNet(imagenet_r_path, split='test')
            imagenet_r_images = torch.stack([self.normalized_images(img) for img in imagenet_r_dataset.data])
            self.datasets['test_imagenetr'] = TensorDataset(imagenet_r_images, 
                                                           torch.tensor(imagenet_r_dataset.targets))
        else:
            print("ImageNet-R dataset not found. Please download it to the data directory.")


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

def test_dataset(dataset_name='cifar10', num_samples=5):
    # Set up the dataset
    if dataset_name.lower() == 'cifar10':
        dataset = RobCifar10(root=None, augment=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Test training data
    assert 'train' in dataset.datasets, "Training dataset is missing"
    print(f"Training dataset size: {len(dataset.datasets['train'])}")
    train_sample = dataset.datasets['train'][0]
    print(f"Training data shape: {train_sample[0].shape}")
    print(f"Training label shape: {train_sample[1].shape}")

    # Test validation data
    assert 'val' in dataset.datasets, "Validation dataset is missing"
    print(f"Validation dataset size: {len(dataset.datasets['val'])}")

    # Test standard test set
    assert 'test_standard' in dataset.datasets, "Standard test set is missing"
    print(f"Standard test set size: {len(dataset.datasets['test_standard'])}")

    # Test corruption test sets
    for corruption in dataset.corruptions[:num_samples]:
        corruption_key = f'test_corruption_{corruption}'
        assert corruption_key in dataset.datasets, f"Corruption test set '{corruption}' is missing"
        print(f"Corruption test set '{corruption}' size: {len(dataset.datasets[corruption_key])}")

    # Test additional test sets (CIFAR-10.1 and CIFAR-10.2)
    for additional_test in ['test_cifar10.1', 'test_cifar10.2']:
        if additional_test in dataset.datasets:
            print(f"{additional_test.upper()} test set size: {len(dataset.datasets[additional_test])}")
        else:
            print(f"{additional_test.upper()} test set not found")

    # Test data loading
    print("\nTesting data loading:")
    for key, data in dataset.datasets.items():
        try:
            sample = data[0]
            print(f"Successfully loaded sample from {key}")
            if isinstance(sample, tuple):
                print(f"  Sample shape: {sample[0].shape}, Label: {sample[1]}")
            else:
                print(f"  Sample shape: {sample.shape}")
        except Exception as e:
            print(f"Error loading data from {key}: {str(e)}")

    print(f"\nAll tests for {dataset_name} passed successfully!")
    
def visualize_dataset_tsne(dataset_name='cifar10', n_samples=1000, perplexity=30, n_iter=1000):
    # Set up data transformation
    # non_augment = data_transform(dataset_name, augmentation_name=None)
    # augment = data_transform(dataset_name, augmentation_name='photometric')

    # Load dataset
    if dataset_name.lower() == 'cifar10':
        dataset = RobCifar10(root=None, augment='ddpm')
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Prepare data for t-SNE
    all_images = []
    all_labels = []
    dataset_types = []

    for key, data in dataset.datasets.items():
        loader = DataLoader(data, batch_size=n_samples, shuffle=True)
        images, labels = next(iter(loader))

        if key == 'train':
            origin_images = torch.stack([non_augment(img) for img in images])
            all_images.append(origin_images)
            all_labels.append(labels)
            dataset_types.extend(['train_without_aug'] * len(origin_images))
            
            augmented_images = torch.stack([augment(img) for img in images])
            all_images.append(augmented_images)
            all_labels.append(labels)
            dataset_types.extend(['augmented'] * len(augmented_images))
            continue
        
        if key.startswith('test_') and key != 'test_standard':
            all_images.append(images)
            all_labels.append(labels)
            dataset_types.extend(['test_ds'] * len(images))
        # else:
        #     all_images.append(images)
        #     all_labels.append(labels)
        #     dataset_types.extend([key] * len(images))

    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_images_flat = all_images.view(all_images.size(0), -1).numpy()

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    tsne_results = tsne.fit_transform(all_images_flat)

    # Visualize results
    plt.figure(figsize=(16, 12))
    
    # Define a fixed color map
    fixed_color_map = {
        'train_without_aug': '#1f77b4',  # blue
        'augmented': '#ff7f0e',          # orange
        'val': '#2ca02c',                # green
        'test_standard': '#d62728',      # red
        'test_ds': '#9467bd',            # purple
        'test_cifar10.1': '#8c564b',     # brown
        'test_cifar10.2': '#e377c2'      # pink
    }
    
    for dtype in fixed_color_map.keys():
        mask = np.array(dataset_types) == dtype
        if np.any(mask):  # Only plot if there are data points for this type
            plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], 
                        c=fixed_color_map[dtype], label=dtype, alpha=0.6)

    plt.legend()
    plt.title(f't-SNE visualization of {dataset_name} dataset')
    plt.savefig(f'{dataset_name}_tsne_visualization.png')
    plt.close()

    print(f"t-SNE visualization has been saved as '{dataset_name}_tsne_visualization.png'")

def visualize_mnist_data():
    # 初始化數據集
    mnist_data = ColoredMNIST(
        root=None,
        augment=None
    )

    train_loader = torch.utils.data.DataLoader(mnist_data.datasets['train'], batch_size=16)
    val_loader = torch.utils.data.DataLoader(mnist_data.datasets['val'], batch_size=16)
    test_standard_loader = torch.utils.data.DataLoader(mnist_data.datasets['test_standard'], batch_size=16)
    test_shift_loader = torch.utils.data.DataLoader(mnist_data.datasets['test_shift'], batch_size=16)

    # 獲取第一個batch的數據
    train_samples, train_labels = next(iter(train_loader))
    val_samples, val_labels = next(iter(val_loader))
    test_standard_samples, test_standard_labels = next(iter(test_standard_loader))
    test_shift_samples, test_shift_labels = next(iter(test_shift_loader))

    # 創建圖形
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Colored MNIST Data Visualization', fontsize=16)

    # 顯示訓練集圖片
    grid_img = make_grid(train_samples, nrow=4, normalize=True)
    axes[0, 0].imshow(grid_img.permute(1, 2, 0))
    axes[0, 0].set_title(f'Training Data\nLabels: {train_labels.numpy()}')
    axes[0, 0].axis('off')

    # 顯示驗證集圖片
    grid_img = make_grid(val_samples, nrow=4, normalize=True)
    axes[0, 1].imshow(grid_img.permute(1, 2, 0))
    axes[0, 1].set_title(f'Validation Data\nLabels: {val_labels.numpy()}')
    axes[0, 1].axis('off')

    # 顯示標準測試集圖片
    grid_img = make_grid(test_standard_samples, nrow=4, normalize=True)
    axes[1, 0].imshow(grid_img.permute(1, 2, 0))
    axes[1, 0].set_title(f'Test Standard Data\nLabels: {test_standard_labels.numpy()}')
    axes[1, 0].axis('off')

    # 顯示shift測試集圖片
    grid_img = make_grid(test_shift_samples, nrow=4, normalize=True)
    axes[1, 1].imshow(grid_img.permute(1, 2, 0))
    axes[1, 1].set_title(f'Test Shift Data\nLabels: {test_shift_labels.numpy()}')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('mnist_data_visualization.png')

    # 打印數據集大小信息
    print(f"Training data shape: {train_samples.shape}")
    print(f"Validation data shape: {val_samples.shape}")
    print(f"Test standard data shape: {test_standard_samples.shape}")
    print(f"Test shift data shape: {test_shift_samples.shape}")

def test_robimagenet(root=None, num_samples=5):
    """
    Test the RobImageNet dataset implementation.
    
    Args:
        root (str): Root directory for the dataset
        num_samples (int): Number of corruption types to test
    """
    print("=== Testing RobImageNet Dataset ===\n")
    
    # Initialize dataset
    try:
        dataset = RobImageNet(root=root, augment=False)
        print("✓ Dataset initialization successful")
    except Exception as e:
        print(f"✗ Dataset initialization failed: {str(e)}")
        return

    # Test basic properties
    print("\n=== Testing Basic Properties ===")
    print(f"Input shape: {dataset.input_shape}")
    print(f"Number of classes: {dataset.num_classes}")

    # Test main dataset splits
    print("\n=== Testing Main Dataset Splits ===")
    main_splits = ['train', 'val', 'test_standard']
    for split in main_splits:
        if split in dataset.datasets:
            data = dataset.datasets[split]
            sample, label = data[0]
            print(f"\n{split.upper()} Split:")
            print(f"✓ Total samples: {len(data)}")
            print(f"✓ Sample shape: {sample.shape}")
            print(f"✓ Label shape: {label.shape}")
            print(f"✓ Data type: {sample.dtype}")
            print(f"✓ Value range: [{sample.min():.3f}, {sample.max():.3f}]")
        else:
            print(f"✗ {split} split not found")

    # Test corruption datasets
    print("\n=== Testing Corruption Datasets ===")
    for corruption in dataset.corruptions[:num_samples]:
        key = f'test_corruption_{corruption}'
        if key in dataset.datasets:
            data = dataset.datasets[key]
            print(f"\n{corruption} corruption:")
            print(f"✓ Total samples: {len(data)}")
            sample, label = data[0]
            print(f"✓ Sample shape: {sample.shape}")
        else:
            print(f"✗ {corruption} corruption dataset not found")

    # Test additional test sets
    print("\n=== Testing Additional Test Sets ===")
    additional_tests = ['test_imagenet_a', 'test_imagenet_o', 'test_imagenet_r']
    for test_set in additional_tests:
        if test_set in dataset.datasets:
            data = dataset.datasets[test_set]
            print(f"\n{test_set.upper()}:")
            print(f"✓ Total samples: {len(data)}")
            sample, label = data[0]
            print(f"✓ Sample shape: {sample.shape}")
            print(f"✓ Value range: [{sample.min():.3f}, {sample.max():.3f}]")
        else:
            print(f"✗ {test_set} not found")

    # Test data loading with DataLoader
    print("\n=== Testing DataLoader ===")
    try:
        from torch.utils.data import DataLoader
        batch_size = 4
        for split in ['train', 'val', 'test_standard']:
            if split in dataset.datasets:
                loader = DataLoader(dataset.datasets[split], 
                                  batch_size=batch_size, 
                                  shuffle=True)
                batch = next(iter(loader))
                print(f"\n{split.upper()} DataLoader:")
                print(f"✓ Batch shape: {batch[0].shape}")
                print(f"✓ Labels shape: {batch[1].shape}")
    except Exception as e:
        print(f"✗ DataLoader test failed: {str(e)}")

    # Visualize sample images
    print("\n=== Generating Sample Visualization ===")
    try:
        import matplotlib.pyplot as plt
        from torchvision.utils import make_grid
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('RobImageNet Sample Images', fontsize=16)
        
        # Plot samples from different splits
        splits_to_show = ['train', 'val', 'test_standard', 'test_imagenet_a']
        for idx, split in enumerate(splits_to_show):
            if split in dataset.datasets:
                data = dataset.datasets[split]
                samples, labels = zip(*[data[i] for i in range(4)])
                samples = torch.stack(samples)
                
                ax = axes[idx//2, idx%2]
                grid_img = make_grid(samples, nrow=2, normalize=True)
                ax.imshow(grid_img.permute(1, 2, 0))
                ax.set_title(f'{split}\nLabels: {labels}')
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('robimagenet_samples.png')
        print("✓ Sample visualization saved as 'robimagenet_samples.png'")
    except Exception as e:
        print(f"✗ Visualization failed: {str(e)}")

    print("\n=== Test Complete ===")  

if __name__ == "__main__":
    # test_dataset('cifar10')
    # test_dataset('cifar100')
    # test_dataset('imagenet')

    # visualize_dataset_tsne(dataset_name='cifar10', n_samples=1000)

    # visualize_mnist_data()

    test_robimagenet()

