# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import TensorDataset, Subset, ConcatDataset, Dataset
from torchvision.datasets import MNIST, ImageNet, CIFAR10, CIFAR100


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision


from robustbench.data import load_cifar10c, load_cifar100c, load_imagenetc

from transopt.benchmark.HPO.augmentation import ImageNetPolicy, CIFAR10Policy, CIFAR10PolicyGeometric, CIFAR10PolicyPhotometric, Cutout

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
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    transform_list = [transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean, std)]
    # transform_list = [transforms.ToPILImage(), transforms.ToTensor()]

    if augmentation_name:
        if dataset_name.lower() in ['cifar10', 'cifar100']:
            if augmentation_name.lower() == 'cutout':
                transform_list.insert(-1,Cutout(n_holes=1, length=16))
            elif augmentation_name.lower() == 'geometric':
                transform_list.insert(1, CIFAR10PolicyGeometric())
            elif augmentation_name.lower() == 'photometric':
                transform_list.insert(1, CIFAR10PolicyPhotometric())
            elif augmentation_name.lower() == 'autoaugment':
                transform_list.insert(1, CIFAR10Policy())
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
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes, augment=None):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                   original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                   original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        # Split into train, val and test
        val_size = len(original_images) // 10
        test_size = len(original_images) // 10
        train_size = len(original_images) - val_size - test_size

        train_images = original_images[:train_size]
        train_labels = original_labels[:train_size]
        val_images = original_images[train_size:train_size+val_size]
        val_labels = original_labels[train_size:train_size+val_size]
        test_images = original_images[train_size+val_size:]
        test_labels = original_labels[train_size+val_size:]

        # Create environments for train set
        for i in range(len(environments)):
            images = train_images[i::len(environments)]
            labels = train_labels[i::len(environments)]
            self.datasets[f'train'] = dataset_transform(images, labels, environments[i])

        # Create validation and test sets
        self.datasets['val'] = dataset_transform(val_images, val_labels, environments[0])
        self.datasets['test'] = dataset_transform(test_images, test_labels, environments[0])

        self.input_shape = input_shape
        self.num_classes = num_classes

        # Add data augmentation if specified
        if augment:
            if augment == 'ddpm':
                ddpm_path = os.path.join(root, 'mnist_ddpm.npz')
                ddpm_data = np.load(ddpm_path)
                ddpm_images = torch.from_numpy(ddpm_data['image']).float()
                ddpm_images = ddpm_images.reshape(-1, 1, 28, 28)
                ddpm_labels = torch.from_numpy(ddpm_data['label']).long()
                self.datasets['train'] = TensorDataset(ddpm_images, ddpm_labels)
            else:
                dataset_transform = data_transform('mnist', augment)
                train_images = self.datasets['train'].tensors[0]
                train_labels = self.datasets['train'].tensors[1]
                transformed_images = torch.stack([dataset_transform(img) for img in train_images])
                self.datasets['train'] = TensorDataset(transformed_images, train_labels)

class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    INPUT_SHAPE = (2, 28, 28)

    def __init__(self, root, test_envs, hparams):
        augment = hparams.get('augment', None)
        super().__init__(root, [0.1, 0.2, 0.9],
                        self.color_dataset, self.INPUT_SHAPE, 2, augment)

    def color_dataset(self, images, labels, environment):
        labels = (labels < 5).float()
        labels = self.torch_xor_(labels,
                                self.torch_bernoulli_(0.25, len(labels)))

        colors = self.torch_xor_(labels,
                                self.torch_bernoulli_(environment,
                                                    len(labels)))
        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']
    INPUT_SHAPE = (1, 28, 28)

    def __init__(self, root, test_envs, hparams):
        augment = hparams.get('augment', None)
        super().__init__(root, [0, 15, 30, 45, 60, 75],
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

        return TensorDataset(x, y)


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
        
        
        dataset_transform = data_transform('cifar10', augment)
        normalized_images = data_transform('cifar10', None)
        
        transformed_images = torch.stack([dataset_transform(img) for img in original_images])
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

        standard_test_images = torch.stack([normalized_images(img) for img in original_dataset_te.data])

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
            x_test_corrupt = torch.stack([normalized_images(img) for img in x_test_corrupt])
            self.datasets[f'test_corruption_{corruption}'] = TensorDataset(x_test_corrupt, y_test_corrupt)

        # Load CIFAR-10.1 dataset
        cifar101_path = os.path.join(root, 'cifar10.1_v6_data.npy')
        cifar101_labels_path = os.path.join(root, 'cifar10.1_v6_labels.npy')
        if os.path.exists(cifar101_path) and os.path.exists(cifar101_labels_path):
            cifar101_data = np.load(cifar101_path)
            cifar101_labels = np.load(cifar101_labels_path)
            cifar101_data = torch.from_numpy(cifar101_data).float() / 255.0
            cifar101_data = cifar101_data.permute(0, 3, 1, 2)  # Change from (N, 32, 32, 3) to (N, 3, 32, 32)
            cifar101_data = torch.stack([normalized_images(img) for img in cifar101_data])
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
            cifar102_images = torch.stack([normalized_images(img) for img in cifar102_images])
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
        
        dataset_transform = data_transform('cifar100', augment)
        normalized_images = data_transform('cifar100', None)
        
        transformed_images = torch.stack([dataset_transform(img) for img in original_images])
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

        standard_test_images = torch.stack([normalized_images(img) for img in original_dataset_te.data])

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
            x_test_corrupt = torch.stack([normalized_images(img) for img in x_test_corrupt])
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
            print("Data augmentation is enabled.")
            return transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            print("Data augmentation is disabled.")
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

if __name__ == "__main__":
    # test_dataset('cifar10')
    # test_dataset('cifar100')
    # test_dataset('imagenet')

    visualize_dataset_tsne(dataset_name='cifar10', n_samples=1000)

    # ... (之后的代码保持不变)