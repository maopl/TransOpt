import torch
import numpy as np
import random
from transopt.benchmark.HPO.image_options import *
from torchvision import transforms

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from PIL import Image
from torchvision.models import resnet18
import seaborn as sns
from sklearn.mixture import GaussianMixture


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes = None):
        if n_holes is None:
            self.n_holes = 1
        else:
            self.n_holes = n_holes

    def __call__(self, img, length = None):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """

        self.length = length
            

        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
    

class Policy(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        self.func = {
            "shearX": ShearX(fillcolor=fillcolor),
            "shearY": ShearY(fillcolor=fillcolor),
            "translateX": TranslateX(fillcolor=fillcolor),
            "translateY": TranslateY(fillcolor=fillcolor),
            "rotate": Rotate(),
            "color": Color(),
            "posterize": Posterize(),
            "solarize": Solarize(),
            "contrast": Contrast(),
            "sharpness": Sharpness(),
            "brightness": Brightness(),
            "autocontrast": AutoContrast(),
            "equalize": Equalize(),
            "invert": Invert()
        }
        
        self.operations = list(self.func.keys())
        self.num_operations = len(self.operations) - 1

    def __call__(self, img, random):
        # 根据random值均匀选择一个增强方法
        operation_idx = round(random * self.num_operations)
        operation_name = self.operations[operation_idx]
        
        # 随机选择增强强度
        magnitude_idx = np.random.randint(0, 10)  # 随机选择0-9的索引
        magnitude = self.ranges[operation_name][magnitude_idx]
        
        # 应用选中的增强方法
        return self.func[operation_name](img, magnitude)



class GaussianMixtureAugmentation:
    def __init__(self, mu1=0.3, sigma1=0.1, mu2=0.7, sigma2=0.1, weight=0.5, fillcolor=(128, 128, 128)):
        # 两个高斯分布的参数
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.mu2 = mu2 
        self.sigma2 = sigma2
        self.weight = weight # 控制两个高斯的混合权重
        
        self.cutout = Cutout(n_holes=1)
        self.policy = Policy()
        self.num_aug = 2
        self.augmentation_methods = [self.cutout, self.policy]

    def sample_parameters(self):
        # 根据权重决定从哪个高斯分布采样
        if np.random.random() < self.weight:
            params = np.random.normal(self.mu1, self.sigma1, 2)
        else:
            params = np.random.normal(self.mu2, self.sigma2, 2)
        
        # 将参数限制在[0,1]范围内
        params = np.clip(params, 0, 1)
        return params


    def reset_gaussian(self, mu1=0.3, sigma1=0.1, mu2=0.7, sigma2=0.1, weight=0.5):
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.weight = self.weight
        

    def __call__(self, img):
        # 采样参数
        params = self.sample_parameters()
        
        # 使用第一个参数选择增强方法
        method_idx = int(params[0] * self.num_aug) % self.num_aug
        method = self.augmentation_methods[method_idx]
        
        # 使用第二个参数控制增强强度
        if method == self.cutout:
            # 将参数映射到合理的cutout长度范围(8-32像素)
            length = int(8 + params[1] * 24)
            # 先转换为tensor
            img_tensor = transforms.ToTensor()(img)
            img_tensor = self.cutout(img_tensor, length=length)
            return transforms.ToPILImage()(img_tensor)
        else:
            # 直接使用参数作为policy的强度
            return method(img, params[1])
            




def visualize_augmentation(original_img, augmented_img):
    """Visualize the original and augmented images side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(np.transpose(original_img.numpy(), (1, 2, 0)))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(np.transpose(augmented_img.numpy(), (1, 2, 0)))
    axes[1].set_title("Augmented Image")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    # Load a sample dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize the GaussianMixtureAugmentation
    augmenter = GaussianMixtureAugmentation()

    # Example data to fit the GMM
    # This should be replaced with actual data that represents the parameter space
    example_data = np.random.rand(100, 4)
    augmenter.fit(example_data)

    # Get a sample image from the dataset
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    original_img = images[0]

    # Apply the augmentation
    augmented_img = augmenter(original_img)

    # Visualize the original and augmented images
    visualize_augmentation(original_img, augmented_img)

if __name__ == "__main__":
    main()