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

def mixup_data(x, y, alpha=0.3, device='cpu'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    print('mixup in the device:', device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

    
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes = None, length = None):
        if n_holes is None:
            self.n_holes = 1
        else:
            self.n_holes = n_holes
        if length is None:
            self.length = 16
        else:
            self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
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
    
    
    
    
class ImageNetPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.

        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform = transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"
    
    
class CIFAR10PolicyPhotometric(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Photometric Policy"


class CIFAR10PolicyGeometric(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "shearX", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.7, "translateX", 9, 0.9, "autocontrast", 1, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Geometric Policy"


class SVHNPolicy(object):
    """ Randomly choose one of the best 25 Sub-policies on SVHN.

        Example:
        >>> policy = SVHNPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     SVHNPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.9, "shearX", 4, 0.2, "invert", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.7, "invert", 5, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.6, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 3, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "equalize", 1, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.8, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.4, "invert", 5, fillcolor),
            SubPolicy(0.9, "shearY", 5, 0.2, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 6, 0.8, "autocontrast", 1, fillcolor),
            SubPolicy(0.6, "equalize", 3, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.3, "solarize", 3, fillcolor),
            SubPolicy(0.8, "shearY", 8, 0.7, "invert", 4, fillcolor),
            SubPolicy(0.9, "equalize", 5, 0.6, "translateY", 6, fillcolor),
            SubPolicy(0.9, "invert", 4, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.3, "contrast", 3, 0.8, "rotate", 4, fillcolor),

            SubPolicy(0.8, "invert", 5, 0.0, "translateY", 2, fillcolor),
            SubPolicy(0.7, "shearY", 6, 0.4, "solarize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 0.8, "rotate", 4, fillcolor),
            SubPolicy(0.3, "shearY", 7, 0.9, "translateX", 3, fillcolor),
            SubPolicy(0.1, "shearX", 6, 0.6, "invert", 5, fillcolor),

            SubPolicy(0.7, "solarize", 2, 0.6, "translateY", 7, fillcolor),
            SubPolicy(0.8, "shearY", 4, 0.8, "invert", 8, fillcolor),
            SubPolicy(0.7, "shearX", 9, 0.8, "translateY", 3, fillcolor),
            SubPolicy(0.8, "shearY", 5, 0.7, "autocontrast", 3, fillcolor),
            SubPolicy(0.7, "shearX", 2, 0.1, "invert", 5, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment SVHN Policy"


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),  # 修改这里
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        func = {
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

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img


class AugMixOps(object):
    def __init__(self, all_ops=False):
        self.all_ops = all_ops
        self.IMAGE_SIZE = 32  # Add IMAGE_SIZE as class attribute
    
    def int_parameter(self, level, maxval):
        """Helper function to scale `val` between 0 and maxval .

        Args:
            level: Level of the operation that will be between [0, `PARAMETER_MAX`].
            maxval: Maximum value that the operation can have. This will be scaled to
            level/PARAMETER_MAX.

        Returns:
            An int that results from scaling `maxval` according to `level`.
        """
        return int(level * maxval / 10)

    def float_parameter(self, level, maxval):
        """Helper function to scale `val` between 0 and maxval.

        Args:
            level: Level of the operation that will be between [0, `PARAMETER_MAX`].
            maxval: Maximum value that the operation can have. This will be scaled to
            level/PARAMETER_MAX.

        Returns:
            A float that results from scaling `maxval` according to `level`.
        """
        return float(level) * maxval / 10.

    def sample_level(self, n):
        return np.random.uniform(low=0.1, high=n)

    def autocontrast(self, pil_img, _):
        return ImageOps.autocontrast(pil_img)

    def equalize(self, pil_img, _):
        return ImageOps.equalize(pil_img)

    def posterize(self, pil_img, level):
        level = self.int_parameter(self.sample_level(level), 4)
        return ImageOps.posterize(pil_img, 4 - level)

    def rotate(self, pil_img, level):
        degrees = self.int_parameter(self.sample_level(level), 30)
        if np.random.uniform() > 0.5:
            degrees = -degrees
        return pil_img.rotate(degrees, resample=Image.BILINEAR)

    def solarize(self, pil_img, level):
        level = self.int_parameter(self.sample_level(level), 256)
        return ImageOps.solarize(pil_img, 256 - level)

    def shear_x(self, pil_img, level):
        level = self.float_parameter(self.sample_level(level), 0.3)
        if np.random.uniform() > 0.5:
            level = -level
        return pil_img.transform((self.IMAGE_SIZE, self.IMAGE_SIZE),
                                Image.AFFINE, (1, level, 0, 0, 1, 0),
                                resample=Image.BILINEAR)

    def shear_y(self, pil_img, level):
        level = self.float_parameter(self.sample_level(level), 0.3)
        if np.random.uniform() > 0.5:
            level = -level
        return pil_img.transform((self.IMAGE_SIZE, self.IMAGE_SIZE),
                                Image.AFFINE, (1, 0, 0, level, 1, 0),
                                resample=Image.BILINEAR)

    def translate_x(self, pil_img, level):
        level = self.int_parameter(self.sample_level(level), self.IMAGE_SIZE / 3)
        if np.random.random() > 0.5:
            level = -level
        return pil_img.transform((self.IMAGE_SIZE, self.IMAGE_SIZE),
                                Image.AFFINE, (1, 0, level, 0, 1, 0),
                                resample=Image.BILINEAR)

    def translate_y(self, pil_img, level):
        level = self.int_parameter(self.sample_level(level), self.IMAGE_SIZE / 3)
        if np.random.random() > 0.5:
            level = -level
        return pil_img.transform((self.IMAGE_SIZE, self.IMAGE_SIZE),
                                Image.AFFINE, (1, 0, 0, 0, 1, level),
                                resample=Image.BILINEAR)

    # operation that overlaps with ImageNet-C's test set
    def color(self, pil_img, level):
        level = self.float_parameter(self.sample_level(level), 1.8) + 0.1
        return ImageEnhance.Color(pil_img).enhance(level)

    # operation that overlaps with ImageNet-C's test set
    def contrast(self, pil_img, level):
        level = self.float_parameter(self.sample_level(level), 1.8) + 0.1
        return ImageEnhance.Contrast(pil_img).enhance(level)

    # operation that overlaps with ImageNet-C's test set
    def brightness(self, pil_img, level):
        level = self.float_parameter(self.sample_level(level), 1.8) + 0.1
        return ImageEnhance.Brightness(pil_img).enhance(level)

    # operation that overlaps with ImageNet-C's test set
    def sharpness(self, pil_img, level):
        level = self.float_parameter(self.sample_level(level), 1.8) + 0.1
        return ImageEnhance.Sharpness(pil_img).enhance(level)

    def get_ops_list(self):
        if not self.all_ops:
            self.augmentations = [
                self.autocontrast, self.equalize, self.posterize, self.rotate, self.solarize, self.shear_x, self.shear_y, self.translate_x, self.translate_y]
            return self.augmentations
        else:
            self.augmentations_all = [
                self.autocontrast, self.equalize, self.posterize, self.rotate, self.solarize, self.shear_x, self.shear_y,
                self.translate_x, self.translate_y, self.color, self.contrast, self.brightness, self.sharpness]
            return self.augmentations_all

    def aug(self, image, mixture_width=3, mixture_depth=-1, aug_severity=3):
        """Perform AugMix augmentations and compute mixture.

        Args:
            image: PyTorch tensor input image
            preprocess: Preprocessing function which should return a torch tensor.
            mixture_width: Number of augmentation chains to mix per augmented example
            mixture_depth: Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]
            aug_severity: Severity of base augmentation operators

        Returns:
            mixed: Augmented and mixed image.
        """
        # Convert tensor to PIL image
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(image)

        aug_list = self.get_ops_list()
        ws = np.float32(np.random.dirichlet([1] * mixture_width))
        m = np.float32(np.random.beta(1, 1))

        # Convert back to tensor for mixing
        to_tensor = transforms.ToTensor()
        mix = torch.zeros_like(image)

        for i in range(mixture_width):
            image_aug = pil_image.copy()  # Use PIL image copy
            depth = mixture_depth if mixture_depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(aug_list)
                image_aug = op(image_aug, aug_severity)
            # Convert augmented PIL image to tensor before mixing
            image_aug_tensor = to_tensor(image_aug)
            mix += ws[i] * image_aug_tensor

        mixed = (1 - m) * image + m * mix
        return mixed


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self, no_jsd=False, all_ops=False):
        self.augmix = AugMixOps(all_ops)
        self.no_jsd = no_jsd

    def augment(self, x, y):
        if self.no_jsd:
            return self.augmix.aug(x), y
        else:
            im_tuple = (x, self.augmix.aug(x),
                       self.augmix.aug(x))
            return im_tuple, y
        

class ParameterizedAugmentation(object):
    """Parameterized data augmentation that applies all operations with weighted intensities.
    
    Args:
        fillcolor (tuple): RGB fill color for augmentations that introduce empty pixels
        
    Example:
        >>> augmenter = ParameterizedAugmentation()
        >>> # weights is a list of 14 values between 0-1 for each operation
        >>> transformed = augmenter(image, weights)
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        # Define augmentation operations in order
        self.all_ops = False
        self.IMAGE_SIZE = 32
        self.ops_num = len(self.get_ops_list())

    def int_parameter(self, level, maxval):
        """Helper function to scale `val` between 0 and maxval .

        Args:
            level: Level of the operation that will be between [0, `PARAMETER_MAX`].
            maxval: Maximum value that the operation can have. This will be scaled to
            level/PARAMETER_MAX.

        Returns:
            An int that results from scaling `maxval` according to `level`.
        """
        return int(level * maxval / 10)

    def float_parameter(self, level, maxval):
        """Helper function to scale `val` between 0 and maxval.

        Args:
            level: Level of the operation that will be between [0, `PARAMETER_MAX`].
            maxval: Maximum value that the operation can have. This will be scaled to
            level/PARAMETER_MAX.

        Returns:
            A float that results from scaling `maxval` according to `level`.
        """
        return float(level) * maxval / 10.

    # def sample_level(self, n):
    #     return np.random.uniform(low=0.1, high=n)

    def autocontrast(self, pil_img, _):
        return ImageOps.autocontrast(pil_img)

    def equalize(self, pil_img, _):
        return ImageOps.equalize(pil_img)

    def posterize(self, pil_img, level):
        level = self.int_parameter(level, 4)
        return ImageOps.posterize(pil_img, 4 - level)

    def rotate(self, pil_img, level):
        degrees = self.int_parameter(level, 30)
        if np.random.uniform() > 0.5:
            degrees = -degrees
        return pil_img.rotate(degrees, resample=Image.BILINEAR)

    def solarize(self, pil_img, level):
        level = self.int_parameter(level, 256)
        return ImageOps.solarize(pil_img, 256 - level)

    def shear_x(self, pil_img, level):
        level = self.float_parameter(level, 0.3)
        if np.random.uniform() > 0.5:
            level = -level
        return pil_img.transform((self.IMAGE_SIZE, self.IMAGE_SIZE),
                                Image.AFFINE, (1, level, 0, 0, 1, 0),
                                resample=Image.BILINEAR)

    def shear_y(self, pil_img, level):
        level = self.float_parameter( (level), 0.3)
        if np.random.uniform() > 0.5:
            level = -level
        return pil_img.transform((self.IMAGE_SIZE, self.IMAGE_SIZE),
                                Image.AFFINE, (1, 0, 0, level, 1, 0),
                                resample=Image.BILINEAR)

    def translate_x(self, pil_img, level):
        level = self.int_parameter( (level), self.IMAGE_SIZE / 3)
        if np.random.random() > 0.5:
            level = -level
        return pil_img.transform((self.IMAGE_SIZE, self.IMAGE_SIZE),
                                Image.AFFINE, (1, 0, level, 0, 1, 0),
                                resample=Image.BILINEAR)

    def translate_y(self, pil_img, level):
        level = self.int_parameter( (level), self.IMAGE_SIZE / 3)
        if np.random.random() > 0.5:
            level = -level
        return pil_img.transform((self.IMAGE_SIZE, self.IMAGE_SIZE),
                                Image.AFFINE, (1, 0, 0, 0, 1, level),
                                resample=Image.BILINEAR)

    # operation that overlaps with ImageNet-C's test set
    def color(self, pil_img, level):
        level = self.float_parameter(level, 1.8) + 0.1
        return ImageEnhance.Color(pil_img).enhance(level)

    # operation that overlaps with ImageNet-C's test set
    def contrast(self, pil_img, level):
        level = self.float_parameter(level, 1.8) + 0.1
        return ImageEnhance.Contrast(pil_img).enhance(level)

    # operation that overlaps with ImageNet-C's test set
    def brightness(self, pil_img, level):
        level = self.float_parameter(level, 1.8) + 0.1
        return ImageEnhance.Brightness(pil_img).enhance(level)

    # operation that overlaps with ImageNet-C's test set
    def sharpness(self, pil_img, level):
        level = self.float_parameter(level, 1.8) + 0.1
        return ImageEnhance.Sharpness(pil_img).enhance(level)

    def get_ops_list(self):
        if not self.all_ops:
            self.augmentations = [
                self.autocontrast, self.equalize, self.posterize, self.rotate, self.solarize, self.shear_x, self.shear_y, self.translate_x, self.translate_y]
            return self.augmentations
        else:
            self.augmentations_all = [
                self.autocontrast, self.equalize, self.posterize, self.rotate, self.solarize, self.shear_x, self.shear_y,
                self.translate_x, self.translate_y, self.color, self.contrast, self.brightness, self.sharpness]
            return self.augmentations_all

    def __call__(self, img, level):
        """Apply all augmentations with weighted intensities.
        
        Args:
            img: PIL Image to augment
            weights: List of 14 weights between 0-1 for each operation's intensity
            
        Returns:
            Augmented PIL Image
        """
        augmented = img
        ops = self.get_ops_list()
        for op, weight in zip(ops, level):
            # Apply each operation with its weight as intensity
            if weight > 0:  # Only apply if weight > 0
                augmented = op(augmented, weight)
        return augmented

    def __repr__(self):
        return "ParameterizedAugmentation"

def test_parameterized_augmentation(image, weight_sets, num_samples=10):
    """
    测试不同权重组合下的数据增强效果
    
    Args:
        image: 输入图像 (PIL Image 或 Tensor)
        weight_sets: 权重组合列表,每个元素是包含14个权重的列表
        num_samples: 每组参数生成的样本数
    """
    if isinstance(image, torch.Tensor):
        to_pil = transforms.ToPILImage()
        aug_img = to_pil(image)
    else:
        aug_img = image
    
    augmenter = ParameterizedAugmentation()
    ops = augmenter.get_ops_list()
    num_ops = len(ops)
    
    fig, axes = plt.subplots(len(weight_sets), num_ops+1, figsize=(20, 4*len(weight_sets)))
    if len(weight_sets) == 1:
        axes = axes.reshape(1, -1)
    
    for i, weights in enumerate(weight_sets):
        # 显示原图
        axes[i,0].imshow(aug_img)
        axes[i,0].set_title('Original')
        axes[i,0].axis('off')
        
        # 对每个操作进行增强并显示
        img_copy = aug_img.copy()
        for j, (op, weight) in enumerate(zip(ops, weights)):
            # 只应用当前操作和权重
            augmented = op(img_copy, weight)
            
            # 显示增强后的图片
            axes[i,j+1].imshow(augmented)
            axes[i,j+1].set_title(f'{op.__name__}\nw={weight:.2f}')
            axes[i,j+1].axis('off')
            
            img_copy = aug_img.copy()  # 重置图片,避免叠加效果
    
    plt.tight_layout()
    plt.savefig('augmentation_samples.png')

def visualize_augmented_images(image, weight_matrix, num_samples=1):
    """
    可视化原图和经过不同权重组合增强后的图片对比
    
    Args:
        image: 输入图像 (PIL Image 或 Tensor)
        weight_matrix: 权重矩阵,每一行是一组权重列表
        num_samples: 每组参数生成的样本数
    """
    if isinstance(image, torch.Tensor):
        to_pil = transforms.ToPILImage()
        aug_img = to_pil(image)
    else:
        aug_img = image
    
    augmenter = ParameterizedAugmentation()
    num_weight_sets = len(weight_matrix)
    
    # 创建子图布局
    fig, axes = plt.subplots(num_weight_sets, 2, figsize=(8, 4*num_weight_sets))
    if num_weight_sets == 1:
        axes = axes.reshape(1, -1)
    
    for i, weights in enumerate(weight_matrix):
        # 显示原图
        axes[i,0].imshow(aug_img)
        axes[i,0].set_title('Original')
        axes[i,0].axis('off')
        
        # 应用所有增强操作
        augmented = augmenter(aug_img, weights)
        
        # 显示增强后的图片
        axes[i,1].imshow(augmented)
        axes[i,1].set_title(f'Augmented\nweights={[f"{w:.2f}" for w in weights]}')
        axes[i,1].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmented_comparison.png')






def extract_features(image, weight_sets, num_samples=100):
    """
    提取增强后图片的特征用于分布可视化
    
    Args:
        image: 输入图像 (PIL Image 或 Tensor)
        weight_sets: 权重组合列表
        num_samples: 每组参数生成的样本数
    """
    if isinstance(image, torch.Tensor):
        to_pil = transforms.ToPILImage()
        img = to_pil(image)
    else:
        img = image
    
    model = resnet18(pretrained=True)
    model.eval()
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    augmenter = ParameterizedAugmentation()
    
    features = []
    labels = []
    
    for weight_idx, weights in enumerate(weight_sets):
        for _ in range(num_samples):
            noisy_weights = [max(0, min(1, w + np.random.normal(0, 0.1))) for w in weights]
            aug_img = augmenter(img, noisy_weights)
            
            input_tensor = transform(aug_img).unsqueeze(0)
            
            with torch.no_grad():
                feature = feature_extractor(input_tensor)
                feature = feature.squeeze().flatten().numpy()
                
            features.append(feature)
            labels.append(weight_idx)
    
    return np.array(features), np.array(labels)

def visualize_distribution(features, labels, weight_sets):
    """使用t-SNE可视化特征分布"""
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    
    for i, weights in enumerate(weight_sets):
        mask = labels == i
        # 获取前3个最大权重的操作
        op_weights = list(zip([f'Op{i}' for i in range(len(weights))], weights))
        top3 = sorted(op_weights, key=lambda x: x[1], reverse=True)[:3]
        label = ' + '.join([f'{op}({w:.2f})' for op,w in top3])
        
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   label=label, alpha=0.6)
    
    plt.legend()
    plt.title('t-SNE visualization of augmented samples')
    plt.savefig('augmentation_distribution.png')

# 测试代码
if __name__ == "__main__":
    # 定义不同的权重组合
    weight_sets = [
        [random.randint(0, 10) for _ in range(13)],
        [random.randint(0, 10) for _ in range(13)],
        [random.randint(0, 10) for _ in range(13)],
        [random.randint(0, 10) for _ in range(13)],
    ]
    
    transform = transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    
    test_idx = random.randint(0, len(trainset)-1)
    test_image, _ = trainset[test_idx]
    
    print("Testing augmentation effects...")
    test_parameterized_augmentation(test_image, weight_sets)
    visualize_augmented_images(test_image, weight_sets)
    
    print("Extracting features and visualizing distribution...")
    features, labels = extract_features(test_image, weight_sets)
    visualize_distribution(features, labels, weight_sets)