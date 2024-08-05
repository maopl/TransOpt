import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class RandomColorJitter:
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5):
        self.transform = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
    
    def __call__(self, img):
        return self.transform(img)

class RandomNoise:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        img = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(self.mean, self.std, img.shape)
        # img = np.clip(img + noise, 0, 1) * 255
        img = np.clip(noise, 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8))

class BGColored:
    def __call__(self, img):
        img = np.array(img)
        dtype = img.dtype
        h, w = img.shape
        img = np.reshape(img, [h, w, 1])
        
        # Convert to red or blue
        if np.random.rand() < 0.5:
            img = np.concatenate([img, np.zeros((h, w, 2), dtype=dtype)], axis=2)  # Red
        else:
            img = np.concatenate([np.zeros((h, w, 2), dtype=dtype), img], axis=2)  # Blue
        
        return Image.fromarray(img, 'RGB')
    
    
class BGGreen(object):

    def __call__(self, img):
        img = np.array(img)
        dtype = img.dtype
        h, w = img.shape
        img = np.reshape(img, [h, w, 1])
        
        # Convert to red
        # img = np.concatenate([img, np.zeros((h, w, 2), dtype=dtype)], axis=2)
        # Convert to green
        img = np.concatenate([np.zeros((h, w, 1), dtype=dtype), img, np.zeros((h, w, 1), dtype=dtype)], axis=2)
    
        return Image.fromarray(img, 'RGB')
    
    
    
# 定义组合转换
transform = transforms.Compose([
    BGColored(),
    transforms.Resize((32, 32)),
    # transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    # transforms.RandomRotation(20),
    # transforms.RandomHorizontalFlip(),
    # RandomColorJitter(),
    RandomNoise(),
    transforms.ToTensor()
])

# 定义组合转换
transform2 = transforms.Compose([
    BGGreen(),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# 加载MNIST数据集
dataset = MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
]))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 获取一批图片并应用转换
data_iter = iter(dataloader)
images, labels = next(data_iter)

# 将原始图像批次转换为PIL图像列表
original_pil_images = [transforms.ToPILImage()(image) for image in images]

# 应用组合转换
transformed_images = [transform(image) for image in original_pil_images]

test_images = [transform2(image) for image in original_pil_images]

# 显示原始和转换后的图片
def show_images(images, title):
    fig, axs = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle(title)
    for i, img in enumerate(images):
        ax = axs[i // 8, i % 8]
        img = img.permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis('off')
    plt.show()

# 显示原始图片
# show_images(test_images, title="Original MNIST Images")
# plt.savefig('test3')
# plt.clf()
# 显示转换后的图片
show_images(transformed_images, title="Augmented MNIST Images")
plt.savefig('test4')