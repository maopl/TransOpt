import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import AutoAugmentPolicy
from vendi_score import image_utils
import matplotlib.pyplot as plt
import numpy as np
from transopt.benchmark.HPO import datasets
from transopt.benchmark.HPO.augmentation import mixup_data, mixup_criterion, AugMixDataset, ParameterizedAugmentation


data_dir = None

# 获取训练数据
non_aug_data = vars(datasets)['RobCifar10'](root=data_dir, augment=None)
non_aug_train_data = non_aug_data.datasets['train']
non_aug_train_imgs = [transforms.ToPILImage()(img[0].squeeze()) for img in non_aug_train_data]
non_aug_pixel_vs = image_utils.pixel_vendi_score(non_aug_train_imgs)
non_aug_inception_vs = image_utils.embedding_vendi_score(non_aug_train_imgs, device="cuda")
print(f"pixel vs:{non_aug_pixel_vs}, inception vs::{non_aug_inception_vs}")

aug_data = vars(datasets)['RobCifar10'](root=data_dir, augment='autoaugment')
aug_train_data = aug_data.datasets['train']
aug_train_imgs = [transforms.ToPILImage()(img[0].squeeze()) for img in aug_train_data]
aug_pixel_vs = image_utils.pixel_vendi_score(aug_train_imgs)
aug_inception_vs = image_utils.embedding_vendi_score(aug_train_imgs, device="cuda")


cut_data = vars(datasets)['RobCifar10'](root=data_dir, augment='cutout')
cut_train_data = cut_data.datasets['train']
cut_train_imgs = [transforms.ToPILImage()(img[0].squeeze()) for img in cut_train_data]
cut_pixel_vs = image_utils.pixel_vendi_score(cut_train_imgs)
cut_inception_vs = image_utils.embedding_vendi_score(cut_train_imgs, device="cuda")


#mixup
mixup_train_data = non_aug_data.datasets['train']

all_images = [mixup_train_data[i][0] for i in range(0, len(mixup_train_data))]
all_labels = [mixup_train_data[i][1] for i in range(0, len(mixup_train_data))]
all_images = torch.stack(all_images)
all_labels = torch.stack(all_labels)

mixed_images, mixed_y_a, mixed_y_b, _ = mixup_data(all_images, all_labels, alpha=0.5, device="cpu")
mixup_train_imgs = [transforms.ToPILImage()(img.squeeze()) for img in mixed_images]
mixup_pixel_vs = image_utils.pixel_vendi_score(mixup_train_imgs)
mixup_inception_vs = image_utils.embedding_vendi_score(mixup_train_imgs, device="cuda")



# 打印结果
print(f"Non-Augmented Data - Pixel VS: {non_aug_pixel_vs:.2f}, Inception VS: {non_aug_inception_vs:.2f}")
print(f"Augmented Data - Pixel VS: {aug_pixel_vs:.2f}, Inception VS: {aug_inception_vs:.2f}")
print(f"Cutout Data - Pixel VS: {cut_pixel_vs:.2f}, Inception VS: {cut_inception_vs:.2f}")
print(f"Mixup Data - Pixel VS: {mixup_pixel_vs:.2f}, Inception VS: {mixup_inception_vs:.2f}")