import torch
import torchvision
from torchvision import transforms
from pytorch_fid import fid_score
from transopt.benchmark.HPO import datasets
from transopt.benchmark.HPO.augmentation import mixup_data
import numpy as np
import os
from PIL import Image

def save_images_to_dir(images, dir_path):
    os.makedirs(dir_path, exist_ok=True)
    for idx, img in enumerate(images):
        img.save(os.path.join(dir_path, f'img_{idx}.png'))

def calculate_fid(path1, path2, device='cuda'):
    """计算两个图像目录之间的FID分数"""
    return fid_score.calculate_fid_given_paths([path1, path2], 
                                             batch_size=50,
                                             device=device,
                                             dims=2048)

def main():
    data_dir = None
    temp_dir = 'temp_fid_images'
    
    # 创建基础目录
    base_dir = os.path.join(temp_dir, 'non_aug')
    auto_dir = os.path.join(temp_dir, 'autoaugment')
    cut_dir = os.path.join(temp_dir, 'cutout')
    mixup_dir = os.path.join(temp_dir, 'mixup')
    
    try:
        # 获取非增强数据
        non_aug_data = vars(datasets)['RobCifar10'](root=data_dir, augment=None)
        non_aug_train_data = non_aug_data.datasets['train']
        non_aug_train_imgs = [transforms.ToPILImage()(img[0].squeeze()) 
                            for img in non_aug_train_data]
        save_images_to_dir(non_aug_train_imgs, base_dir)

        # 获取AutoAugment数据
        aug_data = vars(datasets)['RobCifar10'](root=data_dir, augment='autoaugment')
        aug_train_data = aug_data.datasets['train']
        aug_train_imgs = [transforms.ToPILImage()(img[0].squeeze()) 
                         for img in aug_train_data]
        save_images_to_dir(aug_train_imgs, auto_dir)

        # 获取Cutout数据
        cut_data = vars(datasets)['RobCifar10'](root=data_dir, augment='cutout')
        cut_train_data = cut_data.datasets['train']
        cut_train_imgs = [transforms.ToPILImage()(img[0].squeeze()) 
                         for img in cut_train_data]
        save_images_to_dir(cut_train_imgs, cut_dir)

        # 获取Mixup数据
        mixup_train_data = non_aug_data.datasets['train']
        all_images = [mixup_train_data[i][0] for i in range(len(mixup_train_data))]
        all_labels = [mixup_train_data[i][1] for i in range(len(mixup_train_data))]
        all_images = torch.stack(all_images)
        all_labels = torch.stack(all_labels)
        
        mixed_images, _, _, _ = mixup_data(all_images, all_labels, alpha=0.5, device="cpu")
        mixup_train_imgs = [transforms.ToPILImage()(img.squeeze()) 
                           for img in mixed_images]
        save_images_to_dir(mixup_train_imgs, mixup_dir)

        # 计算各种增强方法与原始数据之间的FID
        print("Computing FID scores...")
        auto_fid = calculate_fid(base_dir, auto_dir)
        cut_fid = calculate_fid(base_dir, cut_dir)
        mixup_fid = calculate_fid(base_dir, mixup_dir)

        # 计算非增强数据集与其自身的FID
        self_fid = calculate_fid(base_dir, base_dir)

        # 打印结果
        print(f"AutoAugment FID (vs. Non-Augmented): {auto_fid:.2f}")
        print(f"Cutout FID (vs. Non-Augmented): {cut_fid:.2f}")
        print(f"Mixup FID (vs. Non-Augmented): {mixup_fid:.2f}")
        print(f"Self FID (Non-Augmented vs. Non-Augmented): {self_fid:.2f}")

    finally:
        # 清理临时文件
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
