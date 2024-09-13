import torch
import torchvision
import torchvision.transforms as transforms
from robustbench.data import load_cifar10c
from robustbench.utils import clean_accuracy
from robustbench.model_zoo.cifar10 import ResNet18

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的ResNet50模型
model = ResNet18(pretrained=True).to(device)
model.eval()

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载CIFAR-10测试集
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# 评估在CIFAR-10测试集上的准确率
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on clean CIFAR-10 test images: {100 * correct / total:.2f}%')

# 评估在CIFAR-10-C上的准确率
corruptions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]

for corruption in corruptions:
    x_test, y_test = load_cifar10c(n_examples=10000, corruptions=[corruption], severity=5)
    x_test = torch.from_numpy(x_test).float().to(device)
    y_test = torch.from_numpy(y_test).long().to(device)
    
    acc = clean_accuracy(model, x_test, y_test)
    print(f'Accuracy on CIFAR-10-C ({corruption}): {100 * acc:.2f}%')