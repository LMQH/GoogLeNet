from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def get_card_dataloaders(data_root, batch_size, num_workers):
    """创建并返回卡片数据集的数据加载器，包含数据增强"""

    # 训练集增强
    train_transform = transforms.Compose([
        transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 1.5)),  # 高斯模糊
        # 随机裁剪：裁剪比例为原图的 x%~y%，然后缩放到 224x224
        transforms.RandomResizedCrop(
            size=224,  # 输出尺寸
            scale=(0.95, 1.0),  # 裁剪区域占原图比例范围
            ratio=(0.95, 1.05)  # 宽高比范围
        ),
        # transforms.RandomHorizontalFlip(p=0.3),  # 随机水平翻转
        transforms.RandomRotation(5),   # 随机旋转
        transforms.ColorJitter(         # 随机改变亮度、对比度、饱和度
            brightness=0.1,  # 亮度
            contrast=0.1,  # 对比度
            saturation=0.1,  # 饱和度
            hue=0.02  # 色调
        ),
        transforms.RandomAffine(        # 平移+缩放
            degrees=0,  # 旋转角度
            translate=(0.05, 0.05),
            scale=(0.97, 1.03)
        ),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.15),  # 透视变换
        transforms.ToTensor(),
        transforms.RandomErasing(       # 随机擦除
            p=0.15,
            scale=(0.02, 0.08),
            ratio=(0.5, 2.0)
        ),
        transforms.Normalize(           # 使用 ImageNet 预训练模型的均值和标准差
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 验证集 & 测试集：不做数据增强，只做基础预处理
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 加载数据集
    # ImageFolder返回类别索引和标签,形式为(图片,标签)，标签类型为int
    train_dataset: ImageFolder = datasets.ImageFolder(
        root=f"{data_root}/train",
        transform=train_transform
    )
    valid_dataset: ImageFolder = datasets.ImageFolder(
        root=f"{data_root}/valid",
        transform=eval_transform
    )
    test_dataset: ImageFolder = datasets.ImageFolder(
        root=f"{data_root}/test",
        transform=eval_transform
    )

    # 创建 DataLoader
    # DataLoader返回一个迭代器，迭代器返回的是一个批次的数据和标签
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # 启用内存映射
        # collate_fn=collate_fn
    )

    valid_loader: DataLoader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        # collate_fn=collate_fn
    )

    test_loader: DataLoader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        # collate_fn=collate_fn
    )

    # 调试用
    # import torch
    # def check_dataloader(loader, name):
    #     batch = next(iter(loader))
    #     inputs, labels = batch
    #     print(f"{name}批次检查：")
    #     print(f"inputs类型：{type(inputs)}，形状：{inputs.shape if isinstance(inputs, torch.Tensor) else '非张量'}")
    #     print(f"labels类型：{type(labels)}，形状：{labels.shape if isinstance(labels, torch.Tensor) else '非张量'}")
    #
    # check_dataloader(train_loader, "训练集")
    # check_dataloader(valid_loader, "验证集")
    # check_dataloader(test_loader, "测试集")

    return train_loader, valid_loader, test_loader, train_dataset


if __name__ == '__main__':
    # 独立测试部分，用于验证函数可用性
    data_root = "../dataset/cards_image/img"
    batch_size = 32
    num_workers = 4

    # 测试数据加载器是否能正常工作
    train_loader, valid_loader, test_loader, train_dataset = get_card_dataloaders(
        data_root, batch_size, num_workers
    )

    # 打印基础信息，确认数据加载成功
    # print(f"数据加载测试：")
    # print(f"类别数：{len(train_dataset.classes)}")  # 53分类
    # print(f"训练集样本数：{len(train_loader.dataset)}")  # type: ignore
    # print(f"验证集样本数：{len(valid_loader.dataset)}")  # type: ignore
    # print(f"测试集样本数：{len(test_loader.dataset)}")  # type: ignore

    # 随机查看3个训练集样本的标签
    # for i in range(3):
    #     img, label = train_dataset[i]
    #     print(f"样本{i}的标签: {label}, 类型: {type(label)}")  # 应输出整数和int类型

    print(train_dataset.class_to_idx)  # 53个类别的索引
