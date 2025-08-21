import torch
import torch.nn as nn
import torch.optim as optim
from data import get_card_dataloaders
from model import get_frozen_googlenet
from train import train_model
from test import test_model
from torch.optim.lr_scheduler import CosineAnnealingLR
import time


# ----------------------------
# 超参数配置
# ----------------------------

# 数据相关参数
data_root = "../dataset/cards_image/img"
batch_size = 32  # 批次大小
num_workers = 4  # 工作线程数
label_smoothing = 0.1  # 标签平滑，默认为0

# 训练相关参数
num_epochs = 40  # 训练轮数
learning_rate = 1e-4  # 学习率
weight_decay = 1e-3  # 权重衰减
dropout_prob = 0.2  # Dropout概率
patience = 8  # 耐心值
delta = 0.001  # 停止条件，提升小于该值则停止训练

# 辅助分类器权重，默认为0.3
# 如果 w 较小，辅助分类器对梯度贡献有限，前几轮主分类器更新慢
# 如果 w 太大，初始阶段随机辅助输出会引入噪声，也可能让训练不稳定
W = 0.3

# 模型保存参数
model_save_dir = "../model"
best_model_name = "best_googlenet.pth"

# TensorBoard日志参数
log_dir = "../logs"  # TensorBoard日志目录
experiment_name = "googlenet_card_classification"  # 实验名称


# ----------------------------
# 主函数
# ----------------------------
def main():
    # 加载数据
    print("\nLoading data(加载数据中)...", flush=True)
    train_loader, valid_loader, test_loader, train_dataset = get_card_dataloaders(
        data_root,
        batch_size=batch_size,
        num_workers=num_workers
    )

    time.sleep(0.5)

    # 显示数据集信息
    print(f"\nNumber of classes(类别总数): {len(train_dataset.classes)}")  # 类别数量
    # print(f"Class names: {train_dataset.classes[:5]}...")  # 类别名称前五个
    print(f"Training samples(训练样本数): {len(train_dataset)}")  # 训练样本数量
    print(f"Validation samples(验证样本数): {len(valid_loader.dataset)}")  # 验证样本数量
    print(f"Test samples(测试样本数): {len(test_loader.dataset)}")  # 测试样本数量

    # 配置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device(使用设备为): {device}")

    # 初始化模型,参数：类别数量, dropout概率
    model = get_frozen_googlenet(len(train_dataset.classes), dropout_prob=dropout_prob).to(device)

    # 定义损失函数
    # 标签平滑
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # 定义优化器，AdamW（初期快速收敛）
    # 主分类器和两个辅助分类器都进行训练
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # 定义学习率调度器
    # 防止震荡或过拟合的关键
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 训练模型
    print("\nStarting training(开始训练)...")
    model = train_model(
        model, train_loader, valid_loader, criterion, optimizer, device,
        num_epochs, model_save_dir, best_model_name, w=W, patience=patience, delta=delta, scheduler=scheduler,
        # 启用混合精度训练
        use_amp=True,
        # 启用TensorBoard日志
        log_dir=log_dir,
        experiment_name=experiment_name
    )

    # 测试模型
    print("\nEvaluating on test set(测试评估)...", flush=True)
    test_model(model, test_loader, device)


if __name__ == '__main__':
    main()
