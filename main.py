import torch
import torch.nn as nn
import torch.optim as optim
import time
# 从最上层导入GoogLeNet_cards
from GoogLeNet_cards.cards_classification.data.dataloader import get_card_dataloaders
from GoogLeNet_cards.cards_classification.models.GoogLeNet import get_frozen_googlenet
from GoogLeNet_cards.cards_classification.training.train import train_model
from GoogLeNet_cards.cards_classification.testing.test import test_model
from GoogLeNet_cards.cards_classification.config import (  # 导入配置
    DATA_ROOT, BATCH_SIZE, NUM_EPOCHS, DROPOUT_PROB, PATIENCE, DELTA,
    NUM_WORKERS, LABEL_SMOOTHING, LEARNING_RATE, WEIGHT_DECAY, MAX_LR,
    PCT_START, DIV_FACTOR, FINAL_DIV_FACTOR, AUX_WEIGHT, MODEL_SAVE_DIR,
    BEST_MODEL_NAME, LOG_DIR, EXPERIMENT_NAME
)


# 设置CUDA优化标志,提升训练速度(需cuDNN,可选)
torch.backends.cudnn.benchmark = True  # 固定输入尺寸
torch.backends.cudnn.deterministic = False  # 提高速度，牺牲可重复性


# ----------------------------
# 主函数 main()
# ----------------------------
def main():
    # 加载数据
    print("\nLoading data(加载数据中)...", flush=True)
    train_loader, valid_loader, test_loader, train_dataset = get_card_dataloaders(
        DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    time.sleep(0.5)

    # 显示数据集信息
    print(f"\nNumber of classes(类别总数): {len(train_dataset.classes)}")
    print(f"Training samples(训练样本数): {len(train_dataset)}")
    print(f"Validation samples(验证样本数): {len(valid_loader.dataset)}")
    print(f"Test samples(测试样本数): {len(test_loader.dataset)}")

    # 配置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device(使用设备为): {device}")

    # 初始化模型
    model = get_frozen_googlenet(
        len(train_dataset.classes),
        dropout_prob=DROPOUT_PROB
    ).to(device)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # 定义优化器
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        steps_per_epoch=len(train_loader),
        epochs=NUM_EPOCHS,
        pct_start=PCT_START,
        div_factor=DIV_FACTOR,
        final_div_factor=FINAL_DIV_FACTOR,
        anneal_strategy='linear'
    )

    # 训练模型
    print("\nStarting training(开始训练)...")
    model = train_model(
        model, train_loader, valid_loader, criterion, optimizer, device,
        num_epochs=NUM_EPOCHS,
        model_save_dir=MODEL_SAVE_DIR,
        best_model_name=BEST_MODEL_NAME,
        w=AUX_WEIGHT,
        patience=PATIENCE,
        delta=DELTA,
        scheduler=scheduler,
        use_amp=True,
        log_dir=LOG_DIR,
        experiment_name=EXPERIMENT_NAME
    )

    # 测试模型
    print("\nEvaluating on test set(测试评估)...", flush=True)
    test_model(
        model, test_loader, device,
        log_dir=LOG_DIR,
        experiment_name="Test_Experiment"
    )


if __name__ == "__main__":
    main()
