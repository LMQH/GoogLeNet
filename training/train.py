import os
import torch
import torch.nn as nn
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader
from .evaluate import _evaluate_model
from .early_stop import EarlyStopping
from GoogLeNet_cards.cards_classification.utils.logger import TensorBoardLogger
# 导入混合精度相关模块
from torch.amp import autocast, GradScaler


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion: nn.Module,  # 损失函数
        optimizer: torch.optim.Optimizer,  # 优化器
        device: torch.device,
        num_epochs: int,
        model_save_dir: str,
        best_model_name: str,
        w: float,  # 辅助分类器权重
        patience: int,  # 早停的耐心值
        delta: float,  # 早停判断值
        scheduler,  # 可选学习率调度器
        use_amp: bool,  # 新增：添加混合精度开关参数
        log_dir: str,
        experiment_name: str
) -> nn.Module:
    """
    训练模型并保存验证集准确率最高的模型
    """
    best_acc = 0.0
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, best_model_name)

    # 初始化早停机制
    early_stopping = EarlyStopping(
        patience=patience,  # 耐心值
        delta=delta,
        verbose=True,  # 显示进度
        path=best_model_path  # 保存模型的路径
    )

    # 初始化日志记录器
    logger = TensorBoardLogger(
        log_dir=log_dir or os.path.join(model_save_dir, "logs"),
        experiment_name=experiment_name,
        subdir="Train"
    )

    # 初始化混合精度缩放器
    scaler = GradScaler("cuda", enabled=use_amp)

    # 记录超参数（简化调用）
    hyperparams = {
        "num_epochs": num_epochs,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "weight_decay": optimizer.param_groups[0].get('weight_decay', 0),
        "batch_size": train_loader.batch_size,
        "patience": patience,
        "w_aux": w,
        "use_amp": use_amp
    }
    logger.add_hparams(hyperparams, {})

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 50)

        # 训练阶段
        model.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0

        # tqdm 显示训练进度
        train_pbar = tqdm(train_loader, total=len(train_loader),
                          desc=f"Training Epoch {epoch + 1}", position=0, leave=True,
                          file=sys.stdout)  # 指定输出流

        # _,忽略索引
        for _, (inputs, labels) in enumerate(train_pbar):

            assert isinstance(inputs, torch.Tensor)
            assert isinstance(labels, torch.Tensor)

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            # 混合精度训练：根据设备动态适配（支持CPU/GPU）
            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)

                # 处理带辅助分类器的输出
                if isinstance(outputs, tuple):
                    main_output, aux1, aux2 = outputs
                    loss_main = criterion(main_output, labels)
                    loss_aux1 = criterion(aux1, labels)
                    loss_aux2 = criterion(aux2, labels)
                    loss = loss_main + w * (loss_aux1 + loss_aux2)
                    # 取主输出的预测结果,简化
                    preds = main_output.argmax(dim=1).long()
                else:
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1).long()

                # 关键修复：确保preds是long类型
                preds = preds.long()

            # 使用scaler进行反向传播
            scaler.scale(loss).backward()

            # 使用scaler进行参数更新
            scaler.step(optimizer)
            scaler.update()

            # 如果 scheduler 是 OneCycleLR，需要在每个 batch 后 step
            if scheduler is not None and scheduler.__class__.__name__ == "OneCycleLR":
                scheduler.step()

            # 更新指标
            batch_size = inputs.size(0)
            train_running_loss += loss.item() * batch_size
            train_total += batch_size
            train_correct += (preds == labels).sum().item()

            train_pbar.set_postfix({
                'loss': f'{train_running_loss / train_total:.4f}',
                'acc': f'{(train_correct / train_total) * 100:.2f}%'
            })

        # 关闭进度条
        train_pbar.close()

        train_epoch_loss = train_running_loss / train_total
        train_epoch_acc = train_correct / train_total
        print(f'Train Loss: {train_epoch_loss:.4f} | Acc: {train_epoch_acc * 100:.2f}%', flush=True)

        # 记录训练指标（简化调用）
        logger.add_scalar('Train/Epoch_Loss', train_epoch_loss, epoch)
        logger.add_scalar('Train/Epoch_Accuracy', train_epoch_acc, epoch)
        logger.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # 验证阶段
        valid_epoch_loss, valid_epoch_acc = _evaluate_model(
            model, valid_loader, criterion, device, desc=f"Validation Epoch {epoch + 1}"
        )
        print(f'Valid Loss: {valid_epoch_loss:.4f} | Acc: {valid_epoch_acc * 100:.2f}%', flush=True)

        # 记录验证指标（简化调用）
        logger.add_scalar('Validation/Epoch_Loss', valid_epoch_loss, epoch)
        logger.add_scalar('Validation/Epoch_Accuracy', valid_epoch_acc, epoch)

        # 更新最佳准确率
        if valid_epoch_acc > best_acc:
            best_acc = valid_epoch_acc
            # 新增：记录最佳准确率
            logger.add_scalar('Best_Accuracy', best_acc, epoch)

        # 调用早停函数（其中包含保存模型部分）
        early_stopping(valid_epoch_acc, model)

        # 检查是否需要提前停止
        if early_stopping.early_stop:
            print(f">> EarlyStop at Epoch {epoch + 1}/{num_epochs} (早停触发，终止训练)")
            # 记录早停事件
            logger.add_text('Training_Event', 'Early Stopping Triggered', epoch)
            break

        # 如果 scheduler 是 CosineAnnealingLR / StepLR 等，每个 epoch 后 step
        if scheduler is not None and scheduler.__class__.__name__ != "OneCycleLR":
            scheduler.step()

    print('\nTraining complete(训练完成)!', flush=True)
    print(f'Best validation accuracy: {(best_acc * 100):.2f}% \n(saved at {best_model_path})', flush=True)

    # 记录最终结果
    logger.add_text('Final_Result', f'Best Accuracy: {best_acc * 100:.2f}%')
    logger.close()  # 关闭日志

    # 加载最佳模型权重
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))

    # 导出ONNX模型
    dummy_input = (torch.randn(1, 3, 224, 224).to(device),)  # 注意末尾的逗号，表示单元素元组
    onnx_path = os.path.join(model_save_dir, "final_model.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"\n最终ONNX模型已保存到: {onnx_path}")

    return model


# 独立测试
if __name__ == '__main__':
    pass
