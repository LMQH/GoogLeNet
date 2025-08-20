from typing import Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import sys


def _evaluate_model(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        desc: str = "Evaluating"
) -> Tuple[float, float]:
    """
    内部评估函数，用于验证和测试阶段

    参数:
        model: 要评估的模型
        data_loader: 数据加载器
        criterion: 损失函数
        device: 评估设备
        desc: 进度条描述

    返回:
        (evaluation_loss, evaluation_accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(data_loader, desc=desc, position=0, leave=True, file=sys.stdout)
        for inputs, labels in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)

            # 如果是 tuple，则取第一个元素作为主输出
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, dim=1)

            batch_correct = preds.eq(labels).sum().item()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size
            correct += batch_correct

            # 进度条显示
            pbar.set_postfix({
                'loss': f'{running_loss / total:.4f}',
                'acc': f'{(correct / total)*100:.2f}%'
            })
    # 关闭进度条
    pbar.close()

    eval_loss = running_loss / total
    eval_acc = correct / total
    return eval_loss, eval_acc
