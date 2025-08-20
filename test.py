import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple
from evaluate import _evaluate_model
from logger import TensorBoardLogger  # 导入新日志类


def test_model(
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        criterion: nn.Module = None,
        # 新增：TensorBoard相关参数
        log_dir: str = None,
        experiment_name: str = "experiment"
) -> Tuple[float, float]:

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # 初始化测试日志记录器
    logger = None
    if log_dir:
        logger = TensorBoardLogger(
            log_dir=log_dir,
            experiment_name=experiment_name,
            subdir="test"  # 测试日志放在test子目录
        )

    test_loss, test_acc = _evaluate_model(
        model, test_loader, criterion, device, desc="Testing"
    )

    print(f'\nTest Set Results:')
    print(f'Loss: {test_loss:.4f} | Accuracy: {(test_acc * 100):.2f}%', flush=True)

    # 记录测试结果
    if logger:
        logger.add_scalar('Test/Loss', test_loss, 0)
        logger.add_scalar('Test/Accuracy', test_acc, 0)
        logger.add_text('Test_Results', f'Loss: {test_loss:.4f} | Accuracy: {test_acc * 100:.2f}%')
        logger.close()

    return test_loss, test_acc
