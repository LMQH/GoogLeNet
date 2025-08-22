# test.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Union, Dict, Any
import numpy as np
from evaluate import _evaluate_model
from logger import TensorBoardLogger  # 导入日志类
from tqdm import tqdm
import sys
import os
from pathlib import Path
from calculate import (_calculate_and_print_metrics,
                       _calculate_and_plot_confusion_matrix,
                       _generate_and_save_classification_report)


def test_model(
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        criterion: Optional[nn.Module] = None,  # 更明确的Optional
        log_dir: Optional[Union[str, Path]] = None,  # 支持Path类型
        experiment_name: str = "experiment"
) -> Tuple[float, float, Optional[Dict[str, Any]]]:
    """
    测试模型，计算并记录详细的评估指标。
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    test_loss, test_acc = _evaluate_model(
        model, test_loader, criterion, device, desc="Testing"
    )

    print(f'Test Loss: {test_loss:.4f} | Accuracy: {(test_acc * 100):.2f}%', flush=True)

    print(f'\n[Collecting Predictions and Labels (收集预测结果与真实标签)...]', flush=True)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Collecting Predictions", position=0, leave=True, file=sys.stdout)
        for inputs, labels in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        pbar.close()

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    class_names = getattr(test_loader.dataset, 'classes', None)
    if class_names is None:
        print("Warning: test_loader.dataset does not have 'classes' attribute. "
              "Using generic labels.")

        # 使用 int(len(...)) 明确类型
        num_unique_labels = int(len(np.unique(all_labels)))
        class_names = [f"Class_{i}" for i in range(num_unique_labels)]
    print(f"Number of classes detected (类名数量): {len(class_names)}")  # 输出类名数量
    # print(f"log_dir received: {log_dir}")

    logger = None
    excel_save_path = None

    if log_dir:
        try:
            logger = TensorBoardLogger(
                log_dir=log_dir,
                experiment_name=experiment_name,
                subdir="Test"
            )
            # 创建Excel保存路径
            if logger.log_path:
                excel_save_path = os.path.join(logger.log_path, "classification_report.xlsx")
            print(f"TensorBoard logger initialized. Logs at: {logger.log_path if logger else 'N/A'}")
        except Exception as e:
            print(f"Warning: Failed to initialize TensorBoard logger: {e}")
            logger = None
    # 计算并打印评估指标
    metrics_dict = _calculate_and_print_metrics(all_labels, all_preds)
    # 绘制混淆矩阵
    _calculate_and_plot_confusion_matrix(all_labels, all_preds, class_names, logger)
    # 生成分类报告
    _generate_and_save_classification_report(all_labels, all_preds, class_names, logger, excel_save_path)

    if logger:
        try:
            logger.add_scalar('Test/Loss', test_loss, 0)
            logger.add_scalar('Test/Accuracy', test_acc, 0)
            logger.add_text('Test_Results', f'Loss: {test_loss:.4f} | Accuracy: {test_acc * 100:.2f}%')
            logger.close()
        except Exception as e:
            print(f"Warning: Failed to log standard metrics to TensorBoard: {e}")

    print("\n[Test completed and detailed metrics logged/saved.(测试完成并记录保存详细指标)]", flush=True)
    return test_loss, test_acc, metrics_dict


if __name__ == '__main__':
    print("This script is intended to be imported and used in your main training/testing script.")
