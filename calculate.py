from typing import Optional, Dict
import numpy as np
import sklearn.metrics as sk_metrics
from logger import TensorBoardLogger  # 导入日志类
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from typing import Union


def _calculate_and_print_metrics(
    all_labels: np.ndarray,
    all_preds: np.ndarray
) -> Dict[str, Union[float, np.ndarray]]:
    """计算并打印宏平均精确率、召回率、F1分数和整体准确率"""
    print(f"\n[计算并打印宏平均精确率、召回率、F1分数和整体准确率...]")

    # 计算宏平均指标 (macro-average)
    precision_macro = sk_metrics.precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = sk_metrics.recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = sk_metrics.f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # 计算总体准确率 (overall accuracy)
    accuracy = sk_metrics.accuracy_score(all_labels, all_preds)

    # 打印每个指标，每个一行
    print(f"Macro-Average Precision (宏平均精确率): {precision_macro:.4f}")
    print(f"Macro-Average Recall (宏平均召回率):    {recall_macro:.4f}")
    print(f"Macro-Average F1-Score (F1分数):  {f1_macro:.4f}")
    print(f"Overall Accuracy (总体准确率):        {accuracy:.4f}")

    return {
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "accuracy": accuracy
    }


def _calculate_and_plot_confusion_matrix(all_labels: np.ndarray, all_preds: np.ndarray, class_names: list,
                                         logger: Optional[TensorBoardLogger] = None):
    """计算并绘制混淆矩阵，可选地记录到TensorBoard"""
    print(f"\n[Calculating Confusion Matrix...]")
    cm_matrix = sk_metrics.confusion_matrix(all_labels, all_preds)
    print(f"Confusion Matrix (混淆矩阵):\n{cm_matrix}")

    # --- 修正类型警告部分 ---
    # 确保 num_classes 是 int 类型，以消除类型检查器的警告
    num_classes = int(len(class_names))  # len() 返回 int，显式转换加强类型提示

    # 绘制混淆矩阵
    # 提前计算并注解类型
    width: float = max(8.0, num_classes * 0.5)
    height: float = max(6.0, num_classes * 0.5)
    fig, ax = plt.subplots(figsize=(width, height))
    # --- 修正结束 ---

    try:
        cmap = mpl.colormaps.get('Blues', mpl.colormaps['viridis'])
    except AttributeError:
        cmap = plt.get_cmap('Blues') if 'Blues' in plt.colormaps() else plt.get_cmap('viridis')

    # 确保最终得到的是Colormap对象
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]

    cax = ax.matshow(cm_matrix, cmap=cmap)
    plt.colorbar(cax, ax=ax)

    # --- 修正类型警告部分 ---
    # 确保 tick_marks 是整数数组
    if class_names:
        tick_marks = np.arange(num_classes)  # np.arange 接受 int
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
    # --- 修正结束 ---

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')

    thresh = cm_matrix.max() / 2.
    for i in range(cm_matrix.shape[0]):
        for j in range(cm_matrix.shape[1]):
            ax.text(j, i, format(cm_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_matrix[i, j] > thresh else "black")

    fig.tight_layout()

    if logger and hasattr(logger, 'writer'):
        try:
            logger.writer.add_figure('Test/Confusion_Matrix', fig, global_step=0)
            print("Confusion matrix image logged to TensorBoard.")
        except Exception as e:
            print(f"Warning: Failed to log confusion matrix image to TensorBoard: {e}")
    elif logger:
        print("Warning: Provided logger does not have a 'writer' attribute. Confusion matrix image not logged.")

    plt.close(fig)


def _generate_and_save_classification_report(all_labels: np.ndarray, all_preds: np.ndarray, class_names: list,
                                             logger: Optional[TensorBoardLogger] = None,
                                             excel_save_path: Optional[str] = None):
    """生成分类报告，可选地记录到TensorBoard和保存为Excel"""
    print(f"\n[Generating Classification Report (生成分类报告)...]")
    report_str = sk_metrics.classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(f"Classification Report (分类报告如下):\n{report_str}")
    print(f"Excel will be saved to: {excel_save_path}")  # 添加在文件保存前

    if logger:
        try:
            logger.add_text('Test_Classification_Report', report_str)
            print("Classification report text logged to TensorBoard.")
        except Exception as e:
            print(f"Warning: Failed to log classification report text: {e}")

    if excel_save_path:
        try:
            report_dict = sk_metrics.classification_report(
                all_labels, all_preds, target_names=class_names, output_dict=True, digits=4
            )
            # 将字典转换为DataFrame
            df = pd.DataFrame(report_dict).transpose()
            # 保存为Excel
            df.to_excel(excel_save_path, index=True)
            print(f"Classification report saved to Excel: {excel_save_path}")
        except Exception as e:
            print(f"Warning: Failed to save classification report to Excel: {e}")
