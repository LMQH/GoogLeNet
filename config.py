import torch
# from pathlib import Path


# 未下载数据集,可使用kagglehub
# import kagglehub
# path = kagglehub.dataset_download("gpiosenka/cards_image-datasetclassification")
# print("Path to dataset files:", path)  # 查看下载的默认路径


# --------------------------------------
# 超参数配置 Super parameter configuration
# --------------------------------------

# 数据集路径
DATA_ROOT = "../dataset/cards_image/img"

# 自动批处理大小（根据GPU内存）
GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory
BATCH_SIZE = min(128, int(GPU_MEMORY / (1024**3) * 32))  # 经验公式

# 训练相关参数
NUM_EPOCHS = 60               # 训练轮数
DROPOUT_PROB = 0.2            # Dropout概率
PATIENCE = 8                  # 早停耐心值
DELTA = 0.001                 # 早停判断阈值
NUM_WORKERS = 8               # 数据加载工作线程数
LABEL_SMOOTHING = 0.2         # 标签平滑系数

# 优化器参数
LEARNING_RATE = 3e-4          # 学习率
WEIGHT_DECAY = 1e-4           # 权重衰减

# OneCycleLR调度器参数
MAX_LR = LEARNING_RATE * 3    # 最大学习率
PCT_START = 0.5               # 学习率上升阶段比例
DIV_FACTOR = 25               # 初始学习率 = MAX_LR / DIV_FACTOR
FINAL_DIV_FACTOR = 1e4        # 最终学习率 = MAX_LR / FINAL_DIV_FACTOR

# 辅助分类器权重
AUX_WEIGHT = 0.3              # 辅助分类器损失权重

# 模型保存参数
MODEL_SAVE_DIR = "../model"   # 模型保存目录
BEST_MODEL_NAME = "best_googlenet.pth"  # 最佳模型文件名

# 日志参数
LOG_DIR = "../logs"           # TensorBoard日志目录
EXPERIMENT_NAME = "googlenet_card_classification"  # 实验名称
