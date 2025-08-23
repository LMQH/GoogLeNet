card_classification_project/
│
├── data/                          # 数据相关
│   ├── __init__.py
│   └── dataloader.py              # get_card_dataloaders()
│
├── models/                        # 模型定义
│   ├── __init__.py
│   └── GoogLeNet.py               # CustomGoogLeNet 类
│
├── training/                      # 训练逻辑
│   ├── __init__.py
│   ├── train.py                   # train_model()
│   ├── early_stop.py              # EarlyStopping 类
│   └── evaluate.py                # _evaluate_model()
│
├── testing/                       # 测试评估
│   ├── __init__.py
│   ├── test.py                    # test_model()
│   └── inference.py               # ONNX推理脚本
│
├── utils/                         # 工具函数
│   ├── __init__.py
│   ├── metrics.py                 # 评估指标计算
│   └── logger.py                  # TensorBoardLogger
│
├── main.py
├── config.py                      # 配置文件
├── __init__.py
└── README.txt



dataset/                           # 实际数据存放目录
│
└── cards_image/
        └── img/
            ├── train/             # 训练集
            ├── valid/             # 验证集
            └── test/              # 测试集