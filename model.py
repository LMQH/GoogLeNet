import torch.nn as nn
from torchvision import models
from torchvision.models import GoogLeNet_Weights


class CustomGoogLeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 加载预训练模型
        # 主干预训练，辅助分支默认随机初始化
        self.model = models.googlenet(
            weights=GoogLeNet_Weights.IMAGENET1K_V1,
            aux_logits=True  # 辅助分类器启用
        )

        # 1. 先冻结所有层（基础操作）
        for param in self.model.parameters():
            param.requires_grad = False

        # 2. 解冻最后几个Inception模块（根据需求调整层数）
        # GoogLeNet的主要特征层在model.features中，包含多个Inception模块
        # 这里以解冻最后3个Inception模块为例（可根据实际情况增减）
        # 注意：不同版本torchvision的层命名可能略有差异，需结合模型结构调整
        for name, param in self.model.named_parameters():
            # 解冻inception4d、inception4e、inception5a、inception5b（较深层）
            if any(layer in name for layer in ['inception4d', 'inception4e', 'inception5a', 'inception5b']):
                param.requires_grad = True

        # 3. 替换并解冻主分类器（全连接层）
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        for param in self.model.fc.parameters():
            param.requires_grad = True

        # 4. 替换并解冻辅助分类器（若启用）
        if self.model.aux_logits:
            # 替换辅助分类器全连接层
            self.model.aux1.fc2 = nn.Linear(1024, num_classes)
            self.model.aux2.fc2 = nn.Linear(1024, num_classes)
            # 初始化权重
            nn.init.xavier_uniform_(self.model.aux1.fc2.weight)
            nn.init.zeros_(self.model.aux1.fc2.bias)
            nn.init.xavier_uniform_(self.model.aux2.fc2.weight)
            nn.init.zeros_(self.model.aux2.fc2.bias)
            # 解冻辅助分类器
            for param in self.model.aux1.parameters():
                param.requires_grad = True
            for param in self.model.aux2.parameters():
                param.requires_grad = True

    # 重写前向传播
    def forward(self, x):
        # 根据模型模式（train/eval）动态处理输出
        if self.model.aux_logits:
            if self.training:
                # 返回顺序是 (main, aux2, aux1)
                main_output, aux1, aux2 = self.model(x)
                return main_output, aux1, aux2
            else:
                # 验证 / 测试时只返回主输出
                main_output = self.model(x)
                return main_output
        else:
            return self.model(x)

    @property
    def fc(self):
        """提供对主分类器的直接访问（供优化器使用）"""
        return self.model.fc


def get_frozen_googlenet(num_classes):
    """创建并返回冻结特征层的GoogleNet模型"""
    model = CustomGoogLeNet(num_classes)
    return model
