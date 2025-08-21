import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, Any


class TensorBoardLogger:
    def __init__(self, log_dir: str, experiment_name: str, subdir: str = ""):
        """
        初始化TensorBoard日志记录器
        """
        self.log_dir = str(log_dir)
        self.experiment_name = str(experiment_name)
        self.subdir = str(subdir)
        self.writer: Optional[SummaryWriter] = None
        self.log_path: str = ""
        self._init_writer()

    def _init_writer(self):
        """初始化SummaryWriter，创建带时间戳的日志目录"""
        if not self.log_dir:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_components = [self.log_dir, f"{self.experiment_name}_{timestamp}"]
        if self.subdir:
            dir_components.insert(1, self.subdir)

            # 显式转换为str类型，解决类型注解警告
            self.log_path = str(os.path.join(*map(str, dir_components)))
            os.makedirs(self.log_path, exist_ok=True)
            # 传入前再次确认类型为str
            self.writer = SummaryWriter(log_dir=str(self.log_path))
            print(f"TensorBoard logs saved to: {self.log_path}")
            print(f"Start TensorBoard with: tensorboard --logdir={self.log_path}")

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        """记录标量值"""
        if self.writer:
            self.writer.add_scalar(str(tag), value, step)

    def add_text(self, tag: str, text: str, step: Optional[int] = None) -> None:
        """记录文本信息"""
        if self.writer:
            self.writer.add_text(str(tag), str(text), step)

    def add_hparams(self, hparams: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        """记录超参数"""
        if self.writer:
            # 确保hparams的键值为字符串类型
            str_hparams = {str(k): v for k, v in hparams.items()}
            str_metrics = {str(k): v for k, v in metrics.items()}
            self.writer.add_hparams(str_hparams, str_metrics)

    def close(self) -> None:
        """关闭日志写入器"""
        if self.writer:
            self.writer.close()
            self.writer = None
