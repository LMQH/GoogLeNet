from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, Any, Union
from matplotlib.figure import Figure  # 显式导入


class TensorBoardLogger:
    def __init__(
            self,
            log_dir: Union[str, Path],
            experiment_name: str,
            subdir: str = "",
            max_queue: int = 10,  # 新增队列大小参数
            flush_secs: int = 120  # 新增刷新间隔

    ):
        """
        初始化TensorBoard日志记录器
        """
        self.log_dir = Path(log_dir) if isinstance(log_dir, str) else log_dir
        self.experiment_name = str(experiment_name)
        self.subdir = subdir
        self.max_queue = max_queue
        self.flush_secs = flush_secs
        self.writer: Optional[SummaryWriter] = None
        self.log_path: Optional[Path] = None
        self._init_writer()

    def _init_writer(self) -> None:
        """初始化SummaryWriter"""
        if not self.log_dir:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_components = [self.log_dir]
        if self.subdir:
            dir_components.append(self.subdir)
        dir_components.append(f"{self.experiment_name}_{timestamp}")

        # 创建目录
        self.log_path = Path(*dir_components)
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(
            log_dir=str(self.log_path),
            max_queue=self.max_queue,
            flush_secs=self.flush_secs
        )
        print(f"TensorBoard logs saved to: {self.log_path}")
        print(f"Start TensorBoard with: tensorboard --logdir={self.log_path}")

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        """记录标量值"""
        if self.writer:
            self.writer.add_scalar(str(tag), value, step)

    def add_figure(
            self,
            tag: str,
            figure: Figure,
            global_step: Optional[int] = None,
            close: bool = True
    ) -> None:
        """添加matplotlib图像"""
        if self.writer:
            self.writer.add_figure(
                tag=tag,
                figure=figure,
                global_step=global_step,
                close=close
            )

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
