from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class TrainingConfig:
    # 数据与输出
    data_root: Path = Path("pcap_data")
    train_dir: Path = Path("pcap_data/Train")
    test_dir: Path = Path("pcap_data/Test")
    output_dir: Path = Path("outputs")
    log_file: str = "training_charBERT.log"
    eval_only = True  # 仅评估
    checkpoint = Path("outputs/best_model.pt")  # 或者指定其他已有模型

    # 训练超参
    epochs: int = 32
    start_epoch: int = 1
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_len: int = 512
    num_workers: int = 8
    seed: int = 42

    # 模型尺寸
    vocab_size: int = 259  # 256 字节 + PAD/CLS/SEP
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1

    # 训练设备
    device: str = "cuda"

    # 类别标签（如不提供则按目录自动生成）
    label_list: List[str] = field(default_factory=list)

    def resolve_paths(self, base: Path):
        if not self.train_dir.is_absolute():
            self.train_dir = base / self.train_dir
        if not self.test_dir.is_absolute():
            self.test_dir = base / self.test_dir
        if not self.output_dir.is_absolute():
            self.output_dir = base / self.output_dir
        if self.checkpoint and not self.checkpoint.is_absolute():
            self.checkpoint = base / self.checkpoint
        return self

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
