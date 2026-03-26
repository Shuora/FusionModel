from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fusion_malicious.config import StageConfig
from fusion_malicious.trainers.engine import OOMFallbackPolicy, Trainer


class TinyDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        return {
            "image": torch.randn(3, 28, 28),
            "input_ids": torch.ones(8, dtype=torch.long),
            "attention_mask": torch.ones(8, dtype=torch.long),
            "label": torch.tensor(index % 2, dtype=torch.long),
        }


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3 * 28 * 28 + 8, 2)

    def forward(self, image, input_ids, attention_mask):
        batch = image.size(0)
        flat = image.view(batch, -1)
        joined = torch.cat([flat, input_ids.float()], dim=1)
        return self.linear(joined)


def test_oom_fallback_policy_steps_down_to_partial() -> None:
    policy = OOMFallbackPolicy(current_text_mode="full")
    assert policy.next_mode() == "partial"


def test_trainer_fit_one_epoch_returns_loss_and_acc(tmp_path: Path) -> None:
    model = TinyModel()
    loader = DataLoader(TinyDataset(), batch_size=2)
    trainer = Trainer(
        model,
        torch.optim.Adam(model.parameters(), lr=1e-3),
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        tmp_path,
        [StageConfig(name="warmup", enable_fusion=False, text_train_mode="head_only")],
    )
    metrics = trainer.run_epoch(loader, train_mode=True)
    assert "loss" in metrics
    assert "acc" in metrics
