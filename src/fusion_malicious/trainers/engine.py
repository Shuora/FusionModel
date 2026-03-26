"""Minimal stage-aware training utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn
from tqdm.auto import tqdm

from fusion_malicious.config import StageConfig


class OOMFallbackPolicy:
    """Progressively degrade the text train mode when CUDA OOMs occur."""

    _ORDER = ["full", "partial", "head_only"]

    def __init__(self, current_text_mode: str) -> None:
        self._current_text_mode = current_text_mode

    @property
    def current_text_mode(self) -> str:
        return self._current_text_mode

    def next_mode(self) -> str:
        try:
            current_index = self._ORDER.index(self._current_text_mode)
        except ValueError:
            current_index = -1
        next_index = min(current_index + 1, len(self._ORDER) - 1)
        self._current_text_mode = self._ORDER[next_index]
        return self._current_text_mode


class Trainer:
    """Skeleton trainer that supports simple stages and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        run_dir: Path | str,
        stages: Sequence[StageConfig],
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.scaler: torch.cuda.amp.GradScaler | None = (
            torch.cuda.amp.GradScaler() if device.type == "cuda" else None
        )
        self.stages = list(stages)
        self.current_stage_index = 0
        self._active_text_mode: str | None = None
        self._oom_policy: OOMFallbackPolicy | None = None
        self._sync_stage_mode()

    @property
    def current_stage(self) -> StageConfig | None:
        if not self.stages:
            return None
        return self.stages[self.current_stage_index]

    @property
    def active_text_mode(self) -> str | None:
        if self._active_text_mode:
            return self._active_text_mode
        stage = self.current_stage
        return stage.text_train_mode if stage else None

    def run_epoch(self, loader: Iterable[dict], train_mode: bool) -> dict[str, float]:
        """Execute a single epoch of train/eval and collect loss/accuracy."""
        self._sync_stage_mode()
        self.model.train(train_mode)
        total_loss = 0.0
        correct = 0
        total = 0
        steps = 0
        progress = tqdm(
            loader, desc="train" if train_mode else "eval", leave=False, unit="batch"
        )

        for batch in progress:
            tensors = {
                key: value.to(self.device)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            labels = tensors.get("label")
            if labels is None:
                raise KeyError("Batch must include 'label'.")

            try:
                with torch.set_grad_enabled(train_mode), torch.amp.autocast(
                    device_type=self.device.type, enabled=self.scaler is not None
                ):
                    logits = self.model(
                        tensors.get("image"),
                        tensors.get("input_ids"),
                        tensors.get("attention_mask"),
                    )
                    loss = self.criterion(logits, labels)

                if train_mode:
                    self.optimizer.zero_grad()
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    self._handle_oom()
                    continue
                raise

            total_loss += float(loss.item())
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += labels.size(0)
            steps += 1
            progress.set_postfix(
                loss=total_loss / steps, acc=(correct / total if total else 0.0)
            )

        avg_loss = total_loss / steps if steps else 0.0
        accuracy = correct / total if total else 0.0
        return {"loss": avg_loss, "acc": accuracy}

    def save_checkpoint(self, model_name: str) -> Path:
        """Save the current model weights and return the checkpoint path."""
        checkpoint_path = self.run_dir / f"{model_name}.pt"
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def advance_stage(self) -> None:
        """Move to the next stage if available."""
        if self.current_stage_index + 1 < len(self.stages):
            self.current_stage_index += 1
            self._sync_stage_mode()

    def _handle_oom(self) -> None:
        """Handle CUDA OOM by downgrading text mode and clearing cache."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        next_mode = (
            self._oom_policy.next_mode() if self._oom_policy else "partial"
        )
        stage = self.current_stage
        if stage:
            self.stages[self.current_stage_index] = StageConfig(
                name=stage.name,
                enable_fusion=stage.enable_fusion,
                text_train_mode=next_mode,
            )
        self._sync_stage_mode()

    def _sync_stage_mode(self) -> None:
        """Refresh the tracked text mode and OOM policy for the current stage."""
        stage = self.current_stage
        if stage:
            self._active_text_mode = stage.text_train_mode
            self._oom_policy = OOMFallbackPolicy(stage.text_train_mode)
        else:
            self._active_text_mode = None
            self._oom_policy = OOMFallbackPolicy("full")
