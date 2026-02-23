from __future__ import annotations

import torch
import torch.nn.functional as F


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.7,
) -> torch.Tensor:
    """Blend hard-label CE and soft-target KL distillation loss."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    hard_loss = F.cross_entropy(student_logits, labels)

    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
    soft_loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temperature**2)

    return alpha * soft_loss + (1.0 - alpha) * hard_loss
