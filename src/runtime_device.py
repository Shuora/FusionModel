from __future__ import annotations

from typing import Optional, Tuple

import torch


def resolve_runtime_device(requested: Optional[str]) -> Tuple[str, str, bool]:
    requested_name = str(requested or "auto").strip().lower() or "auto"
    if requested_name not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported device: {requested_name}")

    cuda_available = bool(torch.cuda.is_available())
    if requested_name == "cpu":
        return requested_name, "cpu", False
    if requested_name == "cuda":
        if cuda_available:
            return requested_name, "cuda", False
        return requested_name, "cpu", True
    if cuda_available:
        return requested_name, "cuda", False
    return requested_name, "cpu", False
