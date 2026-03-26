from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from fusion_malicious.utils.reporting import _resolve_class_values


def _ensure_parent_path(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)


def save_curve(
    values: Sequence[float],
    title: str,
    ylabel: str,
    output_path: Union[Path, str],
) -> None:
    """Persist a simple line plot tracking the provided metric values."""
    output_path = Path(output_path)
    _ensure_parent_path(output_path)
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(values) + 1), list(values), marker="o", linewidth=2)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.grid(True, which="both", linestyle="--", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_confusion_matrix(
    targets: Sequence[Union[int, str]],
    predictions: Sequence[Union[int, str]],
    labels: Sequence[str],
    output_path: Union[Path, str],
    class_values: Sequence[Union[int, str]] | None = None,
) -> None:
    """Render a confusion matrix heatmap labeled with the given class names."""
    if not labels:
        raise ValueError("At least one label name is required for the confusion matrix.")
    output_path = Path(output_path)
    class_ids = _resolve_class_values(targets, predictions, labels, class_values=class_values)
    cm = confusion_matrix(targets, predictions, labels=list(class_ids))
    _ensure_parent_path(output_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(list(labels))
    ax.set_yticklabels(list(labels))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    max_value = int(cm.max()) if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            text_color = "white" if max_value and value > max_value / 2 else "black"
            ax.text(j, i, f"{value}", ha="center", va="center", color=text_color)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
