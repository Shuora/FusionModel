from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(
    targets: Sequence[Union[int, str]],
    predictions: Sequence[Union[int, str]],
    probabilities: Optional[Sequence[float]] = None,
) -> Mapping[str, float]:
    """Compute common metrics for binary classification runs."""
    target_list = list(targets)
    prediction_list = list(predictions)
    metrics = {
        "acc": accuracy_score(target_list, prediction_list),
        "precision": precision_score(target_list, prediction_list, zero_division=0),
        "recall": recall_score(target_list, prediction_list, zero_division=0),
        "f1": f1_score(target_list, prediction_list, zero_division=0),
        "balanced_acc": balanced_accuracy_score(target_list, prediction_list),
    }
    unique_targets = set(target_list)
    if probabilities is not None and len(probabilities) == len(target_list):
        if len(unique_targets) > 1:
            metrics["roc_auc"] = roc_auc_score(target_list, list(probabilities))
    return metrics


def compute_multiclass_metrics(
    targets: Sequence[Union[int, str]],
    predictions: Sequence[Union[int, str]],
) -> Mapping[str, float]:
    """Compute accuracy and averaged F1 scores for multiclass problems."""
    target_list = list(targets)
    prediction_list = list(predictions)
    return {
        "acc": accuracy_score(target_list, prediction_list),
        "macro_precision": precision_score(target_list, prediction_list, average="macro", zero_division=0),
        "macro_recall": recall_score(target_list, prediction_list, average="macro", zero_division=0),
        "macro_f1": f1_score(target_list, prediction_list, average="macro", zero_division=0),
        "weighted_f1": f1_score(target_list, prediction_list, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(target_list, prediction_list),
    }
