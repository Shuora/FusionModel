from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

from sklearn.metrics import classification_report


def _resolve_class_values(
    targets: Sequence[Union[int, str]],
    predictions: Sequence[Union[int, str]],
    labels: Sequence[str],
    class_values: Sequence[Union[int, str]] | None = None,
) -> Sequence[Union[int, str]]:
    """Determine the class identifiers that back the provided display names."""
    if not labels:
        raise ValueError("At least one label name is required to format the report.")
    if class_values is not None:
        if len(class_values) != len(labels):
            raise ValueError(
                "class_values must match the number of provided label names."
            )
        return list(class_values)
    observed_classes = sorted(set(targets) | set(predictions))
    if len(observed_classes) == len(labels):
        return observed_classes
    if not observed_classes:
        return list(range(len(labels)))
    if all(isinstance(value, int) for value in observed_classes):
        derived = list(range(len(labels)))
        if all(value in derived for value in observed_classes):
            return derived
    raise ValueError(
        "Label names must match the observed classes or provide the same total count when class indices are implied."
    )


def write_classification_report(
    targets: Sequence[Union[int, str]],
    predictions: Sequence[Union[int, str]],
    labels: Sequence[str],
    output_path: Union[Path, str],
    class_values: Sequence[Union[int, str]] | None = None,
) -> None:
    """Write a sklearn-style classification report to a text file."""
    class_ids = _resolve_class_values(targets, predictions, labels, class_values=class_values)
    report = classification_report(
        targets,
        predictions,
        labels=list(class_ids),
        target_names=list(labels),
        zero_division=0,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
