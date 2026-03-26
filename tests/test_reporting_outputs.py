from pathlib import Path

from fusion_malicious.utils.logging import append_metrics_row, create_logger
from fusion_malicious.utils.metrics import compute_multiclass_metrics
from fusion_malicious.utils.plots import save_confusion_matrix
from fusion_malicious.utils.reporting import write_classification_report


def test_compute_multiclass_metrics_returns_macro_and_weighted_f1() -> None:
    metrics = compute_multiclass_metrics(
        targets=[0, 1, 1, 2],
        predictions=[0, 1, 0, 2],
    )
    assert "acc" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics


def test_write_classification_report_saves_text(tmp_path: Path) -> None:
    report_path = tmp_path / "classification_report.txt"
    write_classification_report(
        targets=[0, 1, 1, 2],
        predictions=[0, 1, 0, 2],
        labels=["A", "B", "C"],
        output_path=report_path,
    )
    assert report_path.exists()
    assert "precision" in report_path.read_text()


def test_append_metrics_row_keeps_schema(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    append_metrics_row(csv_path, {"loss": 0.5, "acc": 0.9})
    append_metrics_row(csv_path, {"acc": 0.92, "precision": 0.8})
    lines = csv_path.read_text().splitlines()
    assert lines[0].split(",") == ["loss", "acc", "precision"]
    assert lines[1].split(",") == ["0.5", "0.9", ""]
    assert lines[2].split(",") == ["", "0.92", "0.8"]


def test_create_logger_isolated_by_path(tmp_path: Path) -> None:
    log_dir = tmp_path
    log_a = log_dir / "group_a" / "metrics.log"
    log_b = log_dir / "group_b" / "metrics.log"
    logger_a = create_logger(log_a)
    logger_b = create_logger(log_b)
    logger_a.info("first message")
    logger_b.info("second message")
    for handler in logger_a.handlers + logger_b.handlers:
        handler.flush()
    assert "first message" in log_a.read_text()
    assert "second message" in log_b.read_text()


def test_write_classification_report_includes_missing_classes(tmp_path: Path) -> None:
    report_path = tmp_path / "full_class_report.txt"
    write_classification_report(
        targets=[0, 2, 2],
        predictions=[2, 2, 0],
        labels=["A", "B", "C"],
        output_path=report_path,
    )
    content = report_path.read_text()
    assert report_path.exists()
    assert "B" in content


def test_save_confusion_matrix_handles_missing_classes(tmp_path: Path) -> None:
    cm_path = tmp_path / "confusion.png"
    save_confusion_matrix(
        targets=[0, 2, 0],
        predictions=[2, 2, 0],
        labels=["A", "B", "C"],
        output_path=cm_path,
    )
    assert cm_path.exists()
