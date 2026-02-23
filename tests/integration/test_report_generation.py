from pathlib import Path


def test_report_contains_required_sections(tmp_path: Path):
    from src.fusion.report import generate_report

    metrics = {"acc": 0.91, "macro_f1": 0.87}
    figures = {
        "confusion_matrix": "figures/confusion_matrix_smoke.png",
        "metrics_curve": "figures/metrics_curve_smoke.png",
    }

    output_path = tmp_path / "report.md"
    generate_report(metrics, figures, output_path)

    text = output_path.read_text(encoding="utf-8")
    assert "## 实验信息" in text
    assert "## 指标" in text
    assert "## 混淆" in text
    assert "## 错分分析" in text
