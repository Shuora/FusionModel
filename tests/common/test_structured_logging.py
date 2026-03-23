from src.common.structured_logging import format_log_line


def test_format_log_line_contains_icons_and_fields():
    line = format_log_line(
        level="success",
        module="data",
        event="dataset_loaded",
        kv={"samples": 123, "families": 10},
        timestamp="2026-02-23 14:30:00",
    )

    assert "2026-02-23 14:30:00" in line
    assert "✅SUCCESS" in line
    assert "🧱 Data" in line
    assert "dataset_loaded" in line
    assert "samples=123" in line
    assert "families=10" in line


def test_format_log_line_uses_english_time_module_label():
    line = format_log_line(
        level="info",
        module="time",
        event="train_start",
        timestamp="2026-02-23 14:32:00",
    )

    assert "⏱️ Time" in line


def test_format_log_line_unknown_level_module_fallback():
    line = format_log_line(
        level="mystery",
        module="custom",
        event="custom_event",
        kv={"k": "v"},
        timestamp="2026-02-23 14:31:00",
    )

    assert "ℹ️INFO" in line
    assert "🧩 Module:custom" in line
    assert "k=v" in line
