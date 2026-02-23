def test_group_split_no_capture_overlap():
    from src.pipeline.split_strategy import group_split_by_capture

    rows = [
        {"capture_id": "a", "label": "fam1"},
        {"capture_id": "a", "label": "fam1"},
        {"capture_id": "b", "label": "fam2"},
    ]
    train, val, test = group_split_by_capture(rows, seed=42)
    assert set(x["capture_id"] for x in train).isdisjoint(
        set(x["capture_id"] for x in test)
    )
