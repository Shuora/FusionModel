from src.experiments.stage2_multiclass import build_stage2_tasks


def test_stage2_tasks_are_fixed():
    tasks = build_stage2_tasks()
    assert tasks == [
        {"dataset": "MTA", "num_classes": 7},
        {"dataset": "MFCP", "num_classes": 6},
        {"dataset": "USTC-TFC2016", "num_classes": 10},
    ]

