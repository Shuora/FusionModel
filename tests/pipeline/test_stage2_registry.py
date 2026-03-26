from src.stage2_registry import ACCEPTANCE_GATES, STAGE2_DATASET_ORDER, dataset_num_classes


def test_stage2_dataset_order_and_num_classes_are_stable():
    assert STAGE2_DATASET_ORDER == ("MTA", "MFCP", "USTC-TFC2016")
    assert dataset_num_classes("MTA") == 7
    assert dataset_num_classes("MFCP") == 6
    assert dataset_num_classes("USTC-TFC2016") == 10
    assert ACCEPTANCE_GATES["MTA"]["test_top1_min"] == 0.70
    assert ACCEPTANCE_GATES["USTC-TFC2016"]["test_top1_min"] == 0.86
