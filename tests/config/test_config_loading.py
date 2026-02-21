def test_dataset_profile_has_required_keys():
    from src.common.config import load_yaml

    cfg = load_yaml("configs/dataset_tls_full.yaml")
    required = {"dataset_name", "source_root", "split", "tls_filter", "feature"}
    assert required.issubset(set(cfg.keys()))
