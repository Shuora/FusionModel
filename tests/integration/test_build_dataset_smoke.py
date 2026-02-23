from pathlib import Path

import yaml


def test_build_dataset_creates_paired_dirs(tmp_path: Path):
    from src.pipeline.build_dataset import main

    cfg_path = tmp_path / "dataset_smoke.yaml"
    cfg = {
        "dataset_name": "smoke_dataset",
        "output_root": str(tmp_path),
        "samples": [{"sample_id": "sample_001"}],
    }
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    main(["--config", str(cfg_path)])

    dataset_root = tmp_path / "smoke_dataset"
    image_data = dataset_root / "image_data"
    pcap_data = dataset_root / "pcap_data"

    assert image_data.is_dir()
    assert pcap_data.is_dir()
    assert (image_data / "sample_001.npy").exists()
    assert (pcap_data / "sample_001.json").exists()
