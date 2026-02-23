from pathlib import Path

import yaml


def test_build_dataset_discovers_samples_from_source_root(tmp_path: Path):
    from src.pipeline.build_dataset import main

    source_root = tmp_path / "SourceData" / "demo"
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "a.pcap").write_bytes(b"pcap-a")
    nested = source_root / "nested"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "b.cap").write_bytes(b"pcap-b")

    cfg_path = tmp_path / "dataset_discovery.yaml"
    cfg = {
        "dataset_name": "discovery_dataset",
        "source_root": str(source_root),
        "output_root": str(tmp_path),
        "log_root": str(tmp_path),
    }
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    main(["--config", str(cfg_path)])

    image_files = list((tmp_path / "discovery_dataset" / "image_data").glob("*.npy"))
    pcap_files = list((tmp_path / "discovery_dataset" / "pcap_data").glob("*.json"))
    assert len(image_files) == 2
    assert len(pcap_files) == 2


def test_build_dataset_supports_multi_config_progress(tmp_path: Path, capsys):
    from src.pipeline.build_dataset import main

    cfg1_path = tmp_path / "ds1.yaml"
    cfg2_path = tmp_path / "ds2.yaml"

    cfg1 = {
        "dataset_name": "dataset_one",
        "output_root": str(tmp_path),
        "log_root": str(tmp_path),
        "samples": [{"sample_id": "s1"}],
    }
    cfg2 = {
        "dataset_name": "dataset_two",
        "output_root": str(tmp_path),
        "log_root": str(tmp_path),
        "samples": [{"sample_id": "s2"}],
    }

    cfg1_path.write_text(yaml.safe_dump(cfg1), encoding="utf-8")
    cfg2_path.write_text(yaml.safe_dump(cfg2), encoding="utf-8")

    main(["--config", str(cfg1_path), "--config", str(cfg2_path)])

    out = capsys.readouterr().out
    assert "dataset=dataset_one" in out
    assert "dataset=dataset_two" in out
    assert (tmp_path / "logs" / "preprocess" / "dataset_one.log").exists()
    assert (tmp_path / "logs" / "preprocess" / "dataset_two.log").exists()
