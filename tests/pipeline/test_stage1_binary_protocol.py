from pathlib import Path

import pandas as pd
import pytest

import src.experiments.stage1_binary as stage1_module
from src.experiments.stage1_binary import build_stage1_manifest, main


def _write_manifest(root: Path, dataset: str, policy: str, rows: list[dict]) -> None:
    manifest_dir = root / dataset / policy / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest_dir / "session_manifest.csv", index=False)


def _patch_minimal_paper_specs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    iscx_train: int = 1,
    iscx_test: int = 0,
    mta_train: int = 1,
    mta_test: int = 0,
    mfcp_train: int = 1,
    mfcp_test: int = 0,
) -> None:
    monkeypatch.setattr(
        stage1_module,
        "PAPER_STAGE1_ISCX_SPECS",
        [{"name": "vpn_facebook_chat", "capture_prefixes": ("vpn_facebook_chat",), "train": iscx_train, "test": iscx_test}],
    )
    monkeypatch.setattr(
        stage1_module,
        "PAPER_STAGE1_MTA_SPECS",
        [{"family": "Dridex", "train": mta_train, "test": mta_test}],
    )
    monkeypatch.setattr(
        stage1_module,
        "PAPER_STAGE1_MFCP_SPECS",
        [{"family": "PUA", "train": mfcp_train, "test": mfcp_test}],
    )


def test_stage1_requires_all_datasets(tmp_path: Path):
    processed_root = tmp_path / "processed"
    policy = "session_full"
    _write_manifest(
        processed_root,
        "MFCP",
        policy,
        [
            {
                "session_id": "mfcp_1",
                "dataset": "MFCP",
                "family": "F1",
                "capture_id": "c1.pcap",
                "split": "train",
                "policy": policy,
            }
        ],
    )
    with pytest.raises(FileNotFoundError):
        build_stage1_manifest(processed_root=processed_root, policy=policy)


def test_stage1_label_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    processed_root = tmp_path / "processed"
    policy = "session_full"
    _patch_minimal_paper_specs(monkeypatch)
    datasets = ["ISCX", "MFCP", "MTA"]
    for dataset in datasets:
        family = "F1"
        capture_id = "c1.pcap"
        if dataset == "ISCX":
            capture_id = "vpn_facebook_chat1a.pcap"
        elif dataset == "MFCP":
            family = "PUA"
        elif dataset == "MTA":
            family = "Dridex"
        _write_manifest(
            processed_root,
            dataset,
            policy,
            [
                {
                    "session_id": f"{dataset}_1",
                    "dataset": dataset,
                    "family": family,
                    "capture_id": capture_id,
                    "split": "train",
                    "policy": policy,
                }
            ],
        )

    manifest = build_stage1_manifest(processed_root=processed_root, policy=policy, protocol_mode="paper_strict")
    assert set(manifest["label_binary"].unique().tolist()) == {0, 1}
    assert (manifest[manifest["dataset"] == "ISCX"]["label_binary"] == 0).all()
    assert (manifest[manifest["dataset"] != "ISCX"]["label_binary"] == 1).all()


def test_stage1_accepts_iscx_alias_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    processed_root = tmp_path / "processed"
    policy = "session_full"
    _patch_minimal_paper_specs(monkeypatch)
    _write_manifest(
        processed_root,
        "ISCX-VPN-NonVPN-2016",
        policy,
        [
            {
                "session_id": "iscx_alias_1",
                "dataset": "ISCX-VPN-NonVPN-2016",
                "family": "F1",
                "capture_id": "vpn_facebook_chat1a.pcap",
                "split": "train",
                "policy": policy,
            }
        ],
    )
    for dataset in ("MFCP", "MTA"):
        _write_manifest(
            processed_root,
            dataset,
            policy,
            [
                {
                    "session_id": f"{dataset}_1",
                    "dataset": dataset,
                    "family": "PUA" if dataset == "MFCP" else "Dridex",
                    "capture_id": "c1.pcap",
                    "split": "train",
                    "policy": policy,
                }
            ],
        )

    manifest = build_stage1_manifest(processed_root=processed_root, policy=policy, protocol_mode="paper_strict")
    assert "ISCX" in set(manifest["dataset"].unique())
    assert (manifest[manifest["dataset"] == "ISCX"]["label_binary"] == 0).all()


def test_stage1_prefers_primary_iscx_directory_when_both_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    processed_root = tmp_path / "processed"
    policy = "session_full"
    _patch_minimal_paper_specs(monkeypatch)
    _write_manifest(
        processed_root,
        "ISCX",
        policy,
        [
            {
                "session_id": "iscx_primary_1",
                "dataset": "ISCX",
                "family": "F1",
                "capture_id": "vpn_facebook_chat1a.pcap",
                "split": "train",
                "policy": policy,
            }
        ],
    )
    _write_manifest(
        processed_root,
        "ISCX-VPN-NonVPN-2016",
        policy,
        [
            {
                "session_id": "iscx_alias_1",
                "dataset": "ISCX-VPN-NonVPN-2016",
                "family": "F1",
                "capture_id": "vpn_facebook_chat1b.pcap",
                "split": "train",
                "policy": policy,
            }
        ],
    )
    for dataset in ("MFCP", "MTA"):
        _write_manifest(
            processed_root,
            dataset,
            policy,
            [
                {
                    "session_id": f"{dataset}_1",
                    "dataset": dataset,
                    "family": "PUA" if dataset == "MFCP" else "Dridex",
                    "capture_id": "c1.pcap",
                    "split": "train",
                    "policy": policy,
                }
            ],
        )

    manifest = build_stage1_manifest(processed_root=processed_root, policy=policy, protocol_mode="paper_strict")
    iscx_rows = manifest[manifest["dataset"] == "ISCX"]
    assert "iscx_primary_1" in set(iscx_rows["session_id"].tolist())
    assert "iscx_alias_1" not in set(iscx_rows["session_id"].tolist())
    assert (iscx_rows["dataset_raw"] == "ISCX").all()


def test_stage1_rejects_empty_required_dataset_manifest(tmp_path: Path):
    processed_root = tmp_path / "processed"
    policy = "session_full"
    for dataset in ("ISCX", "MFCP", "MTA"):
        _write_manifest(processed_root, dataset, policy, [])

    with pytest.raises((FileNotFoundError, ValueError)):
        build_stage1_manifest(processed_root=processed_root, policy=policy)


def test_stage1_does_not_require_ustc_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    processed_root = tmp_path / "processed"
    policy = "session_full"
    _patch_minimal_paper_specs(monkeypatch)
    for dataset in ("ISCX", "MFCP", "MTA"):
        family = "F1"
        capture_id = "c1.pcap"
        if dataset == "ISCX":
            capture_id = "vpn_facebook_chat1a.pcap"
        elif dataset == "MFCP":
            family = "PUA"
        elif dataset == "MTA":
            family = "Dridex"
        _write_manifest(
            processed_root,
            dataset,
            policy,
            [
                {
                    "session_id": f"{dataset}_1",
                    "dataset": dataset,
                    "family": family,
                    "capture_id": capture_id,
                    "split": "train",
                    "policy": policy,
                }
            ],
        )
    manifest = build_stage1_manifest(processed_root=processed_root, policy=policy, protocol_mode="paper_strict")
    assert len(manifest) == 3


def test_stage1_manifest_filters_to_mvtba_paper_subset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    processed_root = tmp_path / "processed"
    policy = "session_full"
    _patch_minimal_paper_specs(monkeypatch)

    _write_manifest(
        processed_root,
        "ISCX",
        policy,
        [
            {
                "session_id": "iscx_keep",
                "dataset": "ISCX",
                "family": "F1",
                "capture_id": "vpn_facebook_chat1a.pcap",
                "split": "train",
                "policy": policy,
            },
            {
                "session_id": "iscx_drop",
                "dataset": "ISCX",
                "family": "F1",
                "capture_id": "vpn_bittorrent.pcap",
                "split": "train",
                "policy": policy,
            },
        ],
    )
    _write_manifest(
        processed_root,
        "MFCP",
        policy,
        [
            {
                "session_id": "mfcp_keep",
                "dataset": "MFCP",
                "family": "PUA",
                "capture_id": "a.pcap",
                "split": "train",
                "policy": policy,
            },
            {
                "session_id": "mfcp_drop",
                "dataset": "MFCP",
                "family": "Zeus",
                "capture_id": "b.pcap",
                "split": "train",
                "policy": policy,
            },
        ],
    )
    _write_manifest(
        processed_root,
        "MTA",
        policy,
        [
            {
                "session_id": "mta_keep",
                "dataset": "MTA",
                "family": "Dridex",
                "capture_id": "c.pcap",
                "split": "train",
                "policy": policy,
            }
        ],
    )

    manifest = build_stage1_manifest(processed_root=processed_root, policy=policy, protocol_mode="paper_strict")
    assert set(manifest["session_id"].tolist()) == {"iscx_keep", "mfcp_keep", "mta_keep"}


def test_stage1_manifest_applies_exact_paper_quotas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    processed_root = tmp_path / "processed"
    policy = "session_full"

    monkeypatch.setattr(
        stage1_module,
        "PAPER_STAGE1_ISCX_SPECS",
        [
            {
                "name": "email_nonvpn",
                "capture_prefixes": ("email",),
                "train": 2,
                "test": 1,
            }
        ],
        raising=False,
    )
    monkeypatch.setattr(
        stage1_module,
        "PAPER_STAGE1_MTA_SPECS",
        [
            {
                "family": "Dridex",
                "train": 1,
                "test": 1,
            }
        ],
        raising=False,
    )
    monkeypatch.setattr(
        stage1_module,
        "PAPER_STAGE1_MFCP_SPECS",
        [
            {
                "family": "PUA",
                "train": 2,
                "test": 1,
            }
        ],
        raising=False,
    )

    _write_manifest(
        processed_root,
        "ISCX",
        policy,
        [
            {"session_id": "iscx_train_2", "dataset": "ISCX", "family": "F1", "capture_id": "email_b.pcap", "split": "train", "policy": policy},
            {"session_id": "iscx_train_1", "dataset": "ISCX", "family": "F1", "capture_id": "email_a.pcap", "split": "train", "policy": policy},
            {"session_id": "iscx_train_3", "dataset": "ISCX", "family": "F1", "capture_id": "email_c.pcap", "split": "train", "policy": policy},
            {"session_id": "iscx_test_2", "dataset": "ISCX", "family": "F1", "capture_id": "email_e.pcap", "split": "test", "policy": policy},
            {"session_id": "iscx_test_1", "dataset": "ISCX", "family": "F1", "capture_id": "email_d.pcap", "split": "test", "policy": policy},
        ],
    )
    _write_manifest(
        processed_root,
        "MTA",
        policy,
        [
            {"session_id": "mta_train_2", "dataset": "MTA", "family": "Dridex", "capture_id": "d2.pcap", "split": "train", "policy": policy},
            {"session_id": "mta_train_1", "dataset": "MTA", "family": "Dridex", "capture_id": "d1.pcap", "split": "train", "policy": policy},
            {"session_id": "mta_test_2", "dataset": "MTA", "family": "Dridex", "capture_id": "d4.pcap", "split": "test", "policy": policy},
            {"session_id": "mta_test_1", "dataset": "MTA", "family": "Dridex", "capture_id": "d3.pcap", "split": "test", "policy": policy},
        ],
    )
    _write_manifest(
        processed_root,
        "MFCP",
        policy,
        [
            {"session_id": "mfcp_train_2", "dataset": "MFCP", "family": "PUA", "capture_id": "p2.pcap", "split": "train", "policy": policy},
            {"session_id": "mfcp_train_1", "dataset": "MFCP", "family": "PUA", "capture_id": "p1.pcap", "split": "train", "policy": policy},
            {"session_id": "mfcp_train_3", "dataset": "MFCP", "family": "PUA", "capture_id": "p3.pcap", "split": "train", "policy": policy},
            {"session_id": "mfcp_test_2", "dataset": "MFCP", "family": "PUA", "capture_id": "p5.pcap", "split": "test", "policy": policy},
            {"session_id": "mfcp_test_1", "dataset": "MFCP", "family": "PUA", "capture_id": "p4.pcap", "split": "test", "policy": policy},
        ],
    )

    manifest = build_stage1_manifest(processed_root=processed_root, policy=policy, protocol_mode="paper_strict")
    assert set(manifest["session_id"].tolist()) == {
        "iscx_train_1",
        "iscx_train_2",
        "iscx_test_1",
        "mta_train_1",
        "mta_test_1",
        "mfcp_train_1",
        "mfcp_train_2",
        "mfcp_test_1",
    }


def test_stage1_manifest_raises_when_paper_quota_is_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    processed_root = tmp_path / "processed"
    policy = "session_full"

    monkeypatch.setattr(
        stage1_module,
        "PAPER_STAGE1_ISCX_SPECS",
        [{"name": "email_nonvpn", "capture_prefixes": ("email",), "train": 1, "test": 1}],
        raising=False,
    )
    monkeypatch.setattr(
        stage1_module,
        "PAPER_STAGE1_MTA_SPECS",
        [{"family": "Dridex", "train": 2, "test": 1}],
        raising=False,
    )
    monkeypatch.setattr(
        stage1_module,
        "PAPER_STAGE1_MFCP_SPECS",
        [{"family": "PUA", "train": 1, "test": 1}],
        raising=False,
    )

    _write_manifest(
        processed_root,
        "ISCX",
        policy,
        [
            {"session_id": "iscx_train", "dataset": "ISCX", "family": "F1", "capture_id": "email_a.pcap", "split": "train", "policy": policy},
            {"session_id": "iscx_test", "dataset": "ISCX", "family": "F1", "capture_id": "email_b.pcap", "split": "test", "policy": policy},
        ],
    )
    _write_manifest(
        processed_root,
        "MTA",
        policy,
        [
            {"session_id": "mta_train", "dataset": "MTA", "family": "Dridex", "capture_id": "d1.pcap", "split": "train", "policy": policy},
            {"session_id": "mta_test", "dataset": "MTA", "family": "Dridex", "capture_id": "d2.pcap", "split": "test", "policy": policy},
        ],
    )
    _write_manifest(
        processed_root,
        "MFCP",
        policy,
        [
            {"session_id": "mfcp_train", "dataset": "MFCP", "family": "PUA", "capture_id": "p1.pcap", "split": "train", "policy": policy},
            {"session_id": "mfcp_test", "dataset": "MFCP", "family": "PUA", "capture_id": "p2.pcap", "split": "test", "policy": policy},
        ],
    )

    with pytest.raises(ValueError, match="Dridex"):
        build_stage1_manifest(processed_root=processed_root, policy=policy, protocol_mode="paper_strict")


def test_stage1_manifest_balanced_skips_missing_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    processed_root = tmp_path / "processed"
    policy = "session_full"

    monkeypatch.setattr(
        stage1_module,
        "PAPER_STAGE1_ISCX_SPECS",
        [
            {"name": "present_group", "capture_prefixes": ("email",), "train": 2, "test": 1},
            {"name": "missing_group", "capture_prefixes": ("torrent",), "train": 2, "test": 1},
        ],
    )
    monkeypatch.setattr(stage1_module, "PAPER_STAGE1_MTA_SPECS", [{"family": "Dridex", "train": 1, "test": 0}])
    monkeypatch.setattr(stage1_module, "PAPER_STAGE1_MFCP_SPECS", [{"family": "PUA", "train": 1, "test": 0}])

    _write_manifest(
        processed_root,
        "ISCX",
        policy,
        [
            {"session_id": "iscx_train_1", "dataset": "ISCX", "family": "F1", "capture_id": "email_a.pcap", "split": "train", "policy": policy},
            {"session_id": "iscx_train_2", "dataset": "ISCX", "family": "F1", "capture_id": "email_b.pcap", "split": "train", "policy": policy},
            {"session_id": "iscx_test_1", "dataset": "ISCX", "family": "F1", "capture_id": "email_c.pcap", "split": "test", "policy": policy},
        ],
    )
    _write_manifest(
        processed_root,
        "MTA",
        policy,
        [{"session_id": "mta_1", "dataset": "MTA", "family": "Dridex", "capture_id": "d1.pcap", "split": "train", "policy": policy}],
    )
    _write_manifest(
        processed_root,
        "MFCP",
        policy,
        [{"session_id": "mfcp_1", "dataset": "MFCP", "family": "PUA", "capture_id": "p1.pcap", "split": "train", "policy": policy}],
    )

    manifest = build_stage1_manifest(processed_root=processed_root, policy=policy, protocol_mode="paper_balanced")

    assert set(manifest["session_id"].tolist()) == {"iscx_train_1", "iscx_train_2", "iscx_test_1", "mta_1", "mfcp_1"}


def test_stage1_manifest_balanced_caps_oversupplied_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    processed_root = tmp_path / "processed"
    policy = "session_full"

    monkeypatch.setattr(
        stage1_module,
        "PAPER_STAGE1_ISCX_SPECS",
        [{"name": "email_group", "capture_prefixes": ("email",), "train": 2, "test": 1}],
    )
    monkeypatch.setattr(stage1_module, "PAPER_STAGE1_MTA_SPECS", [{"family": "Dridex", "train": 1, "test": 0}])
    monkeypatch.setattr(stage1_module, "PAPER_STAGE1_MFCP_SPECS", [{"family": "PUA", "train": 1, "test": 0}])

    _write_manifest(
        processed_root,
        "ISCX",
        policy,
        [
            {"session_id": "iscx_train_1", "dataset": "ISCX", "family": "F1", "capture_id": "email_a.pcap", "split": "train", "policy": policy},
            {"session_id": "iscx_train_2", "dataset": "ISCX", "family": "F1", "capture_id": "email_b.pcap", "split": "train", "policy": policy},
            {"session_id": "iscx_train_3", "dataset": "ISCX", "family": "F1", "capture_id": "email_c.pcap", "split": "train", "policy": policy},
            {"session_id": "iscx_train_4", "dataset": "ISCX", "family": "F1", "capture_id": "email_d.pcap", "split": "train", "policy": policy},
            {"session_id": "iscx_test_1", "dataset": "ISCX", "family": "F1", "capture_id": "email_e.pcap", "split": "test", "policy": policy},
            {"session_id": "iscx_test_2", "dataset": "ISCX", "family": "F1", "capture_id": "email_f.pcap", "split": "test", "policy": policy},
            {"session_id": "iscx_test_3", "dataset": "ISCX", "family": "F1", "capture_id": "email_g.pcap", "split": "test", "policy": policy},
        ],
    )
    _write_manifest(
        processed_root,
        "MTA",
        policy,
        [{"session_id": "mta_1", "dataset": "MTA", "family": "Dridex", "capture_id": "d1.pcap", "split": "train", "policy": policy}],
    )
    _write_manifest(
        processed_root,
        "MFCP",
        policy,
        [{"session_id": "mfcp_1", "dataset": "MFCP", "family": "PUA", "capture_id": "p1.pcap", "split": "train", "policy": policy}],
    )

    manifest = build_stage1_manifest(processed_root=processed_root, policy=policy, protocol_mode="paper_balanced")

    iscx_rows = manifest[manifest["dataset"] == "ISCX"]
    assert set(iscx_rows["session_id"].tolist()) == {
        "iscx_train_1",
        "iscx_train_2",
        "iscx_train_3",
        "iscx_test_1",
        "iscx_test_2",
    }


def test_stage1_manifest_balanced_keeps_all_undersupplied_samples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    processed_root = tmp_path / "processed"
    policy = "session_full"

    monkeypatch.setattr(
        stage1_module,
        "PAPER_STAGE1_ISCX_SPECS",
        [{"name": "email_group", "capture_prefixes": ("email",), "train": 10, "test": 5}],
    )
    monkeypatch.setattr(stage1_module, "PAPER_STAGE1_MTA_SPECS", [{"family": "Dridex", "train": 1, "test": 0}])
    monkeypatch.setattr(stage1_module, "PAPER_STAGE1_MFCP_SPECS", [{"family": "PUA", "train": 1, "test": 0}])

    _write_manifest(
        processed_root,
        "ISCX",
        policy,
        [
            {"session_id": "iscx_train_1", "dataset": "ISCX", "family": "F1", "capture_id": "email_a.pcap", "split": "train", "policy": policy},
            {"session_id": "iscx_train_2", "dataset": "ISCX", "family": "F1", "capture_id": "email_b.pcap", "split": "train", "policy": policy},
            {"session_id": "iscx_test_1", "dataset": "ISCX", "family": "F1", "capture_id": "email_c.pcap", "split": "test", "policy": policy},
        ],
    )
    _write_manifest(
        processed_root,
        "MTA",
        policy,
        [{"session_id": "mta_1", "dataset": "MTA", "family": "Dridex", "capture_id": "d1.pcap", "split": "train", "policy": policy}],
    )
    _write_manifest(
        processed_root,
        "MFCP",
        policy,
        [{"session_id": "mfcp_1", "dataset": "MFCP", "family": "PUA", "capture_id": "p1.pcap", "split": "train", "policy": policy}],
    )

    manifest = build_stage1_manifest(processed_root=processed_root, policy=policy, protocol_mode="paper_balanced")

    iscx_rows = manifest[manifest["dataset"] == "ISCX"]
    assert set(iscx_rows["session_id"].tolist()) == {"iscx_train_1", "iscx_train_2", "iscx_test_1"}


def test_stage1_main_emits_progress_logs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    processed_root = tmp_path / "processed"
    policy = "session_full"
    _patch_minimal_paper_specs(monkeypatch)
    for dataset in ("ISCX", "MFCP", "MTA"):
        family = "F1"
        capture_id = "c1.pcap"
        if dataset == "ISCX":
            capture_id = "vpn_facebook_chat1a.pcap"
        elif dataset == "MFCP":
            family = "PUA"
        elif dataset == "MTA":
            family = "Dridex"
        _write_manifest(
            processed_root,
            dataset,
            policy,
            [
                {
                    "session_id": f"{dataset}_1",
                    "dataset": dataset,
                    "family": family,
                    "capture_id": capture_id,
                    "split": "train",
                    "policy": policy,
                }
            ],
        )

    output = tmp_path / "stage1_binary_manifest.csv"
    exit_code = main(
        [
            "--processed-root",
            str(processed_root),
            "--policy",
            policy,
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Stage1Binary" in captured.out
    assert "Manifest 已保存" in captured.out
