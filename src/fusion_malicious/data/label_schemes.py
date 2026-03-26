from fusion_malicious.data.records import SessionRecord


def assign_binary_label(record: SessionRecord) -> int:
    benign_datasets = {"ISCX", "ISCX-VPN-NonVPN-2016"}
    malicious_datasets = {"MTA", "MFCP"}
    dataset = record.dataset
    if dataset in benign_datasets:
        return 0
    if dataset in malicious_datasets:
        return 1
    raise ValueError(
        f"Unsupported dataset {dataset!r} for binary task; expected one of "
        f"{sorted(benign_datasets | malicious_datasets)}"
    )


def build_multiclass_label_map(records: list[SessionRecord]) -> dict[str, int]:
    families = sorted({record.family for record in records})
    return {family: index for index, family in enumerate(families)}
