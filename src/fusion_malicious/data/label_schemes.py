from fusion_malicious.data.records import SessionRecord


def assign_binary_label(record: SessionRecord) -> int:
    return 0 if record.dataset == "ISCX" else 1


def build_multiclass_label_map(records: list[SessionRecord]) -> dict[str, int]:
    families = sorted({record.family for record in records})
    return {family: index for index, family in enumerate(families)}
