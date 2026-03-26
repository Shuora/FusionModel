from fusion_malicious.data.label_schemes import (
    assign_binary_label,
    build_multiclass_label_map,
)
from fusion_malicious.data.records import SessionRecord
from fusion_malicious.data.split import stratified_split_records


def make_record(sample_id: str, dataset: str, family: str) -> SessionRecord:
    return SessionRecord(
        sample_id=sample_id,
        dataset=dataset,
        family=family,
        source_path=f"/tmp/{sample_id}.pcap",
        label_name=family,
        label_id=0 if dataset == "ISCX" else 1,
    )


def test_assign_binary_label_marks_iscx_as_benign() -> None:
    benign = make_record("a", "ISCX", "VPN")
    malicious = make_record("b", "MTA", "Trickbot")
    assert assign_binary_label(benign) == 0
    assert assign_binary_label(malicious) == 1


def test_build_multiclass_label_map_uses_sorted_family_names() -> None:
    records = [
        make_record("a", "MTA", "Trickbot"),
        make_record("b", "MTA", "Dridex"),
        make_record("c", "MTA", "Emotet"),
    ]
    assert build_multiclass_label_map(records) == {
        "Dridex": 0,
        "Emotet": 1,
        "Trickbot": 2,
    }


def test_stratified_split_records_preserves_total_count() -> None:
    records = []
    for index in range(10):
        records.append(make_record(f"benign-{index}", "ISCX", "VPN"))
        records.append(make_record(f"mal-{index}", "MTA", "Trickbot"))
    split = stratified_split_records(
        records, train_size=0.7, val_size=0.1, test_size=0.2, seed=7
    )
    assert len(split["train"]) + len(split["val"]) + len(split["test"]) == 20
    assert len(split["test"]) == 4
    assert {record.label_id for record in split["train"]} == {0, 1}
