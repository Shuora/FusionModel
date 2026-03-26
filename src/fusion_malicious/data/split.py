from sklearn.model_selection import train_test_split

from fusion_malicious.data.records import SessionRecord


def stratified_split_records(
    records: list[SessionRecord],
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> dict[str, list[SessionRecord]]:
    total = round(train_size + val_size + test_size, 5)
    if total != 1.0:
        raise ValueError("split sizes must sum to 1.0")

    labels = [record.label_id for record in records]
    train_records, test_records = train_test_split(
        records,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    train_labels = [record.label_id for record in train_records]
    val_ratio = val_size / (train_size + val_size)
    train_records, val_records = train_test_split(
        train_records,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_labels,
    )
    return {"train": train_records, "val": val_records, "test": test_records}
