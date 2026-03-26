from dataclasses import dataclass


@dataclass(frozen=True)
class SessionRecord:
    sample_id: str
    dataset: str
    family: str
    source_path: str
    label_name: str
    label_id: int
