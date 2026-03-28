from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping, NamedTuple

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch

from fusion_malicious.config import build_run_layout
from fusion_malicious.data.dataset import CachedSessionDataset
from fusion_malicious.evaluation import load_manifest_dataframe, resolve_label_names
from fusion_malicious.models.factory import build_image_backbone, build_text_backbone
from fusion_malicious.models.multimodal import MultimodalClassifier
from fusion_malicious.utils.logging import append_metrics_row, create_logger
from fusion_malicious.utils.metrics import compute_binary_metrics, compute_multiclass_metrics
from fusion_malicious.utils.plots import save_confusion_matrix
from fusion_malicious.utils.reporting import write_classification_report

ALLOWED_CHECKPOINT_SUFFIXES = {".pt", ".pth"}
DEFAULT_IMAGE_MODEL = "resnet18"
DEFAULT_TEXT_MODEL = "distilbert-base-uncased"
DEFAULT_BATCH_SIZE = 32
DEFAULT_HIDDEN_DIM = 32
DEFAULT_NUM_HEADS = 4
DEFAULT_NUM_CLASSES = 2
DEFAULT_NUM_WORKERS = 0


class ManifestArguments(NamedTuple):
    path: Path
    subset: str | None
    subset_column: str
    label_column: str
    label_name_column: str


def build_parser() -> argparse.ArgumentParser:
    default_config = repo_root / "configs" / "binary.yaml"
    parser = argparse.ArgumentParser(description="Evaluate a FusionModel checkpoint using cached samples.")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--subset", type=str)
    parser.add_argument("--subset-column", type=str)
    parser.add_argument("--image-model", type=str)
    parser.add_argument("--text-model", type=str)
    parser.add_argument("--device", type=str, choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    if not path.exists():
        raise FileNotFoundError(f"Config file {path} does not exist.")
    return yaml.safe_load(path.read_text()) or {}


def load_checkpoint_blob(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint {path} does not exist.")
    if path.suffix.lower() not in ALLOWED_CHECKPOINT_SUFFIXES:
        raise ValueError("Checkpoint must end with one of: " + ", ".join(sorted(ALLOWED_CHECKPOINT_SUFFIXES)))
    try:
        return torch.load(path, map_location="cpu")
    except (RuntimeError, ValueError, pickle.UnpicklingError) as exc:
        raise RuntimeError(f"{path} is not a readable PyTorch checkpoint.") from exc


def validate_checkpoint(path: Path) -> None:
    load_checkpoint_blob(path)


def extract_state_dict(blob: Any) -> Mapping[str, Any]:
    if isinstance(blob, dict):
        if "state_dict" in blob and isinstance(blob["state_dict"], dict):
            return blob["state_dict"]
        if "model_state_dict" in blob and isinstance(blob["model_state_dict"], dict):
            return blob["model_state_dict"]
        if all(isinstance(key, str) for key in blob.keys()):
            return blob
    raise ValueError("Checkpoint does not contain a usable state_dict.")


def resolve_manifest_arguments(config: dict[str, Any], args: argparse.Namespace) -> ManifestArguments:
    subset_column = args.subset_column or config.get("subset_column", "subset")
    label_column = config.get("label_column", "label_id")
    label_name_column = config.get("label_name_column", "label_name")
    manifest_path = args.manifest or config.get("manifest", "dataset/binary/manifest.csv")
    manifest_path = Path(manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = (repo_root / manifest_path).resolve()
    subset = args.subset or config.get("subset") or "test"
    return ManifestArguments(
        path=manifest_path,
        subset=subset,
        subset_column=subset_column,
        label_column=label_column,
        label_name_column=label_name_column,
    )


def build_loader(frame, batch_size: int, num_workers: int) -> torch.utils.data.DataLoader:
    dataset = CachedSessionDataset(frame)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def build_model(config: dict[str, Any], args: argparse.Namespace) -> tuple[MultimodalClassifier, int]:
    image_name = args.image_model or config.get("image_model", DEFAULT_IMAGE_MODEL)
    text_name = args.text_model or config.get("text_model", DEFAULT_TEXT_MODEL)
    hidden_dim = int(config.get("hidden_dim", DEFAULT_HIDDEN_DIM))
    num_heads = int(config.get("num_heads", DEFAULT_NUM_HEADS))
    num_classes = int(config.get("num_classes", DEFAULT_NUM_CLASSES))
    pretrained = bool(config.get("pretrained", False))

    image_backbone = build_image_backbone(image_name, pretrained=pretrained)
    text_backbone = build_text_backbone(text_name, pretrained=pretrained)
    return MultimodalClassifier(image_backbone, text_backbone, hidden_dim, num_classes, num_heads), num_classes


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_inference(model: MultimodalClassifier, loader, device: torch.device) -> tuple[list[int], list[int], list[float]]:
    model = model.to(device)
    model.eval()
    all_targets: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(images, input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)
            all_targets.extend(labels.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())
            if logits.size(1) == 2:
                all_probabilities.extend(probabilities[:, 1].detach().cpu().tolist())
    return all_targets, all_predictions, all_probabilities


def write_outputs(
    run_dir: Path,
    *,
    metrics: Mapping[str, float],
    metadata: Mapping[str, Any],
    targets: list[int],
    predictions: list[int],
    label_ids: list[Any],
    label_names: list[str],
) -> None:
    outputs = run_dir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    write_classification_report(
        targets,
        predictions,
        label_names,
        outputs / "classification_report.txt",
        class_values=label_ids,
    )
    save_confusion_matrix(
        targets,
        predictions,
        label_names,
        outputs / "confusion_matrix.png",
        class_values=label_ids,
    )
    (outputs / "metrics.json").write_text(
        json.dumps({"metrics": dict(metrics), "metadata": dict(metadata)}, indent=2),
        encoding="utf-8",
    )
    append_metrics_row(outputs / "metrics.csv", {**metrics, **metadata})


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    checkpoint_blob = load_checkpoint_blob(args.checkpoint)
    state_dict = extract_state_dict(checkpoint_blob)
    manifest_args = resolve_manifest_arguments(config, args)
    manifest = load_manifest_dataframe(
        manifest_args.path,
        subset=manifest_args.subset,
        subset_column=manifest_args.subset_column,
    )
    label_ids, label_names = resolve_label_names(
        manifest,
        label_column=manifest_args.label_column,
        name_column=manifest_args.label_name_column,
    )

    batch_size = int(config.get("batch_size", DEFAULT_BATCH_SIZE))
    num_workers = int(config.get("num_workers", DEFAULT_NUM_WORKERS))
    loader = build_loader(manifest, batch_size, num_workers)
    model, num_classes = build_model(config, args)
    model.load_state_dict(state_dict)

    run_dir = build_run_layout(repo_root / "runs", config.get("task_name", "evaluation")).run_dir
    for child in ("logs", "outputs"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    logger = create_logger(run_dir / "logs" / "evaluation.log")
    device = resolve_device(args.device)
    logger.info(f"Evaluating checkpoint {args.checkpoint} on {device}")
    logger.info(f"Manifest: {manifest_args.path} subset={manifest_args.subset}")

    targets, predictions, probabilities = run_inference(model, loader, device)
    if num_classes == 2:
        metrics = compute_binary_metrics(targets, predictions, probabilities or None)
    else:
        metrics = compute_multiclass_metrics(targets, predictions)

    metadata = {
        "task": config.get("task_name", "evaluation"),
        "manifest": str(manifest_args.path),
        "subset": manifest_args.subset or "all",
        "num_samples": len(targets),
        "num_classes": num_classes,
    }
    write_outputs(
        run_dir,
        metrics=metrics,
        metadata=metadata,
        targets=targets,
        predictions=predictions,
        label_ids=label_ids,
        label_names=label_names,
    )
    logger.info(f"Finished evaluation. Outputs saved under {run_dir / 'outputs'}")


if __name__ == "__main__":
    main()
