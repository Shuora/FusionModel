from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from src.evaluate import main as evaluate_main
from src.report import main as report_main
from src.train import main as train_main


STAGE2_TASKS = [
    {"dataset": "MTA", "num_classes": 7},
    {"dataset": "MFCP", "num_classes": 6},
    {"dataset": "USTC-TFC2016", "num_classes": 10},
]


def build_stage2_tasks() -> List[dict]:
    return [dict(item) for item in STAGE2_TASKS]


def _run_stage_report(run_dir: Path, stage: str, device: str) -> int:
    if stage in {"warmup", "fusion"}:
        eval_code = evaluate_main(["--run-dir", str(run_dir), "--split", "test", "--device", device])
        if eval_code != 0:
            return eval_code
    report_code = report_main(["--run-dir", str(run_dir)])
    return report_code


def _run_stage2_task(
    processed_root: Path,
    policy: str,
    run_root: Path,
    dataset: str,
    num_classes: int,
    stage: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
    num_workers: int,
    hidden_dim: int,
    fusion_layers: int,
    fusion_heads: int,
    fusion_dropout: float,
    alpha: float,
    beta: float,
    val_fraction: float,
    best_metric: str,
    train_max_samples: int | None = None,
    run_id_suffix: str = "",
) -> int:
    run_id = f"stage2-{dataset.lower()}{run_id_suffix}"
    train_args = [
        "--processed-root",
        str(processed_root),
        "--policy",
        policy,
        "--stage",
        stage,
        "--run-root",
        str(run_root),
        "--run-id",
        run_id,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--seed",
        str(seed),
        "--hidden-dim",
        str(hidden_dim),
        "--fusion-layers",
        str(fusion_layers),
        "--fusion-heads",
        str(fusion_heads),
        "--fusion-dropout",
        str(fusion_dropout),
        "--alpha",
        str(alpha),
        "--beta",
        str(beta),
        "--val-fraction",
        str(val_fraction),
        "--best-metric",
        str(best_metric),
        "--device",
        device,
        "--num-workers",
        str(num_workers),
        "--datasets",
        dataset,
        "--label-mode",
        "multiclass",
        "--num-classes",
        str(num_classes),
    ]
    if train_max_samples is not None:
        train_args.extend(["--train-max-samples", str(train_max_samples)])
    train_code = train_main(train_args)
    if train_code != 0:
        return train_code

    run_dir = run_root / run_id
    return _run_stage_report(run_dir=run_dir, stage=stage, device=device)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build stage2 multiclass task list")
    parser.add_argument("--output", default="outputs/protocol/stage2_tasks.json")
    parser.add_argument("--execute", action="store_true", default=False)
    parser.add_argument("--processed-root")
    parser.add_argument("--policy", default="session_full")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--stage", default="fusion", choices=["warmup", "fusion", "stacking", "moe"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--fusion-heads", "--num-heads", dest="fusion_heads", type=int, default=4)
    parser.add_argument("--fusion-dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--best-metric", default="val_macro_f1", choices=["val_macro_f1", "val_acc"])
    parser.add_argument("--ustc-train-limits", nargs="+", type=int, default=[4000, 3000, 2000])
    parser.add_argument("--skip-ustc-limited", action="store_true", default=False)
    args = parser.parse_args(list(argv) if argv is not None else None)

    tasks = build_stage2_tasks()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.execute:
        return 0
    if not args.processed_root:
        raise ValueError("--processed-root is required when --execute is enabled")

    processed_root = Path(args.processed_root)
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    summary: List[dict] = []

    for task in tasks:
        dataset = str(task["dataset"])
        num_classes = int(task["num_classes"])
        code = _run_stage2_task(
            processed_root=processed_root,
            policy=args.policy,
            run_root=run_root,
            dataset=dataset,
            num_classes=num_classes,
            stage=args.stage,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
            num_workers=args.num_workers,
            hidden_dim=args.hidden_dim,
            fusion_layers=args.fusion_layers,
            fusion_heads=args.fusion_heads,
            fusion_dropout=args.fusion_dropout,
            alpha=args.alpha,
            beta=args.beta,
            val_fraction=args.val_fraction,
            best_metric=args.best_metric,
        )
        summary.append(
            {
                "dataset": dataset,
                "num_classes": num_classes,
                "run_id": f"stage2-{dataset.lower()}",
                "train_max_samples": None,
                "code": int(code),
            }
        )
        if code != 0:
            (run_root / "stage2_execution_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return code
        if dataset == "USTC-TFC2016" and not args.skip_ustc_limited:
            for limit in args.ustc_train_limits:
                limit_code = _run_stage2_task(
                    processed_root=processed_root,
                    policy=args.policy,
                    run_root=run_root,
                    dataset=dataset,
                    num_classes=num_classes,
                    stage=args.stage,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    seed=args.seed,
                    device=args.device,
                    num_workers=args.num_workers,
                    hidden_dim=args.hidden_dim,
                    fusion_layers=args.fusion_layers,
                    fusion_heads=args.fusion_heads,
                    fusion_dropout=args.fusion_dropout,
                    alpha=args.alpha,
                    beta=args.beta,
                    val_fraction=args.val_fraction,
                    best_metric=args.best_metric,
                    train_max_samples=int(limit),
                    run_id_suffix=f"-train{int(limit)}",
                )
                summary.append(
                    {
                        "dataset": dataset,
                        "num_classes": num_classes,
                        "run_id": f"stage2-{dataset.lower()}-train{int(limit)}",
                        "train_max_samples": int(limit),
                        "code": int(limit_code),
                    }
                )
                if limit_code != 0:
                    (run_root / "stage2_execution_summary.json").write_text(
                        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    return limit_code

    (run_root / "stage2_execution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
