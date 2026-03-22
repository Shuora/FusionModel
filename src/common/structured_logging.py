from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional


_LEVEL_MAP = {
    "success": ("✅", "SUCCESS"),
    "warning": ("⚠️", "WARNING"),
    "error": ("❌", "ERROR"),
    "info": ("ℹ️", "INFO"),
}

_MODULE_MAP = {
    "data": ("🧱", "数据"),
    "model": ("🧠", "模型"),
    "eval": ("🧪", "评估"),
    "save": ("💾", "保存"),
    "time": ("⏱️", "时间"),
    "metric": ("📈", "指标"),
}

_EVENT_MAP = {
    "dataset_loaded": "数据集已加载",
    "policy_run_start": "策略预处理开始",
    "preprocess_start": "预处理开始",
    "dataset_preprocess_saved": "数据集预处理结果已保存",
    "preprocess_done": "预处理完成",
    "missing_eval_split": "缺少评估切分",
    "eval_done": "评估完成",
    "stage_dispatch_start": "阶段任务派发开始",
    "stage_dispatch_failed": "阶段任务派发失败",
    "stage_dispatch_done": "阶段任务派发完成",
    "empty_dataset": "数据集为空",
    "invalid_num_classes": "类别数配置无效",
    "run_bootstrap": "运行初始化",
    "config_summary": "配置摘要",
    "dataset_stats": "数据集统计",
    "empty_train_split": "训练切分为空",
    "invalid_train_max_samples": "train_max_samples 配置无效",
    "train_samples_limited": "训练样本数已限制",
    "empty_val_split": "验证切分为空",
    "val_split_derived_from_train": "验证集由训练集拆分生成",
    "empty_train_dataset": "训练数据为空",
    "train_start": "训练开始",
    "nan_loss": "损失出现 NaN",
    "invalid_grad_norm": "梯度范数无效",
    "gradient_explosion": "梯度爆炸",
    "grad_clipped": "梯度已裁剪",
    "epoch_done": "Epoch 完成",
    "checkpoint_saved": "Checkpoint 已保存",
    "best_checkpoint_saved": "最佳 Checkpoint 已保存",
    "metrics_saved": "指标文件已保存",
}


def format_log_line(
    level: str,
    module: str,
    event: str,
    kv: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
) -> str:
    level_icon, level_text = _LEVEL_MAP.get(level.lower(), _LEVEL_MAP["info"])
    module_icon, module_text = _MODULE_MAP.get(
        module.lower(), ("🧩", f"模块:{module}")
    )

    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    kv = kv or {}
    kv_text = " ".join(f"{k}={v}" for k, v in kv.items())
    event_text = _EVENT_MAP.get(event, event)
    event_display = f"{event_text} ({event})" if event_text != event else event

    base = f"{ts} | {level_icon}{level_text} | {module_icon} {module_text} | {event_display}"
    return f"{base} | {kv_text}" if kv_text else base
