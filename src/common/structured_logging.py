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
    "data": ("🧱", "Data"),
    "model": ("🧠", "Model"),
    "eval": ("🧪", "Eval"),
    "save": ("💾", "Save"),
    "time": ("⏱️", "Time"),
    "metric": ("📈", "Metric"),
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
        module.lower(), ("🧩", module.capitalize())
    )

    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    kv = kv or {}
    kv_text = " ".join(f"{k}={v}" for k, v in kv.items())

    base = f"{ts} | {level_icon}{level_text} | {module_icon} {module_text} | {event}"
    return f"{base} | {kv_text}" if kv_text else base
