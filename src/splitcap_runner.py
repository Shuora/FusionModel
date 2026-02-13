"""Utilities for invoking SplitCap and collecting split session files."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

LOGGER = logging.getLogger(__name__)


@dataclass
class SplitCapResult:
    success: bool
    session_files: List[Path]
    command: List[str]
    stdout: str
    stderr: str
    returncode: int


def default_splitcap_path(repo_root: Optional[Path] = None) -> Path:
    root = repo_root or Path(__file__).resolve().parent.parent
    return (root / "tools" / "SplitCap_2-1" / "SplitCap.exe").resolve()


def build_splitcap_command(
    splitcap_exe: Path,
    input_pcap: Path,
    output_dir: Path,
    *,
    split_group: str = "session",
    output_filetype: str = "pcap",
    delete_previous: bool = True,
) -> List[str]:
    cmd: List[str] = [str(splitcap_exe), "-r", str(input_pcap), "-o", str(output_dir), "-s", split_group]
    if delete_previous:
        cmd.append("-d")
    if output_filetype and output_filetype.lower() != "pcap":
        cmd.extend(["-y", output_filetype])
    return cmd


def collect_session_files(output_dir: Path) -> List[Path]:
    if not output_dir.exists():
        return []

    preferred_exts = {".pcap", ".pcapng"}
    pcap_files = [p for p in output_dir.rglob("*") if p.is_file() and p.suffix.lower() in preferred_exts]
    if pcap_files:
        return sorted(pcap_files)

    # For non-pcap output (for example L7), SplitCap can emit files without pcap suffix.
    generic_files = [p for p in output_dir.rglob("*") if p.is_file()]
    return sorted(generic_files)


def run_splitcap(
    splitcap_exe: Path,
    input_pcap: Path,
    output_dir: Path,
    *,
    split_group: str = "session",
    output_filetype: str = "pcap",
    delete_previous: bool = True,
    timeout_sec: int = 3600,
) -> SplitCapResult:
    splitcap_exe = Path(splitcap_exe).resolve()
    input_pcap = Path(input_pcap).resolve()
    output_dir = Path(output_dir).resolve()

    if not splitcap_exe.exists():
        raise FileNotFoundError(f"SplitCap executable not found: {splitcap_exe}")
    if not input_pcap.exists():
        raise FileNotFoundError(f"Input pcap not found: {input_pcap}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_splitcap_command(
        splitcap_exe,
        input_pcap,
        output_dir,
        split_group=split_group,
        output_filetype=output_filetype,
        delete_previous=delete_previous,
    )

    LOGGER.debug("Running SplitCap command: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_sec,
        check=False,
    )

    files = collect_session_files(output_dir)
    success = proc.returncode == 0 and len(files) > 0

    if not success:
        LOGGER.warning(
            "SplitCap failed or no sessions generated: returncode=%s input=%s output=%s",
            proc.returncode,
            input_pcap,
            output_dir,
        )

    return SplitCapResult(
        success=success,
        session_files=files,
        command=cmd,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        returncode=proc.returncode,
    )
