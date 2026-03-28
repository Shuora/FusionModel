from __future__ import annotations

from pathlib import Path


def build_splitcap_command(
    splitcap_exe: Path,
    input_pcap: Path,
    output_dir: Path,
    *,
    launcher: list[str] | None = None,
) -> list[str]:
    """
    Build the command line that invokes SplitCap with the specified PCAP and output folder.
    """
    command = [
        str(splitcap_exe),
        "-r",
        str(input_pcap),
        "-s",
        "session",
        "-o",
        str(output_dir),
    ]
    if launcher:
        return [*launcher, *command]
    return command
