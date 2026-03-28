import logging
from pathlib import Path
from typing import Optional

from tqdm import tqdm


def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("train")
    return logger


def progress_bar(iterable, desc: str):
    return tqdm(iterable, desc=desc, leave=False, ncols=100, colour='green')

