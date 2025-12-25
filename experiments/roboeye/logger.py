import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(name: str = __name__, log_dir: str = None) -> logging.Logger:
    """
    Create or return a module logger that logs to both console and a rotating file.

    Args:
        name: Logger name (typically __name__)
        log_dir: Optional log directory. If None, uses env vars or default location.

    Configurable via env:
      ROBOEYE_LOG_LEVEL (default: INFO)  e.g., DEBUG, INFO, WARNING
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger

    level_str = os.getenv("ROBOEYE_LOG_LEVEL", "DEBUG").upper()
    level = getattr(logging, level_str, logging.DEBUG)
    logger.setLevel(level)

    # where to write logs - prioritize passed log_dir, then env var, then default (current working dir)
    if log_dir:
        base_dir = Path(log_dir)
    else:
        base_dir = Path(os.getcwd()) / "logs" / "roboeye"
    base_dir.mkdir(parents=True, exist_ok=True)

    # include pid so DataLoader workers don’t fight over a single file
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = base_dir / f"roboeye_{timestamp}_{os.getpid()}.log"

    # file handler (rotates at ~10MB, keeps 5 backups)
    fh = RotatingFileHandler(log_file, maxBytes=10_000_000, backupCount=5)
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | pid=%(process)d | %(message)s")
    )

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    logger.debug(f"Logger initialized at {log_file}")
    return logger
