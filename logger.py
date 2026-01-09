import sys
import logging
from logging.handlers import RotatingFileHandler

# --- Custom SKIP level ---
SKIP_LEVEL = 25
logging.addLevelName(SKIP_LEVEL, "SKIP")


def _add_skip_method():
    def skip(self, message: str = ""):
        self.log(SKIP_LEVEL, message)
    if not hasattr(logging.Logger, "skip"):
        logging.Logger.skip = skip


_add_skip_method()

def setup_logging(
    level: int = logging.INFO,
    log_to_file: bool = False,
    log_file: str = "app.log"
) -> None:
    """
    Configure application-wide logging.
    Must be called ONCE (e.g., in main.py).
    """

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Prevent duplicate handlers
    if root_logger.handlers:
        return

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Optional file handler
    if log_to_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5_000_000,
            backupCount=3,
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
