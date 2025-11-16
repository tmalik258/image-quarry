import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(name: str, log_dir: str = "logs", max_bytes: int = 5*1024*1024, backup_count: int = 3) -> logging.Logger:
    """
    Configure a logger with console and rotating file handlers.
    
    Args:
        name: Logger name (usually __name__).
        log_dir: Directory for log files.
        max_bytes: Maximum size of log file before rotation.
        backup_count: Number of backup log files to keep.
    
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if not logger.handlers:
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, f"{name}.log"),
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger