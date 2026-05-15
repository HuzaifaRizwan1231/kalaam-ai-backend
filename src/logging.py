import logging
from enum import StrEnum


LOG_FORMAT_DEBUG = "%(levelname)s:%(message)s:%(pathname)s:%(funcName)s:%(lineno)d"


class LogLevels(StrEnum):
    info = "INFO"
    warn = "WARN"
    error = "ERROR"
    debug = "DEBUG"


def configure_logging(log_level: str = LogLevels.info):
    log_level = str(log_level).upper()
    
    # Standard format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if log_level == LogLevels.debug:
        log_format = LOG_FORMAT_DEBUG

    # Configure the root logger
    logging.basicConfig(
        level=log_level,
        format=log_format
    )
    
    # Ensure specific loggers are at least at the desired level
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("src").setLevel(log_level)
    
    # Optional: silence noisy libraries
    logging.getLogger("multipart").setLevel(logging.WARNING)
    
    logging.info(f"Logging configured at level: {log_level}")