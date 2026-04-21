import logging

try:
    import coloredlogs
except ImportError:
    coloredlogs = None

def get_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(__name__)
    if coloredlogs is not None:
        coloredlogs.install(level=level, logger=logger, fmt="%(asctime)s [%(levelname)s] %(message)s", isatty=True)
    elif not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(level)
    logger.info("Logger initialized.")
    return logger
