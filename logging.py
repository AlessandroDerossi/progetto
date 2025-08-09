import logging

def configure_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name)
    return logger
