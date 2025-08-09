import logging

def configure_logger(name: str) -> None:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(name)
