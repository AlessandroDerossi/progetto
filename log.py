import logging

class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.INFO: "\033[92m",    # Green
        logging.WARNING: "\033[91m", # Red
        logging.DEBUG: "\033[94m",   # Blue
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"

def configure_logger(name: str = "my_logger"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = ColorFormatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.handlers = [handler]
    return logger