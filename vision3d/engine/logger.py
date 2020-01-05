import coloredlogs
import logging


def create_logger(log_file=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)

    format_str = '[%(asctime)s] [%(levelname)s] %(message)s'

    stream_handler = logging.StreamHandler()
    colored_formatter = coloredlogs.ColoredFormatter(format_str)
    stream_handler.setFormatter(colored_formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(format_str, datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
