from rich.logging import logging
import sys


def filter_stdout(record):
    return record.levelno < logging.ERROR

class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log lines."""

    COLORS = {
        logging.INFO: "\033[0;37m",  # White
        logging.DEBUG: "\033[0;32m",  # Green
        logging.WARNING: "\033[0;33m",  # Yellow
        logging.ERROR: "\033[0;31m",  # Red
        logging.CRITICAL: "\033[1;41m"  # Red background
    }

    RESET = "\033[0m"

    def format(self, record):
        log_fmt = self.COLORS.get(record.levelno,
                                  self.RESET) + '%(asctime)s - %(filename)s - %(levelname)s - %(message)s' + self.RESET
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class Logger:
    _instance = None

    def __new__(cls, name: str = 'default_logger', level: int = logging.INFO):
        if cls._instance is None:
            # Create a new instance and configure it
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._configure(name, level)
        return cls._instance

    def _configure(self, name: str, level: int):
        """Configure the Singleton Logger."""
        self.logger = logging.getLogger(name)
        if not self.logger.hasHandlers():
            self.logger.setLevel(level)

            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setLevel(logging.DEBUG)
            stdout_handler.addFilter(filter_stdout)

            stderr_handler = logging.StreamHandler(sys.stderr)
            stderr_handler.setLevel(logging.ERROR)

            formatter = ColoredFormatter()
            stdout_handler.setFormatter(formatter)
            stderr_handler.setFormatter(formatter)

            self.logger.addHandler(stdout_handler)
            self.logger.addHandler(stderr_handler)

    def get_logger(self):
        """Return the configured logger."""
        return self.logger

    def set_level(self, level: logging.INFO | logging.DEBUG | logging.WARNING | logging.ERROR | logging.CRITICAL):
        self.logger.setLevel(level)


# Example usage
if __name__ == '__main__':
    logger_instance = Logger().get_logger()
    logger_instance.info('This is an info message from the singleton logger!')
    logger_instance.debug('This is a debug message from the singleton logger!')
    logger_instance.warning('This is an warning message from the singleton logger!')
    logger_instance.error('This is an error message from the singleton logger!')
    logger_instance.critical('This is a critical message from the singleton logger!')
