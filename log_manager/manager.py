import logging
from ast import List
from pathlib import Path

from misc.dclasses import LoggingConfig


class LoggingManager:
    @staticmethod
    def get_default_logger():
        return logging.getLogger(__name__)

    @staticmethod
    def get_logger_by_name(name: str):
        return logging.getLogger(name)

    def __init__(self, config: LoggingConfig) -> None:
        self.config = config

    def register_handlers(
        self, level=logging.DEBUG, name: str = None, save_path: Path = None
    ) -> None:
        # Create a custom logger
        if name is None:
            logger = LoggingManager.get_default_logger()
        else:
            logger = LoggingManager.get_logger_by_name(name)
        logger.setLevel(level)

        # Create handlers
        handlers: List[logging.Handler] = []
        if self.config.enable_stream_log_handler:
            c_handler = logging.StreamHandler()
            c_handler.setLevel(self.config.stream_log_handler_level)
            handlers.append(c_handler)

        if self.config.enable_file_log_handler:
            if save_path is None:
                raise ValueError(
                    "If enable_file_log_handler is set to True, save_path has to be specified but None given."
                )

            f_handler = logging.FileHandler(save_path)
            f_handler.setLevel(self.config.file_log_handler_level)
            handlers.append(f_handler)

        for handler in handlers:
            log_format = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(log_format)
            logger.addHandler(handler)

    def remove_handlers(self, name: str = None):
        """
        to delete data and start new logger with same name, for batch train/testing e.g
        """

        # Create a custom logger
        if name is None:
            logger = LoggingManager.get_default_logger()
        else:
            logger = LoggingManager.get_logger_by_name(name)

        for handler in logger.handlers:
            logger.removeHandler(handler)
