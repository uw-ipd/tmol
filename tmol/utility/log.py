import logging


class LoggerMixin:
    @property
    def logger(self):
        logger = getattr(self, "_logger", None)

        if not logger:
            logger = logging.getLogger(
                f"{self.__class__.__module__}.{self.__class__.__name__}"
            )
            setattr(self, "_logger", logger)

        return logger
