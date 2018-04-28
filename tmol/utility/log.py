import attr
import logging


def classlogger_for(instance: object) -> logging.Logger:
    """Get {module}.{class name} named logger for object."""
    return logger_for_class(type(instance))


def logger_for_class(cls: type) -> logging.Logger:
    """Get {module}.{name} named logger for class."""
    return logging.getLogger(f"{cls.__module__}.{cls.__name__}")


ClassLogger = attr.ib(
    default=attr.Factory(classlogger_for, takes_self=True),
    repr=False,
    init=False,
    cmp=False,
)


class LoggerMixin:
    @property
    def logger(self) -> logging.Logger:
        logger = getattr(self, "_logger", None)

        if not logger:
            logger = classlogger_for(self)
            setattr(self, "_logger", logger)

        return logger
