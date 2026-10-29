import attr
import logging


def _classlogger_for(instance: object) -> logging.Logger:
    """Get {module}.{class name} named logger for object."""
    return _logger_for_class(type(instance))


def _logger_for_class(cls: type) -> logging.Logger:
    """Get {module}.{name} named logger for class."""
    return logging.getLogger(f"{cls.__module__}.{cls.__name__}")


ClassLogger = attr.ib(
    default=attr.Factory(_classlogger_for, takes_self=True),
    repr=False,
    init=False,
    cmp=False,
)


class LoggerMixin:
    """Provide a lazily constructed logger named for the concrete class."""

    @property
    def logger(self) -> logging.Logger:
        logger = getattr(self, "_logger", None)

        if not logger:
            logger = _classlogger_for(self)
            setattr(self, "_logger", logger)

        return logger
