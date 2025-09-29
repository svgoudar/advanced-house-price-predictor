import logging
from abc import ABCMeta


class AnalyzerValidator(type):
    def __init__(cls, name, bases, attrs):
        if name != "BaseAnalyzer":  # avoid checking abstract base
            if not hasattr(cls, "__iter__"):
                raise TypeError(f"{name} must implement __iter__()")
        super().__init__(name, bases, attrs)


class LoggingMeta(type):
    def __new__(meta, name, bases, attrs):
        # If class doesn't already define a "logger", inject one
        if "logger" not in attrs:
            # Create a logger with the class name
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)

            # Avoid adding multiple handlers if class is reloaded
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s"
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)

            attrs["logger"] = logger

        # If no custom log() method, inject one
        if "log" not in attrs:

            def log(self, msg, level=logging.INFO):
                self.logger.log(level, msg)

            attrs["log"] = log

        return super().__new__(meta, name, bases, attrs)


class AnalyzerRegistry(type):
    """Meta for auto-registration + validation"""

    registry = {}

    def __init__(cls, name, bases, attrs):
        if name != "BaseAnalyzer":
            # Auto register
            AnalyzerMeta.registry[name] = cls

            # Enforce methods
            required = ["analyze", "visualize"]
            for method in required:
                if not callable(getattr(cls, method, None)):
                    raise TypeError(f"{name} must implement `{method}`")
        super().__init__(name, bases, attrs)


class AnalyzerMeta(AnalyzerRegistry, AnalyzerValidator, LoggingMeta, ABCMeta):
    """
    Combines:
    - Auto Registration
    - Validation
    - Logging injection
    - Abstract Base Class functionality
    """

    pass
