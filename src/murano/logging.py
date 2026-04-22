"""Murano logging configuration."""

import logging

logger = logging.getLogger("murano")
logger.addHandler(logging.NullHandler())


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the murano logger with a console handler.

    Call explicitly to enable Murano's default console logging:
        murano.setup_logging(logging.DEBUG)
    """
    has_console_handler = any(
        getattr(handler, "_murano_console_handler", False)
        for handler in logger.handlers
    )
    if not has_console_handler:
        handler = logging.StreamHandler()
        handler._murano_console_handler = True
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] murano: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
