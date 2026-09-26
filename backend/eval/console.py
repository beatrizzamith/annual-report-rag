"""Console logging setup shared by the eval command-line scripts."""

import logging


def configure_console_logging(script_logger: logging.Logger) -> None:
    """Shows a script's own INFO report lines while keeping the app's logs quiet.

    Called from each script's `main()`, never at import time: importing a
    module must not reconfigure logging for whoever imported it.

    Args:
        script_logger: The script's module logger, whose INFO messages carry
            its human-readable report.
    """
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    script_logger.setLevel(logging.INFO)
