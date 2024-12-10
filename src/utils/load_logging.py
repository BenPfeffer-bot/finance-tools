import logging
import os
from typing import Optional

def load_logger(
    log_file: str = '/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/Cloud_Code/Projects-Porfolio/Tech-Sector-Deep-Dive/logs/logfile.log', #en config ici
    logger_name: str = __name__,
    log_level: int = logging.INFO,
    log_format: Optional[str] = None,
    create_missing_dirs: bool = True,
    propagate: bool = False
) -> logging.Logger:
    """
    Load and configure a logger instance.

    This function sets up a logger with both a file handler and a stream handler.
    It ensures that the specified log directory and file exist (creating them if needed),
    and applies a specified logging level and format. The returned logger can be used
    throughout your application for consistent logging.

    Parameters
    ----------
    log_file : str, optional
        The full path to the log file. The default is a specific file path.
    logger_name : str, optional
        The name of the logger instance. Defaults to `__name__`.
    log_level : int, optional
        The logging level. Defaults to `logging.INFO`. 
        Examples: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL.
    log_format : str, optional
        The format of the log messages. If None, a default format will be used.
    create_missing_dirs : bool, optional
        If True, creates missing directories for the log file. Defaults to True.
    propagate : bool, optional
        If False, log messages are not passed to the handlers of ancestor loggers.
        Defaults to False.

    Returns
    -------
    logging.Logger
        A configured logger instance with both file and stream handlers.

    Raises
    ------
    OSError
        If directories or the file cannot be created due to filesystem permissions or other issues.
    """

    # Default log format
    if log_format is None:
        log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    # Extract the directory from the log_file path
    log_dir = os.path.dirname(log_file)

    # Create directories if they don't exist
    if create_missing_dirs and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as e:
            raise OSError(f"Failed to create log directory '{log_dir}': {e}")

    # Ensure the log file exists, create if missing
    if not os.path.exists(log_file):
        try:
            with open(log_file, 'w'):
                pass
        except OSError as e:
            raise OSError(f"Failed to create log file '{log_file}': {e}")

    # Get or create the logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = propagate

    # If the logger already has handlers, return it directly to avoid duplicates
    if logger.handlers:
        return logger

    # Create and set formatter
    formatter = logging.Formatter(log_format)

    # File handler
    try:
        file_handler = logging.FileHandler(log_file)
    except OSError as e:
        raise OSError(f"Failed to open file handler for '{log_file}': {e}")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream handler (console)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger