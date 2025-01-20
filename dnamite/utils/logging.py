import logging

def get_logger(name: str, verbosity: int = 0):
    """
    Create a logger for a specific class or function with the given verbosity level.
    
    Parameters:
    name (str): Name of the logger.
    verbosity (int): Verbosity level (0 = None, 1 = Info, 2 = Debug).

    Returns:
    logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    
    if verbosity == 0:
        logger.setLevel(logging.WARNING)  # Only warnings and errors
    elif verbosity == 1:
        logger.setLevel(logging.INFO)     # General progress info
    elif verbosity >= 2:
        logger.setLevel(logging.DEBUG)    # Debugging info
    
    # Clear existing handlers to avoid duplication
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger

class LoggingMixin:
    def __init__(self, verbosity=0):
        self.logger = get_logger(self.__class__.__name__, verbosity)

