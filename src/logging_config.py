# logging_config.py
import logging
import os
from logging.handlers import RotatingFileHandler

LOGS_DIR = "logs"
LOG_FILENAME = os.path.join(LOGS_DIR, "app.log")

# Create a filter to allow specific loggers' INFO messages to show in console
class ConsoleFilter(logging.Filter):
    def filter(self, record):
        # Allow all WARNING+ messages to pass
        if record.levelno >= logging.WARNING:
            return True
            
        # For INFO messages, only allow specific modules/loggers
        if record.levelno == logging.INFO:
            # Allow main_api INFO messages for startup/status
            if record.name == 'src.main_api':
                return True
            # Allow run.py INFO messages
            if record.name == '__main__':
                return True
        
        # Filter out httpx INFO messages completely
        if record.name == 'httpx':
            return False
            
        # Default behavior: filter out INFO and DEBUG from console
        return False

def setup_logging():
    """Configures logging for the application."""
    # Ensure the logs directory exists
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Get the root logger
    logger = logging.getLogger() # Get root logger to configure basicConfig properties for all
    logger.setLevel(logging.INFO) # Set the default minimum level for the root logger

    # Prevent multiple handlers if called more than once (e.g., in tests or reloads)
    if logger.hasHandlers():
        # Check if our specific file handler is already there to avoid duplication during reloads
        # This is a simple check; more robust checks might involve handler names or types.
        has_file_handler = any(isinstance(h, RotatingFileHandler) and h.baseFilename == os.path.abspath(LOG_FILENAME) for h in logger.handlers)
        if has_file_handler:
            # print("Logging already configured with our file handler.")
            return
        # If handlers exist but not our file handler, it might be from a different setup.
        # For simplicity here, we'll clear them to avoid duplicate messages from other basicConfigs.
        # print("Clearing existing handlers to reconfigure logging.")
        # for handler in logger.handlers[:]: 
        # logger.removeHandler(handler)

    # Create handlers
    console_handler = logging.StreamHandler() # To log to console
    # File handler with rotation: 1MB per file, keep 5 backup files
    file_handler = RotatingFileHandler(LOG_FILENAME, maxBytes=1*1024*1024, backupCount=5)

    # Set levels for handlers
    console_handler.setLevel(logging.INFO)  # INFO level, but will use filter
    file_handler.setLevel(logging.INFO)  # Detailed logs in file

    # Add filter to console handler to be selective
    console_filter = ConsoleFilter()
    console_handler.addFilter(console_filter)

    # Create formatters and add it to handlers
    console_formatter = logging.Formatter('%(message)s')  # Clean format without level
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)

    # Add handlers to the logger
    # Check again before adding to be absolutely sure if clearing was not done or was partial
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(console_handler)
    
    if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == os.path.abspath(LOG_FILENAME) for h in logger.handlers):
        logger.addHandler(file_handler)
    
    # Configure specific loggers
    
    # Set httpx logger to WARNING to avoid all those API call logs in the console
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Other third-party loggers can be configured here as needed
    # logging.getLogger("uvicorn").setLevel(logging.WARNING)
    # logging.getLogger("fastapi").setLevel(logging.WARNING)
