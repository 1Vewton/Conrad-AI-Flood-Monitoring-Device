import logging
import os

def setup_logging():
    # logger
    logger = logging.getLogger("logger")
    log_file = "app.log"
    
    # Create logs directory if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Configure logging to file
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s - %(name)s] - %(levelname)s:%(message)s',
        handlers=[
            logging.FileHandler(os.path.join("logs", log_file)),
            logging.StreamHandler() # Also log to console
        ]
    )
    logger.info("Logging setup complete.")
