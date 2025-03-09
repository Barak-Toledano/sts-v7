import logging
import os
import datetime

# ✅ Ensure logs directory exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ✅ Generate a unique log file per session
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"session_{timestamp}.log")

# ✅ Fix: Explicitly set UTF-8 encoding for logs
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"  # ✅ Fix for emoji support
)

def get_logger():
    """ Returns a logger instance for consistent logging across modules. """
    return logging.getLogger()
