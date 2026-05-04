import os
import sys
import logging
from datetime import datetime

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
log_dir = os.path.join(os.getcwd(), "logs")
log_file_name = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
os.makedirs(log_dir, exist_ok=True)
log_filepath = os.path.join(log_dir, log_file_name)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    force=True,

    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)