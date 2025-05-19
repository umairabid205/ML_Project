import logging
import os
import sys
from datetime import datetime


log_file = f"{datetime.now().strftime('%Y-%m-%d')}.log" # Create a log file with the current date
log_path = os.path.join(os.getcwd(), "logs", log_file) # Create a path for the log file
os.makedirs(os.path.dirname(log_path), exist_ok=True) # Create the directory if it doesn't exist


log_fild_path = os.path.join(os.getcwd(), "logs", log_file) # Create a path for the log file

logging.basicConfig(

    filename=log_fild_path,
    format='%(asctime)s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
) # Configure the logging module to write logs to the log file with a specific format and level

