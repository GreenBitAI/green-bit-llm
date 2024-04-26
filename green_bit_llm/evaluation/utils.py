import os
import sys
import json
import time
import logging
from termcolor import colored


def pattern_match(patterns, source_list):
    """
    Function to find unique matching patterns from a source list.
    Args:
        patterns: list of pattern strings to match.
    Returns:
        list: list of task name
    """
    task_names = set()
    for pattern in patterns:
            task_names.add(pattern)
    return list(task_names)

def add_dict_to_json_file(file_path, new_data):
    """
    Update a JSON file based on the top-level keys in new_data. If the key exists, it replaces the existing content
    under that key. If it doesn't exist, it adds the new key with its value.

    Args:
    file_path: Path to the JSON file.
    new_data: Dictionary to add or update in the file.
    """
    # Initialize or load existing data
    if os.path.exists(file_path) and os.stat(file_path).st_size > 0:
        with open(file_path, 'r') as file:
            existing_data = json.load(file)
    else:
        existing_data = {}

    # Merge new data into existing data based on the top-level keys
    existing_data.update(new_data)

    # Write the updated data back to the file
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)

def create_logger(output_dir, dist_rank=0, name=''):
    """
    Creates and configures a logger with console and file handlers.

    Args:
        output_dir (str): Directory where the log files will be stored.
        dist_rank (int): Rank of the process in distributed training. Only the master process (rank 0) will output to the console.
        name (str): Name of the logger, used to differentiate output if multiple loggers are used.

    Returns:
        logger: Configured logger object.
    """
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}_{int(time.time())}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger

