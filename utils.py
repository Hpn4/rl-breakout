import gymnasium as gym
import ale_py
import yaml
import csv
import logging
import os
from rich.logging import RichHandler

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(RichHandler(markup=True))

    return logger

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

class CSVLogger:
    def __init__(self, filename, header=None, buffer_size=100):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        self.filename = filename
        self.buffer_size = buffer_size
        self.buffer = []
        self.file = open(filename, mode='w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)

        if header:
            self.writer.writerow(header)
            self.file.flush()

    def add_row(self, row):
        self.buffer.append(row)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        if self.buffer:
            self.writer.writerows(self.buffer)
            self.buffer.clear()
            self.file.flush()  # force write to disk

    def close(self):
        self.flush()
        self.file.close()

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
