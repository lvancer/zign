import logging
import sys


time_format = '[%(asctime)s] - %(levelname)s: %(message)s'

def init(level=logging.INFO):
    logging.basicConfig(level=level, stream=sys.stdout, format=time_format)
    
def add_file_handler(filepath, level=logging.INFO):
    handler = logging.FileHandler(filepath)
    handler.setLevel(level)
    formatter = logging.Formatter(time_format)
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)






        