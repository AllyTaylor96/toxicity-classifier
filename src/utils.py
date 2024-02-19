import argparse
import logging
import os
import json


def configure_logging(logname):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/{logname}.log', mode='w'),  # Output to a file
            logging.StreamHandler() # Output to console
        ]
    )

def load_json(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    return json_data

def config_parser():
    parser = argparse.ArgumentParser(description='Config path reader for data prep')
    parser.add_argument('-c', '--config_path', type=str, help='Path to the config file')
    args = parser.parse_args()
    return args

