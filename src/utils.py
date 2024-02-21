import argparse
import json
import logging
import os


def configure_logging(logname):
    """Sets up logger to have sensible format and to output to both file and console."""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/{logname}.log', mode='w'),
            logging.StreamHandler()
        ]
    )


def load_json(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    return json_data

def config_parser():
    """Parses the config path given for every set of functionality."""

    parser = argparse.ArgumentParser(description='Config path reader for data prep')
    parser.add_argument('-c', '--config_path', type=str, help='Path to the config file')
    args = parser.parse_args()
    return args

