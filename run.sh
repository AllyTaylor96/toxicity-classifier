#!/bin/bash

# Check if keyword is provided
if [ $# -eq 0 ]; then
	echo "Usage: $0 <keyword> -> Options: [prep, train, test]"
	exit 1
fi

# Check the provided keyword
if [ "$1" = "prep" ]; then
    python3 src/data_prep.py -c config/prep_config.json
elif [ "$1" = "train" ]; then
	python3 src/train.py -c config/train_config.json
elif [ "$1" = "test" ]; then
	echo "Testing functionality goes here"
else
	echo "Invalid keyword. Supported keywords are 'train' and 'test'."
	exit 1
fi
