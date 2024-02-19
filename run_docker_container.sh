#!/bin/bash


docker run --rm -it -v ${PWD}:${PWD} \
	-w ${PWD} \
	--gpus all \
	toxicity-classifier bash
