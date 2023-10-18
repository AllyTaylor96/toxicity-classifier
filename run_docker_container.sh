#!/bin/bash


docker run --rm -it -v /home/ally_taylor:/home/ally_taylor \
	-w ${PWD} \
	toxicity-classifier bash
