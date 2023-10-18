import os

""" Main driver function for training. Will go through the steps in sequence.

1. Data retrieval - (grabbing OxAISH-AL-LLM/wiki_toxic) dataset using datasets.
2. Data pre-processing - clean + format the data into FastText appropriate format and save to file.
3. Model training - train a binary 'toxicity' classifier using FastText, and tune the threshold.

Can potentially use model on Twitch comments etc. to ascertain toxicity? An additional project.

"""


