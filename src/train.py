import os
import argparse
from train_data import retrieve_dataset, preprocess_dataset


""" Main driver function for training. Will go through the steps in sequence.

1. Data retrieval - (grabbing OxAISH-AL-LLM/wiki_toxic) dataset using datasets.
2. Data pre-processing - clean + format the data into FastText appropriate format and save to file.
3. Model training - train a binary 'toxicity' classifier using FastText, and tune the threshold.

Can potentially use model on Twitch comments etc. to ascertain toxicity? An additional project.

"""



def main():

    # data retrieval and prep
    train_dataset, val_dataset, test_dataset = retrieve_dataset()
    processed_train = preprocess_dataset(train_dataset)
    processed_val = preprocess_dataset(train_dataset)
    processed_test = preprocess_dataset(train_dataset)

    print('Hello world!')

if __name__ == "__main__":
    main()
