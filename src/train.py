import os
import argparse
import fasttext.util
from train_data import retrieve_dataset, preprocess_dataset, ToxicDataset
from torch.utils.data import DataLoader


""" Main driver function for training. Will go through the steps in sequence.

1. Data retrieval - (grabbing OxAISH-AL-LLM/wiki_toxic) dataset using datasets.
2. Data pre-processing - clean + format the data into FastText appropriate format and save to file.
3. Model training - train a binary 'toxicity' classifier using FastText, and tune the threshold.

Can potentially use model on Twitch comments etc. to ascertain toxicity? An additional project.

"""



def main():

    # data retrieval and prep
    print('retrieving datasets...')
    train_dataset, val_dataset, test_dataset = retrieve_dataset()

    # data cleaning
    print('cleaning datasets...')
    processed_train = preprocess_dataset(train_dataset)
    processed_val = preprocess_dataset(val_dataset)
    processed_test = preprocess_dataset(test_dataset)

    # download the FastText model (caution, is 4GB in size)
    print('grabbing FastText base English encoder...')
    fasttext.util.download_model('en', if_exists='ignore')

    # loading the FastText model
    print('loading FastText model...')
    ft = fasttext.load_model('cc.en.300.bin')

    # reduce FT dimension to 100 as unnecessary for 300
    fasttext.util.reduce_model(ft, 300)


    # pass into Pytorch datasets
    print('creating Torch Datasets...')
    trainingDataset = ToxicDataset(processed_train, ft)
    print('How many comments in training: ', trainingDataset.__len__())
    valDataset = ToxicDataset(processed_val, ft)
    print('How many comments in validation: ', valDataset.__len__())
    testDataset = ToxicDataset(processed_test, ft)
    print('How many comments in testing: ', testDataset.__len__())

    # now we move to training the actual model
    print('Model training begins...')

if __name__ == "__main__":
    main()
