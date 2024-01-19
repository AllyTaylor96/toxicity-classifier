import os
import argparse
import fasttext.util
from train_data import *
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


""" Main driver function for training. Will go through the steps in sequence.

1. Data retrieval - (grabbing OxAISH-AL-LLM/wiki_toxic) dataset using datasets.
2. Data pre-processing - clean + format the data into FastText appropriate format and save to file.
3. Model training - train a binary 'toxicity' classifier using FastText, and tune the threshold.

Can potentially use model on Twitch comments etc. to ascertain toxicity? An additional project.

"""



def main():

    # data retrieval and prep
    print('retrieving dataset...')
    full_dataset = retrieve_dataset()

    # data cleaning
    print('cleaning and tokenizing dataset...')
    processed_dataset, vocab_dict, max_sent_len = preprocess_dataset(full_dataset)
    print(processed_dataset['train'][0])
    print(len(vocab_dict))


    # processed_train = preprocess_dataset(train_dataset)
    # processed_val = preprocess_dataset(val_dataset)
    # processed_test = preprocess_dataset(test_dataset)

    """ Need to follow procedure in building vocab and getting max sent length
    for CNN model to work properly - can follow steps in tutorial
    https://chriskhanhtran.github.io/posts/cnn-sentence-classification/ """

    # download FT word vectors
    vec_name = "wiki-news-300d-1M.vec.zip"
    vector_path = retrieve_word_vecs(vec_name)
    exit()

    # loading the FastText model
    # print('loading FastText model...')
    # ft = fasttext.load_model('cc.en.300.bin')

    # print(help(ft))
    # reduce FT dimension to 100 as unnecessary for 300
    # fasttext.util.reduce_model(ft, 100)


    # pass into Pytorch datasets
    print('creating Torch Datasets...')
    trainingDataset = ToxicDataset(processed_train, ft)
    print('How many comments in training: ', trainingDataset.__len__())
    valDataset = ToxicDataset(processed_val, ft)
    print('How many comments in validation: ', valDataset.__len__())
    testDataset = ToxicDataset(processed_test, ft)
    print('How many comments in testing: ', testDataset.__len__())

    # set up Dataloaders
    print('setting up Torch Dataloaders...')
    batch_size = 30

    train_sampler = RandomSampler(trainingDataset)
    train_dataloader = DataLoader(trainingDataset, sampler=train_sampler, batch_size=batch_size)

    val_sampler = SequentialSampler(valDataset)
    val_dataloader = DataLoader(valDataset, sampler=val_sampler, batch_size=batch_size)

    test_sampler = SequentialSampler(testDataset)
    test_dataloader = DataLoader(testDataset, sampler=test_sampler, batch_size=batch_size)

    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    # now we move to training the actual model
    print('Model training begins...')
    print(train_features)


if __name__ == "__main__":
    main()
