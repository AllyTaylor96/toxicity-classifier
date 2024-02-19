import os
import argparse
import pickle
import logging
from datasets import load_from_disk
from data_functions import ToxicDataset
from train_model import *
from utils import config_parser, configure_logging, load_json
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


""" Main driver function for training. Will go through the steps in sequence.

1. Data retrieval - (grabbing OxAISH-AL-LLM/wiki_toxic) dataset using datasets.
2. Data pre-processing - clean + format the data into FastText appropriate format and save to file.
3. Model training - train a binary 'toxicity' classifier using FastText, and tune the threshold.

Can potentially use model on Twitch comments etc. to ascertain toxicity? An additional project.

https://chriskhanhtran.github.io/posts/cnn-sentence-classification/

"""

def train(model, optimizer, train_dataloader, val_dataloader, epochs=20, device='cpu'):

    best_accuracy = 0

    # start training loop
    for epoch_i in range(epochs):
        total_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):

            b_input_ids, b_labels = tuple(t.to(device) for t in batch)
            exit()

def main():

    args = config_parser()
    config = load_json(args.config_path)
    configure_logging('train')

    if torch.cuda.is_available():

        device = torch.device("cuda")
        logging.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
        logging.info('Device name:', torch.cuda.get_device_name(0))

    else:

        logging.warning('No GPU available, using the CPU instead - is this intended?')
        device = torch.device("cpu")

    logging.info('Loading data files...')

    processed_dataset = load_from_disk(config['dataDir'] + '/processed_dataset')
    with open(config['dataDir'] + '/word_embeddings.pkl', 'rb') as f:
        word_embeddings = pickle.load(f)
        vocab_size, embed_dim = word_embeddings.shape


    processed_train = processed_dataset['train']
    processed_val = processed_dataset['validation']
    processed_test = processed_dataset['test']

    batch_size = config['batchSize']
    logging.info(f'creating Torch Datasets with batch size {batch_size}...')

    trainingDataset = ToxicDataset(processed_train)
    train_sampler = RandomSampler(trainingDataset)
    train_dataloader = DataLoader(trainingDataset, sampler=train_sampler, batch_size=batch_size)
    logging.info('How many comments in training: ', trainingDataset.__len__())

    valDataset = ToxicDataset(processed_val)
    val_sampler = SequentialSampler(valDataset)
    val_dataloader = DataLoader(valDataset, sampler=val_sampler, batch_size=batch_size)
    logging.info('How many comments in validation: ', valDataset.__len__())

    testDataset = ToxicDataset(processed_test)
    test_sampler = SequentialSampler(testDataset)
    test_dataloader = DataLoader(testDataset, sampler=test_sampler, batch_size=batch_size)
    logging.info('How many comments in testing: ', testDataset.__len__())

    # now we move to training the actual model
    logging.info('Model training beginning...')
    model = Simple_CNN(pretrained_embedding=word_embeddings,
                       vocab_size=vocab_size,
                       embed_dim=embed_dim)

    train(model, 'optimize', train_dataloader, val_dataloader, epochs=10, device='cpu')
    # for step, batch in enumerate(train_dataloader):
    #     batch_inputs, batch_labels = tuple(t for t in batch)
    #     print('batch inputs type: ', type(batch_inputs))
    #     logits = model(batch_inputs)
        # exit()

if __name__ == "__main__":
    main()
