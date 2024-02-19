import os
import argparse
import pickle
import logging
import time
from datasets import load_from_disk
from data_functions import ToxicDataset
from train_model import Simple_CNN
from utils import config_parser, configure_logging, load_json
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


""" Main driver function for training. Will go through the steps in sequence.

1. Data retrieval - (grabbing OxAISH-AL-LLM/wiki_toxic) dataset using datasets.
2. Data pre-processing - clean + format the data into FastText appropriate format and save to file.
3. Model training - train a binary 'toxicity' classifier using FastText, and tune the threshold.

Can potentially use model on Twitch comments etc. to ascertain toxicity? An additional project.

https://chriskhanhtran.github.io/posts/cnn-sentence-classification/

"""


def init_model(word_embeddings, filter_sizes, num_filters, num_classes, dropout, lr, device):
    # initialize cnn model
    cnn_model = Simple_CNN(pretrained_embedding=word_embeddings,
                           filter_sizes=filter_sizes,
                           num_filters=num_filters,
                           num_classes=num_classes,
                           dropout=dropout)

    # send model to the appropriate device
    cnn_model.to(device)

    # set up optimizer
    optimizer = optim.Adadelta(cnn_model.parameters(),
                               lr=lr,
                               rho=0.95)

    return cnn_model, optimizer



def train(model, optimizer, train_dataloader, val_dataloader, epochs=20, device='cpu'):

    best_accuracy = 0
    loss_fn = nn.CrossEntropyLoss()

    # start training loop
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
    print("-" * 60)

    for epoch_i in range(epochs):
        total_loss = 0
        t0_epoch = time.time()

        model.train()

        for step, batch in enumerate(tqdm(train_dataloader)):

            b_input_ids, b_labels = tuple(t.to(device) for t in batch)

            model.zero_grad()

            logits = model(b_input_ids)

            loss = loss_fn(logits, b_labels)
            total_loss += loss.item()

            loss.backward()

            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        if val_dataloader is not None:

            val_loss, val_accuracy = evaluate(model, val_dataloader)
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")

    print("\nTraining complete! Best accuracy: {best_accuracy:.2f}%.")
    torch.save(model.state_dict(), config['modelDir'] + '/initial_model.pt')



def evaluate(model, val_dataloader):
    model.eval()

    val_accuracy, val_loss = [], []

    for batch in val_dataloader:
        b_input_ids, b_labels = tuple(t.to(device) for t in batch)

        # Get batch logits
        with torch.no_grad():
            logits = model(b_input_ids)

        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # get preds
        preds = torch.argmax(logits, dim=1).flatten()

        # calculate accuracy
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # calculate overall accuracy and loss on val set
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy

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
    print(processed_val['encoded_text'][1])
    exit()
    train_sampler = RandomSampler(trainingDataset)
    train_dataloader = DataLoader(trainingDataset, sampler=train_sampler, batch_size=batch_size)
    logging.info('How many comments in training: ', str(trainingDataset.__len__()))

    valDataset = ToxicDataset(processed_val)
    val_sampler = SequentialSampler(valDataset)
    val_dataloader = DataLoader(valDataset, sampler=val_sampler, batch_size=batch_size)

    exit()
    logging.info('How many comments in validation: ', str(valDataset.__len__()))

    testDataset = ToxicDataset(processed_test)
    test_sampler = SequentialSampler(testDataset)
    test_dataloader = DataLoader(testDataset, sampler=test_sampler, batch_size=batch_size)
    logging.info('How many comments in testing: ', str(testDataset.__len__()))

    # now we move to training the actual model
    logging.info('Model training beginning...')

    model, optimizer = init_model(word_embeddings=word_embeddings,
                                  filter_sizes=config['cnnFilterSizes'],
                                  num_filters=config['cnnNumFilters'],
                                  num_classes=config['cnnNumClasses'],
                                  dropout=config['cnnDropout'],
                                  lr=config['cnnLr'],
                                  device=device)


    train(model, optimizer, train_dataloader, val_dataloader, epochs=10, device=device)

if __name__ == "__main__":
    main()
