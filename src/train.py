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
import warnings
warnings.filterwarnings("ignore", message="promote has been superseded by promote_options='default'.", category=FutureWarning, module="datasets")
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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



def train(model, optimizer, train_dataloader, val_dataloader, test_dataloader, model_dir, epochs=20):

    best_val_accuracy = 0
    best_test_accuracy = 0
    loss_fn = nn.CrossEntropyLoss()

    # start training loop
    epoch_is, train_losses, val_losses, val_accs, test_accs = [], [], [], [], []

    for epoch_i in range(epochs):
        epoch_is.append(epoch_i)
        logging.info(f"Epoch: {epoch_i}")
        total_loss = 0
        t0_epoch = time.time()

        # training

        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):

            b_input_ids, b_labels = tuple(t.to(device) for t in batch)

            model.zero_grad()

            logits = model(b_input_ids)

            loss = loss_fn(logits, b_labels)
            total_loss += loss.item()

            loss.backward()

            optimizer.step()

        epoch_train_loss = total_loss / len(train_dataloader)
        logging.info(f"Train Loss: {epoch_train_loss:.2f}")
        train_losses.append(epoch_train_loss)

        logging.info(f"Validating...")

        if val_dataloader is not None:

            val_loss, val_accuracy = evaluate(model, val_dataloader)
            logging.info(f"Val Loss: {val_loss:.2f}")
            logging.info(f"Val Accuracy: {val_accuracy:.2f}")
            val_losses.append(val_loss)
            val_accs.append(val_accuracy)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy


        logging.info(f"Testing...")

        if test_dataloader is not None:

            test_loss, test_accuracy = evaluate(model, test_dataloader)
            logging.info(f"Test Accuracy: {test_accuracy:.2f}")
            test_accs.append(test_accuracy)

            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                logging.info(f"New best model found on test set - test accuracy {best_test_accuracy:.2f}% - saving...")
                torch.save(model.state_dict(), model_dir + '/best_model.pt')

        time_elapsed = time.time() - t0_epoch
        logging.info(f"Epoch complete: time taken {time_elapsed:.2f}")

    logging.info(f"\nTraining complete! Best test accuracy: {best_test_accuracy:.2f}%.")

    return model

def evaluate(model, dataloader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    accuracies, losses = [], []

    for batch in tqdm(dataloader):
        b_input_ids, b_labels = tuple(t.to(device) for t in batch)

        # Get batch logits
        with torch.no_grad():
            logits = model(b_input_ids)

        loss = loss_fn(logits, b_labels)
        losses.append(loss.item())

        # get preds
        preds = torch.argmax(logits, dim=1).flatten()

        # calculate accuracy
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        accuracies.append(accuracy)

    # calculate overall accuracy and loss on val set
    loss = np.mean(losses)
    accuracy = np.mean(accuracies)

    return loss, accuracy

def main():

    args = config_parser()
    config = load_json(args.config_path)
    configure_logging('train')
    global device

    if torch.cuda.is_available():

        device = torch.device("cuda")
        logging.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
        logging.info('Device name:' + str(torch.cuda.get_device_name(0)))

    else:

        logging.warning('No GPU available, using the CPU instead - is this intended?')
        device = torch.device("cpu")

    logging.info('Loading data files...')

    processed_dataset = load_from_disk(config['dataDir'] + '/processed_dataset')
    with open(config['dataDir'] + '/word_embeddings.pkl', 'rb') as f:
        word_embeddings = pickle.load(f)
        vocab_size, embed_dim = word_embeddings.shape

    batch_size = config['batchSize']
    logging.info(f'creating Torch Datasets with batch size {batch_size}...')

    trainingDataset = ToxicDataset(processed_dataset['train'])
    train_sampler = RandomSampler(trainingDataset)
    train_dataloader = DataLoader(trainingDataset, sampler=train_sampler, batch_size=batch_size)
    logging.info('How many comments in training: ' + str(trainingDataset.__len__()))

    valDataset = ToxicDataset(processed_dataset['validation'])
    val_sampler = SequentialSampler(valDataset)
    val_dataloader = DataLoader(valDataset, sampler=val_sampler, batch_size=batch_size)
    logging.info('How many comments in validation: ' + str(valDataset.__len__()))

    testDataset = ToxicDataset(processed_dataset['test'])
    test_sampler = SequentialSampler(testDataset)
    test_dataloader = DataLoader(testDataset, sampler=test_sampler, batch_size=batch_size)
    logging.info('How many comments in testing: ' + str(testDataset.__len__()))

    # now we move to training the actual model
    logging.info('Model training beginning...')

    model, optimizer = init_model(word_embeddings=word_embeddings,
                                  filter_sizes=config['cnnFilterSizes'],
                                  num_filters=config['cnnNumFilters'],
                                  num_classes=config['cnnNumClasses'],
                                  dropout=config['cnnDropout'],
                                  lr=config['cnnLr'],
                                  device=device)


    trained_model = train(model, optimizer, train_dataloader, val_dataloader, test_dataloader, config['modelDir'], epochs=20)

if __name__ == "__main__":
    main()
