import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_functions import ToxicDataset
from train_model import Simple_CNN

def check_for_cuda():
    """Sets global device as either CPU or GPU."""

    global device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
        logging.info('Device name:' + str(torch.cuda.get_device_name(0)))

    else:

        logging.warning('No GPU available, using the CPU instead - is this intended?')
        device = torch.device("cpu")

    return device


def set_up_dataloaders(processed_dataset, batch_size):
    """Returns dataloaders for train/val/test and logs sizes."""

    trainingDataset = ToxicDataset(processed_dataset['balanced_train'])
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

    return train_dataloader, val_dataloader, test_dataloader


def init_model(word_embeddings, filter_sizes, num_filters, num_classes, dropout, lr):
    """Instances both the model and relevant optimizer."""

    cnn_model = Simple_CNN(pretrained_embedding=word_embeddings,
                           filter_sizes=filter_sizes,
                           num_filters=num_filters,
                           num_classes=num_classes,
                           dropout=dropout)

    cnn_model.to(device)

    optimizer = optim.Adadelta(cnn_model.parameters(),
                               lr=lr,
                               rho=0.95)

    return cnn_model, optimizer


def train(model, optimizer, train_dataloader, val_dataloader, test_dataloader, model_dir, epochs=5):
    """Main training loop function."""

    best_val_accuracy = 0
    best_test_accuracy = 0
    loss_fn = nn.CrossEntropyLoss()

    epoch_is, train_losses, val_losses, val_accs, test_accs = [], [], [], [], []

    # start training loop
    for epoch_i in range(epochs):
        epoch_i = epoch_i + 1
        epoch_is.append(epoch_i)
        logging.info(f"Epoch: {epoch_i}")
        total_loss = 0
        t0_epoch = time.time()

        logging.info(f"Training...")

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


        if val_dataloader is not None:
            logging.info(f"Validating...")

            val_loss, val_accuracy = evaluate(model, val_dataloader)
            logging.info(f"Val Loss: {val_loss:.2f}")
            logging.info(f"Val Accuracy: {val_accuracy:.2f}")
            val_losses.append(val_loss)
            val_accs.append(val_accuracy)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy



        if test_dataloader is not None:
            logging.info(f"Testing...")

            test_loss, test_accuracy = evaluate(model, test_dataloader)
            logging.info(f"Test Accuracy: {test_accuracy:.2f}")
            test_accs.append(test_accuracy)

            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                logging.info(f"New best model found on test set - test accuracy {best_test_accuracy:.2f}% - saving...")
                torch.save(model.state_dict(), model_dir + '/best_model.pt')

        time_elapsed = time.time() - t0_epoch
        logging.info(f"Epoch complete: time taken {time_elapsed:.2f}s")

    logging.info(f"\nTraining complete! Best test accuracy: {best_test_accuracy:.2f}%.")

    result_dict = {
        'Epoch': epoch_is,
        'Train Loss': train_losses,
        'Validation Loss': val_losses,
        'Validation Accuracy': val_accs,
        'Test Accuracy': test_accs
    }

    result_df = pd.DataFrame(data=result_dict)

    return model, result_df


def evaluate(model, dataloader):
    """ Assess model on given dataloader: agnostic so works for val and test."""

    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    accuracies, losses = [], []

    for batch in tqdm(dataloader):

        b_input_ids, b_labels = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            logits = model(b_input_ids)

        # calculate batch loss, preds and accuracy

        loss = loss_fn(logits, b_labels)
        losses.append(loss.item())

        preds = torch.argmax(logits, dim=1).flatten()

        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        accuracies.append(accuracy)

    # calculate overall loss and accuracy

    loss = np.mean(losses)
    accuracy = np.mean(accuracies)

    return loss, accuracy

