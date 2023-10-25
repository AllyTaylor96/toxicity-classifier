import os
import datasets
import torch
from torch.utils.data import Dataset

print(torch.cuda.is_available())


def retrieve_dataset():
    full_dataset = datasets.load_dataset("OxAISH-AL-LLM/wiki_toxic")
    train_dataset = full_dataset['balanced_train']
    val_dataset = full_dataset['validation']
    test_dataset = full_dataset['test']

    return train_dataset, val_dataset, test_dataset


def preprocess_dataset(dataset_obj):
    """Convert text files to tensors using Fasttext?"""
    print('processing...')

    return dataset_obj

def vectorize(sentence):
    """ Use FT to vectorize the sentences into embeddings."""

    # need to load in fasttext model as part of docker image prep etc.
    # ft = fasttext.load_model('cc.en.300.bin')
    # sentence_embedding = ft.get_word_vector('hello')

    return sentence_embedding


class ToxicDataset(Dataset):
    def __init__(self, dataset_obj):
        self.dataset = dataset_obj

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sentence = self.dataset[idx]['comment_text']
        label = self.dataset[idx]['label']

        sentence_embedding = vectorize(sentence)

        return sentence_embedding, label
