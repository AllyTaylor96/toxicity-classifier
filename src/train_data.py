import os
import re
import string
import datasets
import torch
from torch.utils.data import Dataset

print(torch.cuda.is_available())


def retrieve_dataset():
    full_dataset = datasets.load_dataset("OxAISH-AL-LLM/wiki_toxic")
    train_dataset = full_dataset['train']
    val_dataset = full_dataset['validation']
    test_dataset = full_dataset['test']

    return train_dataset, val_dataset, test_dataset

def normalise(example):
    """ Clean text from dataset and append to new column. """

    # text cleaning
    norm_text = example['comment_text'].lower()
    exclude = set(string.punctuation)
    norm_text = re.sub(r"\[.*?]", "", norm_text)
    norm_text = ''.join(ch for ch in norm_text if ch not in exclude)
    norm_text = norm_text.replace('\n', '')
    example['norm_comment_text'] = norm_text

    return example


def preprocess_dataset(dataset_obj):
    """Converts raw text from dataset into normalised clean sentences for encoding. """

    dataset_obj = dataset_obj.map(normalise, batched=False, load_from_cache_file=True)

    return dataset_obj


class ToxicDataset(Dataset):
    def __init__(self, dataset_obj, ft_model):
        self.dataset = dataset_obj
        self.ft_model = ft_model

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sentence = self.dataset[idx]['norm_comment_text']
        label = self.dataset[idx]['label']

        sentence_embedding = self.ft_model.get_sentence_vector(sentence)

        return sentence_embedding, label
