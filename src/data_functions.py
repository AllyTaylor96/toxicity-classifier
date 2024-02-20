import os
import re
import string
import logging
import datasets
import requests
import shutil
from collections import Counter
import fasttext
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset

def retrieve_dataset(hf_dataset_name, data_dir):
    hf_dataset_path = data_dir + '/hf_dataset'
    logging.info(f'HF Dataset cached at {hf_dataset_path}')
    full_dataset = datasets.load_dataset(hf_dataset_name, cache_dir=hf_dataset_path)
    return full_dataset





def retrieve_word_vecs(data_dir, vec_url):

    # define FastText URLs
    vec_name = vec_url.split('/')[-1]

    # check if word vectors already there
    output_filepath = data_dir + '/' + vec_name


    if os.path.isfile(output_filepath.replace('.zip', '')):
        logging.info(f'Word vec file already found at {output_filepath} - skipping download...')

    else:
        # if not, download them from the FastText site
        logging.info(f'Downloading Fasttext Vectors from {vec_url}...')

        r = requests.get(vec_url, stream=True)
        total_len = int(r.headers.get('content-length', 0))
        block_size = 1024

        # download in blocks using requests
        with tqdm(total=total_len, unit='B', unit_scale=True) as progress_bar:
            with open(output_filepath, 'wb') as f:
                for data in r.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)

            if total_len != 0 and progress_bar.n != total_len:
                logging.error("Could not download FastText Vectors")
                raise RuntimeError("Could not download FastText Vectors")

        # unzip file
        logging.info('unzipping word vecs...')
        shutil.unpack_archive(output_filepath, extract_dir=data_dir)

    word_vec_path = output_filepath.replace('.zip', '')
    return word_vec_path

def clean_tokenize_text(example):
    """ Clean + tokenize text from dataset and append to new columns. """

    # cleaning
    norm_text = example['comment_text'].lower()
    exclude = set(string.punctuation)
    norm_text = re.sub(r"\[.*?]", "", norm_text)
    norm_text = ''.join(ch for ch in norm_text if ch not in exclude)
    norm_text = norm_text.replace('\n', '')
    example['norm_comment_text'] = norm_text

    # tokenizing
    tokenized_text = fasttext.tokenize(example['norm_comment_text'])
    example['tokenized_text'] = tokenized_text

    # get sentence length for filtering later
    example['sentWordCount'] = len(example['tokenized_text'])

    return example

def encode_text(example, vocab_dict, max_sent_len):

    tokenized_text = example['tokenized_text']

    # pad the text out to max sent len
    tokenized_text += ['<pad>'] * (max_sent_len - len(tokenized_text))

    # encode the padded tokens to their IDs using the vocab dict
    encoded_text = [vocab_dict.get(token, 1) for token in tokenized_text]

    example['encoded_text'] = np.array(encoded_text)

    return example


def preprocess_dataset(dataset_obj, maxSentLen):
    """Clean + tokenize text, filter on size, build vocab, find max sent length and encode. """

    logging.info('Cleaning and tokenizing HF dataset...')
    dataset_obj = dataset_obj.map(clean_tokenize_text, batched=False, load_from_cache_file=True)

    logging.info('Filtering on max word length...')
    logging.info(f"Dataset size before filtering: \n\
    Train: {len(dataset_obj['train'])} \n\
    Validation: {len(dataset_obj['validation'])} \n\
    Test: {len(dataset_obj['test'])}")

    dataset_obj = dataset_obj.filter(lambda example: example['sentWordCount'] < maxSentLen)

    logging.info(f"Dataset size after filtering: \n\
    Train: {len(dataset_obj['train'])} \n\
    Validation: {len(dataset_obj['validation'])} \n\
    Test: {len(dataset_obj['test'])}")

    logging.info('Building vocabulary dict...')
    vocab_dict = {}
    vocab_dict['<pad>'] = 0
    vocab_dict['<unk>'] = 1

    max_sent_len = 0
    idx = 2
    for split in ['train', 'validation', 'test']:
        logging.info(f'Checking {split} data for unknown vocab...')
        for tokenized_text in tqdm(list(dataset_obj[split]['tokenized_text']), total=len(dataset_obj[split])):

            # check if token in vocab, and if not add
            for token in tokenized_text:
                if token not in vocab_dict:
                    vocab_dict[token] = idx
                    idx += 1

            # update max sent length
            max_sent_len = max(max_sent_len, len(tokenized_text))

    # now use the vocab dict for sentence encoding
    logging.info('Encoding HF dataset using vocabulary dict...')
    dataset_obj = dataset_obj.map(encode_text, batched=False, load_from_cache_file=True,
                                  fn_kwargs={'vocab_dict': vocab_dict,
                                             'max_sent_len': max_sent_len})

    return dataset_obj, vocab_dict, max_sent_len

def load_word_vectors(vocab_dict, word_vector_filepath):

    with open(word_vector_filepath, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        vocab_dict_size, embedding_dim_size = map(int, f.readline().split())

        # make random vectors first of all
        embeddings = np.random.uniform(-0.25, 0.25, (len(vocab_dict), embedding_dim_size))
        embeddings[vocab_dict['<pad>']] = np.zeros((embedding_dim_size,))

        # load in pretrained vectors
        pretrained_count = 0
        for line in tqdm(f, total=vocab_dict_size):
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            if word in vocab_dict:
                pretrained_count += 1
                embeddings[vocab_dict[word]] = np.array(tokens[1:], dtype=np.float32)

    logging.info(f'There are {pretrained_count} / {len(vocab_dict)} pretrained vectors found from training data.')

    return embeddings



class ToxicDataset(Dataset):
    def __init__(self, dataset_obj):
        self.dataset = dataset_obj

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        encoded_input = self.dataset[idx]['encoded_text']
        label = self.dataset[idx]['label']

        return (torch.tensor(encoded_input), torch.tensor(label))
