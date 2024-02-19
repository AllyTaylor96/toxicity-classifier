import argparse
import logging
import pickle
from data_functions import *
from utils import configure_logging, load_json, config_parser

def main():

    # load config
    args = config_parser()
    config = load_json(args.config_path)

    configure_logging('data_prep')
    logging.info('Creating dataset files...')

    # huggingface pulling + cleaning

    logging.info('Pulling HuggingFace Dataset...')
    full_dataset = retrieve_dataset(config['hfDatasetName'], config['dataDir'])
    del full_dataset['balanced_train']

    logging.info('Cleaning and tokenizing...')
    processed_dataset, vocab_dict, max_sent_len = preprocess_dataset(full_dataset)

    # fastText pulling + mapping to huggingface vocab

    logging.info('Downloading FastText word vectors...')
    word_vectors_path = retrieve_word_vecs(config['dataDir'], config['ftVectorUrl'])

    logging.info('Mapping FastText word vectors to existing vocab dict...')
    word_embeddings = torch.tensor(load_word_vectors(vocab_dict, word_vectors_path))

    # saving files

    logging.info('Saving HF dataset files to disk for easy loading for training')

    hf_dataset_path = config['dataDir'] + '/processed_dataset'
    word_emb_path = config['dataDir'] + '/word_embeddings.pkl'

    processed_dataset.save_to_disk(hf_dataset_path)
    with open(word_emb_path, 'wb') as f:
        pickle.dump(word_embeddings, f)

    logging.info(f'Processed HF dataset saved at {hf_dataset_path}...')
    logging.info(f'Final embeddings saved at {word_emb_path}')

if __name__ == "__main__":
    main()

