import logging
import pickle

from data_functions import retrieve_dataset, clear_dataset_cache, preprocess_dataset
from data_functions import retrieve_word_vectors, load_word_vectors
from utils import configure_logging, load_json, config_parser

def main():

    args = config_parser()
    config = load_json(args.config_path)

    configure_logging('data_prep')
    logging.info('Creating dataset files...')

    logging.info('Pulling HuggingFace Dataset...')

    full_dataset = retrieve_dataset(config['hfDatasetName'], config['dataDir'])

    """
    We have an imbalanced dataset from HF - in total, there are ~200k non-toxic comments
    and only ~22.5k toxic. Therefore, we use the balanced_train set from HF, which splits
    toxic/non-toxic 50/50 (12934 comments in each), while the validation and test sets
    contain that original dataset imbalance (about 1 in 10 comments are toxic).

    Data distributions:
        training set: ({non-toxic: 12934, toxic: 12934})
        validation set: ({non-toxic: 28624, toxic: 3291})
        test set: ({non-toxic: 57735, toxic: 6243})
    """

    del full_dataset['train']  # as imbalanced - see above

    logging.info('Cleaning, tokenizing and building the vocabulary...')

    processed_dataset, vocab_dict, max_sent_len = preprocess_dataset(full_dataset, config['maxSentWordLen'])

    logging.info('Downloading FastText word vectors...')

    word_vectors_path = retrieve_word_vectors(config['dataDir'], config['ftVectorUrl'])

    logging.info('Mapping FastText word vectors to existing vocab dict...')
    word_embeddings = load_word_vectors(vocab_dict, word_vectors_path)

    logging.info('Saving processed HF dataset files to disk for easy loading for training')

    hf_dataset_path = config['dataDir'] + '/processed_dataset'
    word_emb_path = config['dataDir'] + '/word_embeddings.pkl'
    vocab_dict_path = config['dataDir'] + '/vocab_dict.pkl'

    processed_dataset.save_to_disk(hf_dataset_path)

    with open(word_emb_path, 'wb') as f:
        pickle.dump(word_embeddings, f)

    with open(vocab_dict_path, 'wb') as f:
        pickle.dump(vocab_dict, f)

    logging.info(f'Vocab dict saved at {vocab_dict_path}')
    logging.info(f'Processed HF dataset saved at {hf_dataset_path}...')
    logging.info(f'Final embeddings saved at {word_emb_path}')

    if config['keepHfDatasetCached'] == False:
        logging.info("Deleting HF cache as request...")
        clear_dataset_cache(config['hfDatasetName'], config['dataDir'])


if __name__ == "__main__":
    main()

