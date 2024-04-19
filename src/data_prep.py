import logging
import pickle

from data_functions import retrieve_dataset, clear_dataset_cache, preprocess_dataset
from data_functions import retrieve_word_vectors, load_word_vectors
from utils import configure_logging, load_json, config_parser, check_for_folders

def main():

    args = config_parser()
    config = load_json(args.config_path)
    repo_dir, data_dir, model_dir = check_for_folders(config)


    configure_logging('data_prep')
    logging.info('Creating dataset files...')

    logging.info('Pulling HuggingFace Dataset...')

    full_dataset = retrieve_dataset(config['hfDatasetName'], str(data_dir))

    del full_dataset['train']  # as imbalanced

    logging.info('Cleaning, tokenizing and building the vocabulary...')

    processed_dataset, vocab_dict, max_sent_len = preprocess_dataset(full_dataset, config['maxSentWordLen'])

    logging.info('Downloading FastText word vectors...')

    word_vectors_path = retrieve_word_vectors(str(data_dir), config['ftVectorUrl'])

    logging.info('Mapping FastText word vectors to existing vocab dict...')
    word_embeddings = load_word_vectors(vocab_dict, word_vectors_path)

    logging.info('Saving processed HF dataset files to disk for easy loading for training')

    hf_dataset_path = data_dir / 'processed_dataset'
    word_emb_path = data_dir / '/word_embeddings.pkl'
    vocab_dict_path = data_dir / '/vocab_dict.pkl'

    processed_dataset.save_to_disk(str(hf_dataset_path))

    with open(str(word_emb_path), 'wb') as f:
        pickle.dump(word_embeddings, f)

    with open(str(vocab_dict_path), 'wb') as f:
        pickle.dump(vocab_dict, f)

    logging.info(f'Vocab dict saved at {vocab_dict_path}')
    logging.info(f'Processed HF dataset saved at {hf_dataset_path}...')
    logging.info(f'Final embeddings saved at {word_emb_path}')

    if config['keepHfDatasetCached'] == False:
        logging.info("Deleting HF cache as request...")
        clear_dataset_cache(config['hfDatasetName'], str(data_dir))


if __name__ == "__main__":
    main()

