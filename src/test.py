import logging
import pickle

from utils import configure_logging, load_json, config_parser
from test_functions import load_best_model, predict


def main():

    args = config_parser()
    config = load_json(args.config_path)
    configure_logging('test')

    best_model_path = config['modelDir'] + '/best_model.pt'

    logging.info(f'Loading best saved model: {best_model_path}')

    # not a fan of needing to define this here - a better way?

    with open(config['dataDir'] + '/word_embeddings.pkl', 'rb') as f:
        word_embeddings = pickle.load(f)

    with open(config['dataDir'] + '/vocab_dict.pkl', 'rb') as f:
        vocab_dict = pickle.load(f)

    model = load_best_model(best_model_path, word_embeddings)

    test_sentences = config['testComments']

    for sentence in test_sentences:
        logging.info(f'Inference on: {sentence}')
        probs = predict(sentence, model, vocab_dict)
        if probs[1] > 0.5:
            logging.info(f"This comment is toxic. (Toxic probability: {probs[1] * 100:.2f}%)\n")
        else:
            logging.info(f"This comment is not toxic. (Toxic probability: {probs[1] * 100:.2f}%)\n")







if __name__ == "__main__":
    main()

