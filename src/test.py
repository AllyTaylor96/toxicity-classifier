import logging
import pickle

from utils import configure_logging, load_json, config_parser
from test_functions import load_best_model


def main():

    args = config_parser()
    config = load_json(args.config_path)
    configure_logging('test')

    best_model_path = config['modelDir'] + '/best_model.pt'

    logging.info(f'Loading best saved model: {best_model_path}')

    # not a fan of needing to define this here - a better way?

    with open(config['dataDir'] + '/word_embeddings.pkl', 'rb') as f:
        word_embeddings = pickle.load(f)

    model = load_best_model(best_model_path, word_embeddings)

    test_sentences = config['testComments']
    print(test_sentences)






if __name__ == "__main__":
    main()

