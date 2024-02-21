import logging
import pickle
import warnings

from datasets import load_from_disk

from train_functions import check_for_cuda, set_up_dataloaders, init_model, train
from utils import config_parser, configure_logging, load_json


def main():

    args = config_parser()
    config = load_json(args.config_path)
    configure_logging('train')

    global device
    device = check_for_cuda()

    logging.info('Loading data files...')

    warnings.filterwarnings("ignore", category=FutureWarning, module="datasets")
    processed_dataset = load_from_disk(config['dataDir'] + '/processed_dataset')

    with open(config['dataDir'] + '/word_embeddings.pkl', 'rb') as f:
        word_embeddings = pickle.load(f)
        vocab_size, embed_dim = word_embeddings.shape

    batch_size = config['batchSize']

    logging.info(f'creating Torch Datasets with batch size {batch_size}...')

    train_dataloader, val_dataloader, test_dataloader = set_up_dataloaders(processed_dataset, batch_size)

    logging.info('Model training beginning...')

    model, optimizer = init_model(word_embeddings=word_embeddings,
                                  filter_sizes=config['cnnFilterSizes'],
                                  num_filters=config['cnnNumFilters'],
                                  num_classes=config['cnnNumClasses'],
                                  dropout=config['cnnDropout'],
                                  lr=config['cnnLr'])


    trained_model, results_df = train(model, optimizer, train_dataloader, val_dataloader, test_dataloader, config['modelDir'], epochs=config['trainEpochs'])

    results_df.to_csv(config['modelDir']+'/train_results.tsv', sep='\t', index=False)

if __name__ == "__main__":
    main()
