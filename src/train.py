import os
import argparse
import pickle
import fasttext.util
from datasets import load_from_disk
from train_data import *
from train_model import *
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


""" Main driver function for training. Will go through the steps in sequence.

1. Data retrieval - (grabbing OxAISH-AL-LLM/wiki_toxic) dataset using datasets.
2. Data pre-processing - clean + format the data into FastText appropriate format and save to file.
3. Model training - train a binary 'toxicity' classifier using FastText, and tune the threshold.

Can potentially use model on Twitch comments etc. to ascertain toxicity? An additional project.

"""

def train(model, optimizer, train_dataloader, val_dataloader, epochs=20):

    best_accuracy = 0

    # start training loop
    for epoch_i in range(epochs):
        total_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):

            b_input_ids, b_labels = tuple(t.to(device) for t in batch)

def main():

    if torch.cuda.is_available():

        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # check if data stuff has already been done
    dataset_path = os.getcwd() + '/data/processed_dataset.pkl'
    word_emb_path = os.getcwd() + '/data/word_embeddings.pkl'

    if os.path.isfile(dataset_path) and os.path.isfile(word_emb_path):
        print('Dataset processing already done and saved - skipping dataset creation')

        # load dataset
        json_data_files = {
            "train": "data/processed_dataset-train.json",
            "validation": "data/processed_dataset-validation.json",
            "test": "data/processed_dataset-test.json",
        }
        processed_dataset = load_dataset("json", data_files=json_data_files)

        with open(word_emb_path, 'rb') as f:
            word_embeddings = pickle.load(f)
        pass
    else:
        print('As no dataset files found, creating...')

        # data retrieval and prep
        print('retrieving dataset...')
        full_dataset = retrieve_dataset()

        # data cleaning
        print('cleaning and tokenizing dataset...')
        processed_dataset, vocab_dict, max_sent_len = preprocess_dataset(full_dataset)

        """ Need to follow procedure in building vocab and getting max sent length
        for CNN model to work properly - can follow steps in tutorial
        https://chriskhanhtran.github.io/posts/cnn-sentence-classification/ """

        # download FT word vectors
        vec_name = "wiki-news-300d-1M.vec.zip"
        vector_path = retrieve_word_vecs(vec_name)

        # loading the FastText word vectors as embeddings
        print('loading FastText word vectors...')
        word_embeddings = torch.tensor(load_word_vectors(vocab_dict, vector_path))
        print(word_embeddings.size())

        # save the dataset stuff to speed up for testing

        # save dataset as json
        json_data_files = {
            "train": "data/processed_dataset-train.json",
            "validation": "data/processed_dataset-validation.json",
            "test": "data/processed_dataset-test.json",
        }

        for split, dataset in processed_dataset.items():
            dataset.to_json(f"processed_dataset-{split}.json")

        # save word embedding as pickle
        with open(word_emb_path, 'wb') as f:
            pickle.dump(word_embeddings, f)


    processed_train = processed_dataset['train']
    processed_val = processed_dataset['validation']
    processed_test = processed_dataset['test']

    # pass into Pytorch datasets
    print('creating Torch Datasets...')
    trainingDataset = ToxicDataset(processed_train)
    print('How many comments in training: ', trainingDataset.__len__())
    valDataset = ToxicDataset(processed_val)
    print('How many comments in validation: ', valDataset.__len__())
    testDataset = ToxicDataset(processed_test)
    print('How many comments in testing: ', testDataset.__len__())

    # set up Dataloaders
    print('setting up Torch Dataloaders...')
    batch_size = 30

    train_sampler = RandomSampler(trainingDataset)
    train_dataloader = DataLoader(trainingDataset, sampler=train_sampler, batch_size=batch_size)

    val_sampler = SequentialSampler(valDataset)
    val_dataloader = DataLoader(valDataset, sampler=val_sampler, batch_size=batch_size)

    test_sampler = SequentialSampler(testDataset)
    test_dataloader = DataLoader(testDataset, sampler=test_sampler, batch_size=batch_size)

    # now we move to training the actual model
    print('Model training begins...')
    model = Simple_CNN(pretrained_embedding=word_embeddings,
                       vocab_size=len(vocab_dict),
                       embed_dim=300)

    train(model, 'optimize', train_dataloader, val_dataloader, epochs=10)
    # for step, batch in enumerate(train_dataloader):
    #     batch_inputs, batch_labels = tuple(t for t in batch)
    #     print('batch inputs type: ', type(batch_inputs))
    #     logits = model(batch_inputs)
        # exit()

if __name__ == "__main__":
    main()
