import torch

from train_model import Simple_CNN
from data_functions import clean_tokenize_text, encode_text

def load_best_model(model_path, word_embeddings):
    model = Simple_CNN(pretrained_embedding=word_embeddings)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model

def predict(text, model):

    # tokenize, pad and encode the text

    # convert to tensors

    # compute logits

    # compute probability

    # return results
    pass
