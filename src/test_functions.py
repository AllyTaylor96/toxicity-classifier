import torch
import torch.nn.functional as F

from train_model import Simple_CNN
from data_functions import clean_tokenize_text, encode_text

def load_best_model(model_path, word_embeddings):
    model = Simple_CNN(pretrained_embedding=word_embeddings)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model

def predict(text, model, vocab_dict, max_sent_len=250):

    # put into correct format for encoding and tokenizing to work
    comment = {'comment_text': text}

    # tokenize, pad and encode the text
    tokenized_comment = clean_tokenize_text(comment)
    encoded_comment = encode_text(tokenized_comment, vocab_dict, max_sent_len)

    input_ids = encoded_comment['encoded_text']

    # convert to tensors
    input_tensor = torch.tensor(input_ids).unsqueeze(dim=0)

    # compute logits
    logits = model.forward(input_tensor)

    # compute probability
    probs = F.softmax(logits, dim=1).squeeze(dim=0)
    return probs

