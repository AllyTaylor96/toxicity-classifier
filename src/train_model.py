import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# class CNN_Classifier(nn.Module):
#     """ A Simple CNN for Text Classification on FastText Embeddings."""
#     def __init__(self,
#                  nclasses: int = 2,
#                  window_size: int = 16,
#                  embedding_dim: int = 16,
#                  filter_multiplier: int = 64):

#         super(CNN_Classifier, self).__init__()

#         self.conv = nn.Conv2d(in_channels = 1,
#                               out_channels = filter_multiplier,
#                               stride = CONV_STRIDE,
#                               kernel_size = 3,
#                               padding = 1)

#         self.maxpool = nn.MaxPool2d(kernel_size = 3,
#                                     stride = CONV_STRIDE,
#                                     padding = 1)

#         self.linear = nn.Linear(LINEAR_DIM,
#                                out_features = nclasses)

#         self.softmax = nn.LogSoftmax(1)

#     def forward(self, x): # shape: (batch_size, nclasses=2, vector_dim=300)
#         x = torch.transpose(x, 1, 2) # (batch_size, 1, 300)
#         x = torch.unsqueeze(x, 1) # (batch_size, 1, 300, 1)
#         x = self.conv(x) # (batch_size, 64, 150, 1)
#         x = self.maxpool(x) # (batch_size, 64, 75, 1)
#         F.relu(x) # non linear function activates the neurons
#         x = x.flatten(start_dim=1) # (batch_size, 4800)

#         x = self.linear(x)
#         F.relu(x)
#         x = self.softmax(x)
#         return x

def train(model):
    logits = model(input_ids)


class Simple_CNN(nn.Module):
    """ Simple 1D CNN Classifier. """
    def __init__(self,
                 pretrained_embedding=None,
                 vocab_size=None,
                 embed_dim=300,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 num_classes=2,
                 dropout=0.5):

        super(Simple_CNN, self).__init__()
        self.vocab_size, self.embed_dim = pretrained_embedding.shape
        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                      freeze=True)

        # set up convolution network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])

        # connected layer and dropout
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs):
        """Forward pass through network.

        Args:
            inputs (torch.Tensor): A tensor of data with shape (batch_size, sent_emb_length)

        Returns:
            logits (torch.Tensor): A tensor of output logits with shape (batch_size, n_classes
        """

        print(type(inputs))
        input_tensor = torch.tensor(inputs)
        print(type(input_tensor))
        exit()
        x_embed = self.embedding(inputs).float()
        print(x_embed)
        exit()


