import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


