import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Simple_CNN(nn.Module):
    """ Simple 1D CNN Classifier. """
    def __init__(self,
                 pretrained_embedding=None,
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

        in_embeddings = self.embedding(inputs).float()  # [bs, max_sent_len, embed_dim]
        in_embeddings_shaped = in_embeddings.permute(0, 2, 1)  # [bs, embed_dim, max_sent_len]

        # convolve
        in_convolved_list = [F.relu(conv1d(in_embeddings_shaped)) for conv1d in self.conv1d_list]

        # pool
        in_pool_list = [F.max_pool1d(in_conv, kernel_size=in_conv.shape[2])
                             for in_conv in in_convolved_list]

        # create fully connected layer
        in_fully_connected = torch.cat([in_pool.squeeze(dim=2) for in_pool in in_pool_list], dim=1)

        # logit computation
        logits = self.fc(self.dropout(in_fully_connected))

        return logits


