import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Classifier(nn.Module):
    """ A Simple CNN for Text Classification on FastText Embeddings."""
    def __init__(self,
                 nclasses: int = 2,
                 window_size: int = 16,
                 embedding_dim: int = 16,
                 filter_multiplier: int = 64):

        super(CNN_Classifier, self).__init__()

        self.conv = nn.Conv2d(in_channels = 1,
                              out_channels = filter_multiplier,
                              stride = CONV_STRIDE,
                              kernel_size = 3,
                              padding = 1)

        self.maxpool = nn.MaxPool2d(kernel_size = 3,
                                    stride = CONV_STRIDE,
                                    padding = 1)

        self.linear = nn.Linear(LINEAR_DIM,
                               out_features = nclasses)

        self.softmax = nn.LogSoftmax(1)

    def forward(self, x): # shape: (batch_size, nclasses=2, vector_dim=300)
        x = torch.transpose(x, 1, 2) # (batch_size, 1, 300)
        x = torch.unsqueeze(x, 1) # (batch_size, 1, 300, 1)
        x = self.conv(x) # (batch_size, 64, 150, 1)
        x = self.maxpool(x) # (batch_size, 64, 75, 1)
        F.relu(x) # non linear function activates the neurons
        x = x.flatten(start_dim=1) # (batch_size, 4800)

        x = self.linear(x)
        F.relu(x)
        x = self.softmax(x)
        return x


