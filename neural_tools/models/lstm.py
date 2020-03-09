import torch
from torch.nn import Module, LSTM, MaxPool1d, Linear, CrossEntropyLoss, Embedding, Dropout
from torch.optim import SGD
from torch import LongTensor
from collections import Counter
from torch.nn.utils.rnn import pack_padded_sequence

class LSTMModel(Module):

    def __init__(self, input_len, vocab_len, out_length, embedding_dim=100, hidden_dim=100, loss=None):
        super().__init__()
        if not loss:
            self.loss = CrossEntropyLoss(reduction='none')
        else:
            self.loss = loss
        self.embedding = Embedding(vocab_len, embedding_dim, padding_idx=None,
                                   max_norm=None, norm_type=2., scale_grad_by_freq=False,
                                   sparse=False, _weight=None)
        self.hidden_dim = hidden_dim
        self.lstm_layer = LSTM(embedding_dim, hidden_dim, 2, bidirectional=True, dropout=0.5)
        self.full_connect = Linear(hidden_dim, out_length)

    def forward(self, data, lens, labels=[]):
        data = self.embedding(data)
        data = data.permute([1, 0, 2])
        data = pack_padded_sequence(data, lens, enforce_sorted=False)
        _, (ht, _) = self.lstm_layer(data)
        data = ht[-1]
        data = self.full_connect(data)
        # data = torch.softmax(data,1)

        if len(labels) != 0:
            return data, self.loss(data, labels)
        else:
            return data


def split(x, batch):
    for i in range(0, len(x), batch):
        yield(x[i:i+batch])
