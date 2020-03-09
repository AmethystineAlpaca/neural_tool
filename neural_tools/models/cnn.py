import torch
from torch.nn import Module, Conv2d, MaxPool1d, Linear, CrossEntropyLoss, Embedding, Dropout, ModuleList
from torch.optim import SGD
from torch import LongTensor
from collections import Counter
import torch.nn.functional as F


class CNNModel(Module):

    def __init__(self, input_len, vocab_len, out_length, embedding_dim=100, hidden_dim=100, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', loss=None):
        super().__init__()
        in_channels = 1
        out_channels = 2
        if not loss:
            self.loss = CrossEntropyLoss(reduction='none')
        else:
            self.loss = loss
        self.embedding = Embedding(vocab_len, embedding_dim, padding_idx=None,
                                   max_norm=None, norm_type=2., scale_grad_by_freq=False,
                                   sparse=False, _weight=None)
                                   
        kernel_wins = [1,2,3,4,5]
        self.convs = ModuleList(
            [Conv2d(in_channels, out_channels, (w, embedding_dim)) for w in kernel_wins])

        self.dropout = Dropout(0.5)
        
        self.full_connect_1 = Linear(len(kernel_wins)*out_channels, out_length)

    def forward(self, data, lens,labels=[]):
        data = self.embedding(data)
        data = torch.unsqueeze(data, 1)

        data = [conv(data) for conv in self.convs]
        data = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in data]
        data = torch.cat(data, 1)

        data = data.squeeze(-1)
        data = self.dropout(data)
        data = self.full_connect_1(data)

        if len(labels) != 0:
            return data, self.loss(data, labels)
        else:
            return data


def split(x, batch):
    for i in range(0, len(x), batch):
        yield(x[i:i+batch])



    
