import torch
from torch.nn import Module, Linear, ReLU, NLLLoss, Embedding, LogSoftmax


class GCNModel(Module):
    def __init__(self, input_len, out_length, loss=None):
        super().__init__()
        self.linear = Linear(input_len, input_len)
        self.activation = ReLU()

        self.linear_2 = Linear(input_len, out_length)

        if not loss:
            self.loss = NLLLoss(reduction='none')
        else:
            self.loss = loss

        self.softmax = LogSoftmax()

    def forward(self, data, adjacent_matrix, degree_matrix, y=[]):
        normalized_adjacent_matrix = torch.mm(
            torch.mm(degree_matrix.inverse()**(1/2), adjacent_matrix), degree_matrix.inverse()**(1/2))

        data = torch.mm(data, normalized_adjacent_matrix)
        data = self.activation(self.linear(data))

        data = torch.mm(data, normalized_adjacent_matrix)
        data = self.linear_2(data)
        data = self.softmax(data)
        if len(y) != 0:
            return data, self.loss(data, y)
        else:
            return data
