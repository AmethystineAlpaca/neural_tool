import torch
from torch.optim import Adam
from torch import LongTensor
from neural_tools.models.cnn import CNNModel
from neural_tools.models.lstm import LSTMModel
import re
import random
import joblib
from collections import Counter
from sklearn.metrics import accuracy_score


def split(x, batch):
    for i in range(0, len(x), batch):
        yield(x[i:i+batch])


class MyTextModel:
    vocab2id = {}
    id2vocab = {}
    model = None
    max_seq_len = 0
    tokenizer = None

    def __init__(self, model_tpye='cnn', tokenizer=None):
        if model_tpye == 'cnn':
            self.model_cls = CNNModel

        elif model_tpye == 'lstm':
            self.model_cls = LSTMModel
        else:
            raise Exception('wrong model type')
        self.tokenizer = tokenizer

    def preprocess_text(self, data, update_dict=False):
        if not self.tokenizer:
            data = [re.sub('_', ' ', x) for x in data]
            data = [re.findall(r"(?:\w+(?:'\w+)?|[^\w\s])|(?:#)", x)
                    for x in data]
        else:
            data = [self.tokenizer.tokenize(x) for x in data]
        lens = [len(x) for x in data]

        data = [x[:self.max_seq_len]+['[PAD]'] *
                max([0, self.max_seq_len-len(x)]) for x in data]

        if update_dict:
            vocab = [x for sent in data for x in sent]
            vocab_conter = Counter(vocab)
            vocab = [(x, vocab_conter[x])for x in vocab_conter]
            vocab = sorted(vocab, key=lambda x: x[0], )
            vocab = set([x[0] for x in vocab][:len(vocab)*8//10])
            # print(list(vocab)[:10], list(vocab)[-10:])
            

            vocab.add('[PAD]')

            self.vocab2id = {x: i+1 for i, x in enumerate(vocab)}
            self.id2vocab = {i+1: x for i, x in enumerate(vocab)}
        data = LongTensor([[self.vocab2id.get(x, 0)
                            for x in sent] for sent in data])
        return data, lens

    def train(self, data, label, max_seq_len=100, class_count=2, epochs=30, batch_size=16, lr=0.001, early_stop=0,test=None):
        self.max_seq_len = max_seq_len
        data, lens = self.preprocess_text(data, update_dict=True)

        label = torch.LongTensor(label)
        # batch
        data = list(split(data, batch_size))
        label = list(split(label, batch_size))
        lens = list(split(lens, batch_size))
        self.model = self.model_cls(
            max_seq_len, len(self.id2vocab)+1, class_count)

        optimizer = Adam(self.model.parameters(), lr=lr)
        max_accuracy = 0
        stop_step = early_stop
        for i in range(epochs):
            self.model.train()
            step = 0
            for x, y, original_lengths in zip(data, label, lens):
                step +=1
                optimizer.zero_grad()
                _, loss = self.model(x, original_lengths, labels=y)
                loss = loss.sum()
                if step%50==0:
                    print(i, loss)
                loss.backward()
                optimizer.step()
            if early_stop and test:
                predictions = self.predict(test[0])
                accuracy = accuracy_score(test[1], predictions)
                print(accuracy)
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                else:
                    stop_step-=1
                    if stop_step <=0:
                        print('stop at epoch',i, 'accuracy:', accuracy)
                        break
                

    def predict(self, data, batch_size=64):
        self.model.eval()
        data, lens = self.preprocess_text(data)
        data = split(data, batch_size)
        lens = split(lens, batch_size)
        res = []
        for x, ori_len in zip(data, lens):
            res += torch.argmax(self.model(x, ori_len), 1).tolist()
        return res

    def save(self, path):
        joblib.dump(self.vocab2id, path+'/vocab2id')
        joblib.dump(self.id2vocab, path+'/id2vocab')
        joblib.dump(self.model, path+'/model')
        joblib.dump(self.max_seq_len, path+'/max_seq_len')

    def load(self, path):
        self.vocab2id = joblib.load(path+'/vocab2id')
        self.id2vocab = joblib.load(path+'/id2vocab')
        self.model = joblib.load(path+'/model')
        self.max_seq_len = joblib.load(path+'/max_seq_len')


if __name__ == "__main__":
    model = MyTextModel(model_tpye='lstm')
    # model = MyTextModel()
    data = ['this is good', 'this is bad, very.',
            'this is not OK', 'this is very good'
            ]*10
    label = [1, 0, 0, 1]*10

    model.train(data, label)
    res = model.predict(['this is good', 'this is bad',
                         'this is not OK', 'this is very good', 'this is OK'
                         ])
    print(res)
