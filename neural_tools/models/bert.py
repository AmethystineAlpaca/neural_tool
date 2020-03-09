"""
a simple bert based on huggingface and implemented on my own. 
this classifier has been modified in a rush from my NER extractor, sorry if some code looks bad :(
"""
import transformers
import torch
from transformers import BertForSequenceClassification, AdamW, BertTokenizer, get_linear_schedule_with_warmup
from tqdm.notebook import trange as tnrange


class BertClassifier:
    def __init__(self, args):
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case=True)
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=len(args.tags))
        self.args = args

        if self.args.device == 'gpu':
            self.model.to('cuda')

    def preprocess_texts(self, texts):
        ids_list = []
        for text in texts:
            ids_list.append(self.preprocess_single_text(
                text, return_tensor=False))
        return ids_list

    def preprocess_single_text(self, text, return_tensor=True):
        ids = self.tokenizer.encode(text, add_special_tokens=True)[
            :self.args.max_seq]

        # padding
        ids = self.padding(ids, self.args.max_seq)

        if return_tensor:
            return torch.tensor([ids])
        else:
            return ids

    def padding(self, l, max_len, padding_id=0):
        l = l[:max_len]+[0]*max([max_len-len(l), 0])
        return l

    def preprocess_training_data(self, texts, labels):
        if len(texts) != len(labels):
            raise Exception('training data size not agree.')

        res_texts = []
        res_labels = []

        for i, _ in enumerate(texts):
            test, label = self.preprocess_single_training_data(
                texts[i], labels[i])
            res_texts.append(test)
            res_labels.append(label)

        return torch.tensor(res_texts), torch.tensor(res_labels)

    def preprocess_single_training_data(self, text, label):
        text = self.tokenizer.encode(text, add_special_tokens=True)
        # text = self.tokenizer.convert_tokens_to_ids(text)
        return self.padding(text, self.args.max_seq), label

    def train(self, data, lables):

        optimizer = AdamW(self.model.parameters(), correct_bias=False, lr=1e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.args.num_training_steps)

        self.model.zero_grad()

        epochs = tnrange(self.args.epoch)
        for current_epoch in epochs:
            iterations = tnrange(len(lables)//self.args.batch_size)
            batch = self.make_bach(data, lables, self.args.batch_size)
            for _ in iterations:
                batch_data, batch_lables = next(batch)
                self.model.train()

                batch_data, batch_lables = self.preprocess_training_data(
                    batch_data, batch_lables)

                if self.args.device == 'gpu':
                    batch_data = batch_data.to('cuda')
                    batch_lables = batch_lables.to('cuda')
                loss, res = self.model(batch_data, labels=batch_lables)[:2]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                self.model.zero_grad()

    def make_bach(self, data, lables, batch_size):
        return (
            [
                data[i*batch_size:(i+1)*batch_size], lables[i *
                                                            batch_size:(i+1)*batch_size]
            ]
            for i in range(len(data)//batch_size)
        )

    def evaluate(self, test_tensor, labels):
        predictions = self.predict(test_tensor)
        return self.evaluate_with_metrics(predictions, labels)

    def predict(self, test):
        test_ori = self.preprocess_texts(test)
        test_tensor = torch.tensor(test_ori)

        if self.args.device == 'gpu':
            test_tensor = test_tensor.to('cuda')

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(test_tensor)
            predictions = outputs[0]
        prediction = torch.argmax(predictions, 1)

        return prediction

    def evaluate_with_metrics(self, predictions, labels):
        return None


class TestArgs:
    def __init__(self):
        self.tags = {0, 1}

        self.epoch = 4
        self.batch_size = 16
        self.max_seq = 256

        # warm_up
        # 7613 5709
        self.warmup_steps = 5709*4//16//10
        self.num_training_steps = 5709*4//16

        # gradient clip
        self.max_grad_norm = 1

        self.device = 'gpu'


if __name__ == "__main__":
    extractor = BertClassifier(TestArgs())
    print(extractor.predict(['This is a pen for you! I am becoming a god.','I am a sample.']))
    extractor.train(
        ('This is a pen.','I love playing games.','this is good')*100,
        ([1],[0],[1])*100,
        )
    print(extractor.predict(('This is a pen.','I love playing games.','this is good')*10))
