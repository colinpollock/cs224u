"""Using Torch for SST.

Some of this is redundant with the code Chris wrote, but I wanted to go through it myself.
"""

import os
import itertools
from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import DataLoader, TensorDataset
from torchtext import vocab
from torchtext.data import get_tokenizer

from sklearn.metrics import classification_report, f1_score



import numpy as np

import sst

SST_HOME = os.path.join('data', 'trees')
UNK = '$UNK'


def load_raw_data(train_or_dev):    
    if train_or_dev == 'train':
        func = sst.train_reader
    elif train_or_dev == 'dev':
        func = sst.dev_reader
    else:
        assert False
        
    texts = []
    labels = []
    for tree, label in func(SST_HOME, class_func=sst.ternary_class_func):
        texts.append(text_from_tree(tree))
        labels.append(label)
    return texts, labels




class RnnClassifier(nn.Module):
    GLOVE_DIM = 100

    def __init__(
        self,
        hidden_dimension,
        num_classes,
        rnn_type,
        batch_size=128,
        epochs=1,
        print_every=1,
        bidirectional=False,
        update_glove=False,
    ):
        super(RnnClassifier, self).__init__()

        self.tokenize = get_tokenizer('basic_english')

        self.batch_size = batch_size
        self.epochs = epochs
        self.print_every = print_every

        self.glove = vocab.GloVe(name='6B', dim=self.GLOVE_DIM)
        vocab_size, embedding_dimension = self.glove.vectors.shape
        self.embedding = nn.Embedding.from_pretrained(self.glove.vectors)
        self.embedding.weight.requires_grad = update_glove

        rnn_name_to_type = {
            'rnn': nn.RNN,
            'lstm': nn.LSTM,
            'gru': nn.GRU,
        }

        rnn = rnn_name_to_type[rnn_type]
        self.rnn = rnn(
            input_size=embedding_dimension,
            hidden_size=hidden_dimension,
            batch_first=True,
            bidirectional=bidirectional
        )

        classifier_dimension = hidden_dimension * 2 if bidirectional else hidden_dimension
        self.linear = nn.Linear(classifier_dimension, num_classes)
        
    def forward(self, inputs):
        # dense layer off of the final one
        embedded = self.embedding(inputs)
        output, hidden_states = self.rnn(embedded)
        
        # Output is of shape (batch, num_layers, hidden_dimension). num_layers is only
        # 1 so we can select it. We want the final output for all batches.
        # Index 1 of `0` is 
        final_output = output[:, 0, :]
        densed = self.linear(final_output)
        return densed
        

    def predict(self, text):
        vector = self.featurize(text).unsqueeze(0)
        prediction = self(vector).argmax()

        for label, id_ in self.label_to_id.items():
            if prediction == id_:
                return label

        assert False, 'wtf'


    def fit(self, train_texts, train_labels):
        self.label_to_id = self.fit_labels(train_labels)

        optimizer = Adam(self.parameters())
        criterion = nn.CrossEntropyLoss()

        data_loader = self.build_loader(train_texts, train_labels)

        for epoch in range(1, self.epochs+1):
            epoch_loss = 0

            total = correct = 0
            for inputs, labels in data_loader:
                outputs = self(inputs)

                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                predictions = outputs.argmax(axis=1)
                correct += (predictions == labels).sum().item()
                total += len(labels)

            if epoch % self.print_every == 0:
                accuracy = correct / total
                print(f'Epoch {epoch}: {epoch_loss:.2f}\t{accuracy:.2f}')


    def build_loader(self, texts, labels):
        token_ids_for_each_text = [self.featurize(text) for text in texts]
        padded = pad_sequence(token_ids_for_each_text, batch_first=True)
        inputs_tensor = padded

        labels_tensor = torch.LongTensor([self.label_to_id[label] for label in labels])
        dataset = TensorDataset(inputs_tensor, labels_tensor)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True
        )


    @staticmethod
    def fit_labels(labels):
        label_to_id = {}
        for label in labels:
            if label not in label_to_id:
                label_to_id[label] = len(label_to_id)

        return label_to_id


    def featurize(self, text):
        tokens = self.tokenize(text)
        lookup = self.glove.stoi

        # I'm ignoring tokens that aren't in the vocabulary
        # TODO: handle case where all tokens are out of vocab
        return torch.LongTensor([lookup[token] for token in tokens if token in lookup])








class GloveClassifier(nn.Module):
    """Shallow NN using the mean of all tokens' GloVe vectors.

    Given a text:
    * tokenize it into a list of tokens
    * convert that into a list of token IDs
    * pull the vectors for each of those tokens from the embedding
    * take the mean of these vectors, resulting in a 100 dimensional vector
    * pass this vector to a linear layer
    """
    GLOVE_DIM = 100

    def __init__(self, hidden_dim=10, num_classes=3, epochs=10, batch_size=128, print_every=1, update_glove=True):
        super(GloveClassifier, self).__init__()

        self.tokenize = get_tokenizer('basic_english')

        # These are better as `fit` parameters, but this makes running experiments
        # easier.
        self.epochs = epochs
        self.batch_size = batch_size
        self.print_every = print_every

        self.glove = vocab.GloVe(name='6B', dim=self.GLOVE_DIM)

        vocab_size, embed_dim = self.glove.vectors.shape
        self.embed = nn.Embedding.from_pretrained(self.glove.vectors)
        self.embed.weight.requires_grad = update_glove

        self.linear1 = nn.Linear(self.GLOVE_DIM, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs):
        """Inputs is a [batch, num_words, embed_dim] tensor."""
        # Note: I could just use EmbeddingBag, which would take care of
        # the mean.
        out1 = self.embed(inputs).mean(dim=1)
        out2 = F.relu(self.linear1(out1))
        out3 = self.linear2(out2)
        return out3


    def predict(self, text):
        vector = self.featurize(text).unsqueeze(0)
        prediction = self(vector).argmax()

        for label, id_ in self.label_to_id.items():
            if prediction == id_:
                return label

        assert False, 'wtf'


    def fit(self, train_texts, train_labels):
        self.label_to_id = self.fit_labels(train_labels)

        optimizer = Adam(self.parameters())
        criterion = nn.CrossEntropyLoss()

        data_loader = self.build_loader(train_texts, train_labels)

        for epoch in range(1, self.epochs+1):
            epoch_loss = 0

            total = correct = 0
            for inputs, labels in data_loader:
                outputs = self(inputs)

                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                predictions = outputs.argmax(axis=1)
                correct += (predictions == labels).sum().item()
                total += len(labels)

            if epoch % self.print_every == 0:
                accuracy = correct / total
                print(f'Epoch {epoch}: {epoch_loss:.2f}\t{accuracy:.2f}')


    def build_loader(self, texts, labels):
        token_ids_for_each_text = [self.featurize(text) for text in texts]
        padded = pad_sequence(token_ids_for_each_text, batch_first=True)
        inputs_tensor = padded

        labels_tensor = torch.LongTensor([self.label_to_id[label] for label in labels])
        dataset = TensorDataset(inputs_tensor, labels_tensor)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True
        )


    @staticmethod
    def fit_labels(labels):
        label_to_id = {}
        for label in labels:
            if label not in label_to_id:
                label_to_id[label] = len(label_to_id)

        return label_to_id


    def featurize(self, text):
        tokens = self.tokenize(text)
        lookup = self.glove.stoi

        # I'm ignoring tokens that aren't in the vocabulary
        # TODO: handle case where all tokens are out of vocab
        return torch.LongTensor([lookup[token] for token in tokens if token in lookup])


class BOWClassifier(nn.Module):
    def __init__(self, word_to_id, label_to_id, batch_size=32, l2_strength=0, epochs=1):
        super(BOWClassifier, self).__init__()

        self.tokenize = get_tokenizer('basic_english')

        self.batch_size = batch_size
        self.word_to_id = word_to_id
        self.label_to_id = label_to_id

        self.vocab_size = len(word_to_id)
        self.num_classes = len(label_to_id)
        self.l2_strength = l2_strength
        self.epochs = epochs

        self.linear = nn.Linear(self.vocab_size, self.num_classes)

    def forward(self, inputs):
        x = self.linear(inputs)
        return x


    def fit(self, train_texts, train_labels):
        optimizer = Adam(self.parameters(), weight_decay=self.l2_strength)
        criterion = nn.CrossEntropyLoss()

        data_loader = self._build_train_dataset(train_texts, train_labels)

        for epoch in range(1, self.epochs+1):
            epoch_loss = 0

            total = correct = 0
            for inputs, labels in data_loader:
                outputs = self(inputs)

                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                predictions = outputs.argmax(axis=1)
                correct += (predictions == labels).sum().item()
                total += len(labels)

            print_interval = 5
            if epoch % print_interval == 0:
                accuracy = correct / total
                print(f'Epoch {epoch}: {epoch_loss:.2f}\t{accuracy:.2f}')

    def evaluate(self, assess_texts, assess_labels):
        preds = np.array([self.predict(text) for text in assess_texts])
        actuals = np.array(assess_labels)
        return (preds == actuals).mean()

    def predict(self, text):
        vector = self.make_vector(text)
        prediction = self(vector).argmax()

        for label, id_ in self.label_to_id.items():
            if prediction == id_:
                return label

        assert False, 'wtf'


    def make_vector(self, text):
        unk_id = self.word_to_id[UNK]

        input_ = torch.zeros(self.vocab_size)
        for word in self.tokenize(text):
            word_id = self.word_to_id.get(word, unk_id)
            input_[word_id] += 1

        return input_

    def _build_train_dataset(self, texts, labels):
        inputs_tensor = torch.stack([self.make_vector(text) for text in texts])
        labels_tensor = torch.LongTensor([self.label_to_id[label] for label in labels])
        dataset = TensorDataset(inputs_tensor, labels_tensor)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True
        )


    @classmethod
    def get_words(cls, texts, min_count, max_freq, top_n):
        word_counts = Counter()
        doc_counts = Counter()
        for text in texts:
            seen_words = set()
            for word in cls.tokenize(text):
                word_counts[word] += 1
                if word not in seen_words:
                    doc_counts[word] += 1
                    seen_words.add(word)

        num_docs = len(texts)
        for word, count in doc_counts.items():
            doc_counts[word] = count / num_docs

        for word, count in list(word_counts.items()):
            if count < min_count or doc_counts[word] > max_freq:
                del word_counts[word]

        return [word for (word, count) in word_counts.most_common(top_n)]


    @classmethod
    def build_vocab(cls, texts, labels, min_count=2, max_freq=.4, top_n=30000):
        word_to_id = {}
        for word in cls.get_words(texts, min_count, max_freq, top_n):
            if word not in word_to_id:
                word_to_id[word] = len(word_to_id)

        word_to_id[UNK] = len(word_to_id)

        label_to_id = {}
        for label in labels:
            if label not in label_to_id:
                label_to_id[label] = len(label_to_id)

        return word_to_id, label_to_id


def text_from_tree(tree):
    text = ' '.join(tree.leaves())
    return text.replace(" 's", "'s").replace(" , ", ", ")


def experiment(model, train_texts, train_labels, dev_texts, dev_labels):
    model.fit(train_texts, train_labels)
    print()

    print('## Train ##')
    # This is super slow because I'm refeaturizing and not doing vectorized predictions.
    # TODO: clean it up by pulling out the building of the train and dev loaders and
    # using them throughout.
    train_predictions = np.array([model.predict(train_text) for train_text in train_texts])
    train_report = classification_report(train_labels, train_predictions)
    print(train_report)
    
    dev_predictions = np.array([model.predict(dev_text) for dev_text in dev_texts])
    dev_report = classification_report(dev_labels, dev_predictions)
    print('## Dev ##')
    print(dev_report)


if __name__ == '__main__':
    train_texts, train_labels = load_raw_data('train')
    dev_texts, dev_labels = load_raw_data('dev')

    glove_classifier = GloveClassifier(hidden_dim=50, epochs=10, print_every=1)
    experiment(glove_classifier, train_texts, train_labels, dev_texts, dev_labels)
