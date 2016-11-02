import plac
import collections
import random

import pathlib
import cytoolz
import numpy
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from keras.layers import TimeDistributed
from keras.optimizers import Adam
import cPickle as pickle

import spacy


class SentimentAnalyser(object):
    @classmethod
    def load(cls, path, nlp, max_length=100):
        with (path / 'config.json').open() as file_:
            model = model_from_json(file_.read())
        with (path / 'model').open('rb') as file_:
            lstm_weights = pickle.load(file_)
        embeddings = get_embeddings(nlp.vocab)
        model.set_weights([embeddings] + lstm_weights)
        return cls(model, max_length=max_length)

    def __init__(self, model, max_length=100):
        self._model = model
        self.max_length = max_length

    def __call__(self, doc):
        X = get_features([doc], self.max_length)
        y = self._model.predict(X)
        self.set_sentiment(doc, y)

    def pipe(self, docs, batch_size=1000, n_threads=2):
        for minibatch in cytoolz.partition_all(batch_size, docs):
            minibatch = list(minibatch)
            sentences = []
            for doc in minibatch:
                sentences.extend(doc.sents)
            Xs = get_features(sentences, self.max_length)
            ys = self._model.predict(Xs)
            for sent, label in zip(sentences, ys):
                sent.doc.sentiment += label - 0.5
            for doc in minibatch:
                yield doc

    def set_sentiment(self, doc, y):
        doc.sentiment = float(y[0])
        print doc.sentiment
        # Sentiment has a native slot for a single float.
        # For arbitrary data storage, there's:
        # doc.user_data['my_data'] = y


def get_labelled_sentences(docs, doc_labels):
    labels = []
    sentences = []
    for doc, y in zip(docs, doc_labels):
        for sent in doc.sents:
            sentences.append(sent)
            labels.append(y)
    return sentences, numpy.asarray(labels, dtype='int32')


def get_features(docs, max_length):
    docs = list(docs)
    Xs = numpy.zeros((len(docs), max_length), dtype='int32')
    for i, doc in enumerate(docs):
        j = 0
        for token in doc:
            if token.has_vector and not token.is_punct and not token.is_space:
                Xs[i, j] = token.rank + 1
                j += 1
                if j >= max_length:
                    break
    return Xs


def train(train_texts, train_labels, dev_texts, dev_labels,
        lstm_shape, lstm_settings, lstm_optimizer, batch_size=100, nb_epoch=5,
        by_sentence=False, model=None):
    print("Loading spaCy")
    nlp = spacy.load('en', entity=False)
    embeddings = get_embeddings(nlp.vocab)
    if not model:
        print('Creating a New Model')
        model = compile_lstm(embeddings, lstm_shape, lstm_settings)
    else:
        print('Using an Existing Model')
        model.compile(optimizer=Adam(lr=lstm_settings['lr']), loss='binary_crossentropy',
          metrics=['accuracy'])
    print("Parsing texts...")
    print('got -------------- training docs----- {}'.format(len(train_texts)))
    print('got --------------- dev docs ------------------ {}'.format(len(dev_texts)))
    train_docs = list(nlp.pipe(train_texts, batch_size=5000, n_threads=3))
    dev_docs = list(nlp.pipe(dev_texts, batch_size=5000, n_threads=3))
    if by_sentence:
        train_docs, train_labels = get_labelled_sentences(train_docs, train_labels)
        dev_docs, dev_labels = get_labelled_sentences(dev_docs, dev_labels)

    train_X = get_features(train_docs, lstm_shape['max_length'])
    dev_X = get_features(dev_docs, lstm_shape['max_length'])
    model.fit(train_X, train_labels, validation_data=(dev_X, dev_labels),
              nb_epoch=nb_epoch, batch_size=batch_size)
    return model


def compile_lstm(embeddings, shape, settings):
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=True
        )
    )
    model.add(TimeDistributed(Dense(shape['nr_hidden'] * 2, bias=False)))
    model.add(Dropout(settings['dropout']))
    model.add(Bidirectional(LSTM(shape['nr_hidden'])))
    model.add(Dropout(settings['dropout']))
    model.add(Dense(shape['nr_class'], activation='sigmoid'))
    model.compile(optimizer=Adam(lr=settings['lr']), loss='binary_crossentropy',
		  metrics=['accuracy'])
    return model


def get_embeddings(vocab):
    max_rank = max(lex.rank+1 for lex in vocab if lex.has_vector)
    vectors = numpy.ndarray((max_rank+1, vocab.vectors_length), dtype='float32')
    for lex in vocab:
        if lex.has_vector:
            vectors[lex.rank + 1] = lex.vector
    return vectors


def evaluate(model_dir, texts, labels, max_length=100):
    def create_pipeline(nlp):
        '''
        This could be a lambda, but named functions are easier to read in Python.
        '''
        return [nlp.tagger, nlp.parser, SentimentAnalyser.load(model_dir, nlp,
                                                               max_length=max_length)]

    nlp = spacy.load('en')
    nlp.pipeline = create_pipeline(nlp)

    correct = 0
    i = 0
    for doc in nlp.pipe(texts, batch_size=1000, n_threads=4):
        print bool(labels[i])
        correct += bool(doc.sentiment >= 0) == bool(labels[i])


        # More Verbose output
        if bool(doc.sentiment >= 0) == bool(labels[i]):
            # Got it right
            print 'Correct: sentiment {} ({}) of document {}'.format(bool(labels[i]), doc.sentiment, doc.text)
        else:
            # Got it wrong
            print 'Wrong: got sentiment {} ({}) of document {}'.format(bool(doc.sentiment >= 0.5), doc.sentiment, doc.text)

        i += 1

    return float(correct) / i


def read_data(data_dir, limit=0):
    examples = []
    for subdir, label in (('pos', 1), ('neg', 0)):
        for filename in (data_dir / subdir).iterdir():
            with filename.open() as file_:
                text = file_.read()
            examples.append((text, label))
    random.shuffle(examples)
    if limit >= 1:
        examples = examples[:limit]
    return zip(*examples) # Unzips into two lists



def evaluate_test_content(model_dir, dev_dir, max_length):

    model_dir = pathlib.Path(model_dir)
    dev_dir = pathlib.Path(dev_dir)

    dev_texts, dev_labels = read_data(dev_dir)
    acc = evaluate(model_dir, dev_texts, dev_labels, max_length=max_length)
    print(acc)


def create_new_training(model_dir, dev_dir, train_dir,
        nr_hidden=64, max_length=100,
        dropout=0.5, learn_rate=0.001,
        nb_epoch=500, batch_size=1000, nr_examples=1):

    model_dir = pathlib.Path(model_dir)
    train_dir = pathlib.Path(train_dir)
    dev_dir = pathlib.Path(dev_dir)

    print("Read data")
    train_texts, train_labels = read_data(train_dir, limit=nr_examples)
    dev_texts, dev_labels = read_data(dev_dir, limit=nr_examples)
    train_labels = numpy.asarray(train_labels, dtype='int32')
    dev_labels = numpy.asarray(dev_labels, dtype='int32')

    lstm = train(train_texts, train_labels, dev_texts, dev_labels,
                 {'nr_hidden': nr_hidden, 'max_length': max_length, 'nr_class': 1},
                 {'dropout': 0.5, 'lr': learn_rate},
                 {},
                 nb_epoch=nb_epoch, batch_size=batch_size, model=None)
    weights = lstm.get_weights()
    with (model_dir / 'model').open('wb') as file_:
        pickle.dump(weights[1:], file_)
    with (model_dir / 'config.json').open('wb') as file_:
        file_.write(lstm.to_json())


def continue_training(model_dir, dev_dir, train_dir,
        nr_hidden=64, max_length=100,
        dropout=0.5, learn_rate=0.001,
        nb_epoch=500, batch_size=1000, nr_examples=1):

    print('Using Config: Model - {}, Dev - {}, Train - {}'.format(model_dir, dev_dir, train_dir))

    model_dir = pathlib.Path(model_dir)
    train_dir = pathlib.Path(train_dir)
    dev_dir = pathlib.Path(dev_dir)

    print("Read data")
    train_texts, train_labels = read_data(train_dir, limit=nr_examples)
    dev_texts, dev_labels = read_data(dev_dir, limit=nr_examples)
    train_labels = numpy.asarray(train_labels, dtype='int32')
    dev_labels = numpy.asarray(dev_labels, dtype='int32')

    with (model_dir / 'config.json').open() as file_:
        lstm_model = model_from_json(file_.read())
    with (model_dir / 'model').open('rb') as file_:
            lstm_weights = pickle.load(file_)

    nlp = spacy.load('en')
    embeddings = get_embeddings(nlp.vocab)
    lstm_model.set_weights([embeddings] + lstm_weights)

    lstm_model = train(train_texts, train_labels, dev_texts, dev_labels,
                 {'nr_hidden': nr_hidden, 'max_length': max_length, 'nr_class': 1},
                 {'dropout': 0.5, 'lr': learn_rate},
                 {},
                 nb_epoch=nb_epoch, batch_size=batch_size, model=lstm_model)

    weights = lstm_model.get_weights()
    with (model_dir / 'model').open('wb') as file_:
        pickle.dump(weights[1:], file_)
    with (model_dir / 'config.json').open('wb') as file_:
        file_.write(lstm_model.to_json())


