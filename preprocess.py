import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import os
import itertools
from utils.print_decorator import print_if_complete
from gensim.models import Word2Vec, FastText
from collections import Counter
import config


@print_if_complete
def preprocess(train):
    lbl_enc = sklearn.preprocessing.LabelEncoder()
    y = lbl_enc.fit_transform(train.label.values)

    xtrain1, xvalid1, xtrain2, xvalid2, ytrain, yvalid = sklearn.model_selection.train_test_split(train.sentence1.values, train.sentence2.values, y,
                                                                                                  stratify=y,
                                                                                                  random_state=42,
                                                                                                  test_size=0.1, shuffle=True)

    return xtrain1, xvalid1, xtrain2, xvalid2, ytrain, yvalid


@print_if_complete
def word_embedding(train, test):
    ''' make word2vec embedding
    '''

    if False and os.path.isfile('embedding.model'):
        embedding = Word2Vec.load('embedding.model')
    else:
        sent1 = [row.split() for row in train.sentence1.values]
        sent2 = [row.split() for row in train.sentence2.values]
        sent3 = [row.split() for row in test.sentence1.values]
        sent4 = [row.split() for row in test.sentence2.values]

        sentences = sent1 + sent2
        freq = Counter(list(itertools.chain(*sentences)))

        print("Creating embedding/frequency")

        if config.opt_embedding == "w2v":
            embedding = Word2Vec(min_count=1,
                                 window=3,
                                 size=300,
                                 sample=6e-5,
                                 alpha=0.03,
                                 min_alpha=0.0007,
                                 negative=20)
        elif config.opt_embedding == "fast":
            embedding = FastText(size=4, window=3, min_count=1)  # instantiate

        embedding.build_vocab(sentences)
        embedding.train(
            sentences, total_examples=embedding.corpus_count, epochs=30)  # train
        embedding.init_sims(replace=True)
        embedding.save('embedding.model')

        return embedding
