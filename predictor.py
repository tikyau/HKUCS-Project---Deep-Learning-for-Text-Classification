import multiprocessing

import cntk as C
import jieba
import snownlp
from scipy.sparse import csr_matrix
import numpy as np

from benchmark import get_size
from buildctf import get_vocab
from process_text import UNKNOWN_TOKEN


class Predictor(object):
    '''
    Predictor class
    '''

    def __init__(self, vocabulary, labels, model):
        jieba.enable_parallel(multiprocessing.cpu_count())
        self.model = C.load_model(model)
        self.vocab = get_vocab(vocabulary)
        self.x_dim = len(self.vocab)
        self.y_dim = get_size(labels)
        self.x = C.sequence.input_variable(self.x_dim, is_sparse=True)
        self.model = self.model(self.x)
        self.predictor = C.argmax(self.model)

    def predict(self, sentence):
        words = list(
            map(
                lambda w: w if w in self.vocab else UNKNOWN_TOKEN,
                jieba.cut(snownlp.SnowNLP(sentence).han)
            )
        )
        length = len(words)
        inp = list(map(lambda w: self.vocab[w], words))
        matrix = csr_matrix(
            (np.ones(length, ), (range(length), inp)),
            shape=(length, self.x_dim), dtype=np.float32)
        out = self.predictor.eval({self.x: matrix})

        return (words, out)
