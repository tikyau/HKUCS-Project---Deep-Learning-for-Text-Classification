import multiprocessing
import sys


import cntk as C
import jieba
import snownlp
from scipy.sparse import csr_matrix
import numpy as np

from benchmark import get_size
from buildctf import get_vocab
from process_text import UNKNOWN_TOKEN


class Predictor(object):
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
        words = list(jieba.cut(snownlp.SnowNLP(sentence).han))
        length = len(words)
        print(list(map(lambda x: x if x in self.vocab else UNKNOWN_TOKEN, words)))
        words = list(
            map(lambda x: self.vocab[x] if x in self.vocab else self.vocab[UNKNOWN_TOKEN], words))
        matrix = csr_matrix(
            (np.ones(length, ), (range(length), words)),
            shape=(length, self.x_dim), dtype=np.float32)
        print(self.predictor.eval({self.x: matrix})[0] + 1)


def main():
    predictor = Predictor(sys.argv[1], sys.argv[2], sys.argv[3])
    while True:
        try:
            inp = input(">> ")
        except UnicodeError:
            continue
        if inp == ":q":
            break
        if inp:
            predictor.predict(inp)


if __name__ == "__main__":
    main()
