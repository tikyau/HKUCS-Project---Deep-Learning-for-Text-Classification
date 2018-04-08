import multiprocessing
import sys


import cntk as C
import jieba
import snownlp
from scipy.sparse import csr_matrix

from benchmark import get_size
from buildctf import get_vocab
from process_text import UNKNOWN_TOKEN


class Predictor(object):
    def __init__(self, vocabulary, labels, model):
        jieba.enable_parallel(multiprocessing.cpu_count())
        self.model = C.load_model(model)
        self.vocab = get_vocab(vocabulary)
        self.y_dim = len(self.vocab)
        self.y_dim = get_size(labels)
        self.x = C.sequence.input_variable(self.x_dim, is_sparse=True)
        self.model = self.model(x)
        self.predict = C.argmax(self.model)

    def predict(self, sentence):
        words = jieba.cut(snownlp.SnowNLP(sentence).han)
        length = len(words)
        words = list(
            map(lambda x: self.vocab[x] if x in self.vocab else self.vocab[UNKNOWN_TOKEN], words))
        matrix = csr_matrix(
            (np.ones(length, ), (length, words)), shape=(length, self.x_dim))
        print(self.predict.eval({self.x: matrix})[0])


def main():
    predictor = Predictor(sys.argv[1], sys.argv[2], sys.argv[3])
    while True:
        inp = input(">>")
        if inp == ":q":
            break
        predictor.predict(inp)


if __name__ == "__main__":
    main()
