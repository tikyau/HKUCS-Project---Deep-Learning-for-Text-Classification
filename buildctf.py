import cntk as C
import sys
import numpy as np


def convert(filename):
    sentences = []
    vocab = set()
    with open(filename) as f:
        for line in f:
            sentence, label = line[:-1].split('\t')
            words = sentence.split()
            for word in words:
                vocab.add(word)
            sentences.append((words, int(label)))
    indexes = {w: i for (i, w) in enumerate(vocab)}
    with open("test.ctf", "w") as f:
        for i in range(len(sentences)):
            s = np.zeros(shape=(len(sentences[i][0]), len(vocab)))
            for j in range(len(sentences[i][0])):
                s[j, indexes[sentences[i][0][j]]] = 1
            print(s)

            s = np.array([[str(indexes[w])+":1"] for w in sentences[i][0]])
            label = np.array([[sentences[i][1]]])
            mapping = {"S0": s, "S1": label}
            print(C.io.sequence_to_cntk_text_format(i, mapping))
            sys.exit(0)
    return len(vocab)


l = convert(sys.argv[1])
