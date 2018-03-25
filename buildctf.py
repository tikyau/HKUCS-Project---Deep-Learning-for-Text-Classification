import sys
import argparse
import os

import cntk as C
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d


GAUSSIAN_MODE = "gaussian"
ONEHOT_MODE = "onehot"
SCALER_MODE = "scaler"


def gaussian(num_labels, i):
    result = np.zeros(num_labels)
    result[int(i) - 1] = 1
    return gaussian_filter1d(result, 0.5)


def onehot(num_labels, i):
    return [str(int(i) - 1) + ":1"]


def scaler(num_labels, i):
    return [int(i) - 1]


def get_vocab(vocab_file):
    indexes = {}
    index = 0
    with open(vocab_file) as f:
        for line in f:
            indexes[line[:-1]] = index
            index += 1
    return indexes


def get_sentences(file):
    sentences = []
    with open(file) as f:
        for line in f:
            sentence, label = line[:-1].split('\t')
            words = sentence.split()
            sentences.append((words, label))
    return sentences


def get_map(mode):
    if mode == GAUSSIAN_MODE:
        return gaussian
    elif mode == SCALER_MODE:
        return scaler
    else:
        return onehot


def test(y_dim, mode, vocab_dim, file):
    print("testing ctf")
    y_dim = num_labels if mode != SCALER_MODE else 1
    streamDefs = C.io.StreamDefs(
        sentence=C.io.StreamDef(field="S0", shape=vocab_dim, is_sparse=True),
        label=C.io.StreamDef(field="S1", shape=y_dim)
    )
    reader = C.io.MinibatchSource(
        C.io.CTFDeserializer(file, streamDefs)
    )
    print(reader.next_minibatch(1))


def build(mode, file):
    input_dir = os.path.dirname(file)
    vocab_file = os.path.join(input_dir, "vocabulary.txt")
    prefix = os.path.splitext(file)
    vocab = get_vocab(vocab_file)
    sentences = get_sentences(file)
    mapper = get_map(mode)
    with open(os.path.join(input_dir, "ctf.conf"), "w") as f:
        f.write(mode + "\n")
    num_labels = 0
    with open(os.path.join(input_dir, "labels.txt")) as f:
        num_labels = len(f.readlines())
    result = ""
    with open(prefix + ".ctf", "w") as f:
        for i in range(len(sentences)):
            sentence = np.array([[str(vocab[w]) + ":1"]
                                 for w in sentences[i][0]])
            label = np.array([mapper(num_labels, sentences[i][1])])
            mapping = {"S0": sentence, "S1": label}
            result += C.io.sequence_to_cntk_text_format(i, mapping)
            f.write(result)
            if i % 1000 == 0:
                print("[build] written {} lines".format(i))
    test(num_labels, mode, len(vocab), prefix + ".ctf")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: ./buildctf.py mode file, mode must\
              be one of gaussian, scaler, onehot")
        sys.exit(0)
    build(sys.argv[1], sys.argv[2])
