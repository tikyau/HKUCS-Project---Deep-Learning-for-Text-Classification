#!/usr/bin/env python3
import sys

import cntk as C
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d


GAUSSIAN_MODE = "gaussian"
ONEHOT_MODE = "onehot"
SCALER_MODE = "scaler"


def onehot(label2index, i):
    result = np.zeros(len(label2index))
    result[label2index[i]] = 1
    return result


def gaussian(label2index, i):
    return gaussian_filter1d(onehot(label2index, i), 0.5)


def scaler(label2index, i):
    return [label2index[i]]


def get_vocab(vocab_file):
    indexes = {}
    index = 0
    with open(vocab_file) as f:
        for line in f:
            indexes[line[:-1]] = index
            index += 1
    assert("UNKNOWN" in indexes)
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
    if mode == SCALER_MODE:
        return scaler
    return onehot


def test(y_dim, mode, vocab_dim, file):
    print("testing ctf")
    y_dim = y_dim if mode != SCALER_MODE else 1
    streamDefs = C.io.StreamDefs(
        sentence=C.io.StreamDef(field="S0", shape=vocab_dim, is_sparse=True),
        label=C.io.StreamDef(field="S1", shape=y_dim)
    )
    reader = C.io.MinibatchSource(
        C.io.CTFDeserializer(file, streamDefs)
    )
    print(reader.next_minibatch(1))


def build(in_file, out_file, vocab_file, label_file, mode):
    vocab = get_vocab(vocab_file)
    print("{} vocabularies in total".format(len(vocab)))
    sentences = get_sentences(in_file)
    mapper = get_map(mode)
    num_labels = 0
    labels = []
    with open(label_file) as f:
        for line in f:
            labels.append(int(line[:-1]))
    labels = list(sorted(labels))
    label2index = {str(labels[i]): i for i in range(len(labels))}
    with open(out_file, "w") as f:
        for i in range(len(sentences)):
            sentence = np.array([[str(vocab[w]) + ":1"]
                                 for w in sentences[i][0]])
            label = np.array([mapper(label2index, sentences[i][1])])
            mapping = {"S0": sentence, "S1": label}
            result = C.io.sequence_to_cntk_text_format(i, mapping)
            f.write(result)
            f.write('\n')
            if i % 1000 == 0:
                print("[build] written {} lines".format(i))
    test(num_labels, mode, len(vocab), out_file)


if __name__ == "__main__":
    build(*sys.argv[1:])
