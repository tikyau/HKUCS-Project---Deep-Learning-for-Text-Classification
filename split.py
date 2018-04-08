from random import shuffle
import collections
import os
import itertools


def split_dataset(input_file, output_dir, train_prefix="train", test_prefix="test",
                  dev_prefix="dev", train_ratio=0.8, dev_ratio=0.1, max_size=0,
                  ignored_labels="", even=False):
    assert(train_ratio + dev_ratio < 1)
    sentences = read_input(input_file, ignored_labels)
    truncate_labels(sentences, even, max_size)
    records = itertools.chain.from_iterable([[(i[0], i[1], label)
                                              for i in sentences[label]]
                                             for label in sentences.keys()])
    del sentences
    records = list(records)
    print("[split] splitting to train/dev/test set...")
    shuffle(records)
    train_size = int(len(records) * train_ratio)
    dev_size = int(len(records) * dev_ratio)
    print("[build] writing to files...")
    write_to_file(os.path.join(output_dir, train_prefix),
                  records[:train_size])
    write_to_file(os.path.join(output_dir, dev_prefix),
                  records[train_size:train_size + dev_size])
    write_to_file(os.path.join(output_dir, test_prefix),
                  records[train_size + dev_size:])


def read_input(input_file, ignored_labels):
    ignored_labels = set(ignored_labels.split(','))
    sentences = collections.defaultdict(list)
    print("reading file...")
    with open(input_file, "r") as f:
        for line in f:
            words, label, sentence = line[:-1].split('\t')
            if label not in ignored_labels:
                sentences[label].append((sentence, words.split()))
    return sentences


def truncate_labels(sentences, even, max_size):
    if even or max_size:
        print("[build] truncating number of sentences from each label...")
        size = min(map(len, sentences.values()))
        if max_size:
            size = min(max_size, size)
        for label in sentences:
            sentences[label] = sentences[label][:size]


def write_to_file(path, data):
    with open(path + ".origin", "w") as origin, open(path + ".txt", "w") as txt:
        for sentence, words, label in data:
            origin.write("{}\t{}\n".format(sentence, label))
            txt.write("{}\t{}\n".format(" ".join(words), label))
