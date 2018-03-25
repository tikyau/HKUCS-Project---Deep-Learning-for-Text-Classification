#!/usr/bin/env python3
from __future__ import (absolute_import, division, print_function)

from random import shuffle
import subprocess
import multiprocessing
import collections
import itertools
import os
import csv
import json
import sys
import argparse


import jieba
from snownlp import SnowNLP

UNKNOWN_TOKEN = "UNKNOWN"
IGNORED_CHAR = set(["\t", "…", "“", "”"])


def is_Chinese_char(c):
    return any(map(lambda x: c >= x[0] and c <= x[1], (
        ('\u4E00', '\u9FFF'),
        ('\u3400', '\u4DBF'),
        ('\u20000', '\u2A6DF'),
        ('\u2A700', '\u2B73F'),
        ('\u2B740', '\u2B81F'),
        ('\u2B820', '\u2CEAF'),
        ('\u2CEB0', '\u2EBEF'),
        ('\uF900', '\uFAFF')
    )))


def segment(data_field, label_field, row):
    sentence = row[data_field].strip()
    if not sentence:
        return (sentence, [], row[label_field])
    sentence = sentence.replace("\n", "\\n").replace("\t", "\\t").replace('\r', "\\r")
    words = jieba.cut(SnowNLP(sentence).han)
    valid_words = list(filter(lambda x: x and all(map(
        lambda y: is_Chinese_char(y) and y not in IGNORED_CHAR and not y.isspace(), x)),
        words))
    return (sentence, valid_words, row[label_field])


def write_to_file(path, data):
    with open(path + ".origin", "w") as origin, open(path + ".txt", "w") as txt:
        for sentence, words, label in data:
            origin.write("{}\t{}\n".format(sentence, label))
            txt.write("{}\t{}\n".format(" ".join(words), label))


def replace_unknown(known_vocabs, records, train_size):
    print("[build]replacing unknown token...")
    for i in range(train_size, len(records)):
        sentence, words, label = records[i]
        words = list(
            map(lambda x: x if x in known_vocabs else UNKNOWN_TOKEN, words))
        records[i] = (sentence, words, label)


def to_ctf(output_dir, prefix, vocab_file, label_file, mode):
    input_file = os.path.join(output_dir, prefix + ".txt")
    output_file = os.path.join(output_dir, prefix + ".ctf")
    vocab_file = os.path.join(output_dir, vocab_file)
    label_file = os.path.join(output_dir, label_file)
    subprocess.call("./buildctf.py {} {} {} {} {}".format(input_file,
                                                          output_file, vocab_file,
                                                          label_file, mode), shell=True)


def segment_csv(csv_file, data_field, label_field, output_file):
    jieba.enable_parallel(multiprocessing.cpu_count())
    print("[segment]\tprocessing CSV file...")
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f, delimiter=",", quotechar='"')
        result = []
        i = 0
        for row in reader:
            result.append(segment(data_field, label_field, row))
            i += 1
            if i % 1000 == 0:
                print("processed {} rows...".format(i))
    print("[segment]\twriting to result")
    with open(output_file, "w") as f:
        for sentence, words, label in result:
            if not words:
                continue
            f.write("{}\t{}\t{}\n".format(" ".join(words), label, sentence))
    print("[segment] word segmentation successful")


def read_input(input_file, ignored_labels):
    ignored_labels = set(ignored_labels.split(','))
    sentences = collections.defaultdict(list)
    with open(input_file, "r") as f:
        for line in f:
            words, label, sentence = line[:-1].split('\t')
            if label not in ignored_labels:
                sentences[label].append((sentence, words.split()))
    return sentences


def truncate_labels(sentences, even, max_size):
    if even or max_size:
        print("[build] truncating number of sentences from each label...")
        size = min(map(lambda x: len(x), sentences.values()))
        if max_size:
            size = min(max_size, size)
        for label in sentences:
            sentences[label] = sentences[label][:size]


def split(input_file, output_dir, train_prefix="train",
          test_prefix="test", dev_prefix="dev", train_ratio=0.8,
          dev_ratio=0.1, max_size=0, vocab_file="vocabulary.txt",
          label_file="labels.txt", ignored_labels="",
          even=False, ctf=""):
    assert(train_ratio + dev_ratio < 1)
    sentences = read_input(input_file, ignored_labels)
    truncate_labels(sentences, even, max_size)
    labels = sentences.keys()
    records = list(itertools.chain.from_iterable([[(i[0], i[1], label) for i in sentences[label]]
                                                  for label in sentences.keys()]))
    del sentences
    print("[build] splitting to train/dev/test set...")
    shuffle(records)
    train_size = int(len(records) * train_ratio)
    dev_size = int(len(records) * dev_ratio)
    known_vocabs = collections.Counter(itertools.chain.from_iterable(
        i[1] for i in records[:train_size]))
    known_vocabs[UNKNOWN_TOKEN] = 1
    replace_unknown(known_vocabs, records, train_size)
    print("[build] writing to files...")
    write_to_file(os.path.join(output_dir, train_prefix),
                  records[:train_size])
    write_to_file(os.path.join(output_dir, dev_prefix),
                  records[train_size:train_size + dev_size])
    write_to_file(os.path.join(output_dir, test_prefix),
                  records[train_size + dev_size:])
    with open(os.path.join(output_dir, vocab_file), "w") as f:
        f.write("\n".join((i[0] for i in known_vocabs.most_common())))
        f.write('\n')
    with open(os.path.join(output_dir, label_file), "w") as f:
        f.write("\n".join(labels))
    if ctf:
        print("[build]Converting to ctf files...")
        to_ctf(output_dir, train_prefix, vocab_file, label_file, ctf)
        to_ctf(output_dir, dev_prefix, vocab_file, label_file, ctf)
        to_ctf(output_dir, test_prefix, vocab_file, label_file, ctf)


def generate_CTF(dataset_file_path, vocab_file_path, label_file_path):
    output_dir = os.path.dirname(dataset_file_path)
    dataset_name = os.path.splitext(os.path.basename(dataset_file_path))[0]
    print('[generate_CTF]\tGenerating CTF file for {} set ...'
          .format(dataset_name))
    output_file = os.path.join(output_dir, dataset_name + '.ctf')
    os.system(('python /usr/local/cntk/Scripts/txt2ctf.py --map {} {} ' +
               '--annotated True --input {} --output {}')
              .format(
        vocab_file_path, label_file_path,
        dataset_file_path, output_file))
    print('[generate_CTF]\tCTF file {} successfully generated!'
          .format(output_file))


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print(
            'First argument must be "segment", "split", or "ctf".',
            file=sys.stderr)
        sys.exit(0)
    if sys.argv[1] == 'segment':
        parser = argparse.ArgumentParser(
            description='Build dataset from CSV. segment the sentences')
        POSITIONALS = [("csv_file", "path to csv file"),
                       ("data_field", "field name of data entry in CSV"),
                       ("label_field", "field name of label entry in CSV"),
                       ("output_file", "output filename")]

        for arg, h in POSITIONALS:
            parser.add_argument(arg, help=h)
 
        args = vars(parser.parse_args(sys.argv[2:]))
        positionals = [args[i[0]] for i in POSITIONALS]
        segment_csv(*positionals)
    elif sys.argv[1] == "split":
        parser = argparse.ArgumentParser(
            description='Build dataset from CSV. segment the sentences\
            and split dataset.')
        POSITIONALS = [
            ("input_file", "file containing segmented sentences"),
            ("output_dir", "directory for output")
        ]
        OPTIONALS = [
            ("train_prefix", "", "train", str),
            ("dev_prefix", "", "dev", str),
            ("test_prefix", "", "test", str),
            ("train_ratio", "", 0.8, float),
            ("dev_ratio", "", 0.1, float),
            ("vocab_file", "filename for vocabulary", "vocabulary.txt", str),
            ("label_file", "filename for labels", "labels.txt", str),
            ("max_size", "maximum number of entries from each label", 0, int),
            ("ignored_labels", "labels to be ignored", "", str),
            ("even", "keep numbers of entries from all labels balanced", False, bool),
            ("ctf", "perform txt2ctf on the output", "", str)
        ]
        for arg, h in POSITIONALS:
            parser.add_argument(arg, help=h)
        for arg, h, default, t in OPTIONALS:
            parser.add_argument("--" + arg, help=h, default=default, type=t)
        args = vars(parser.parse_args(sys.argv[2:]))
        positionals = []
        output_dir = args["output_dir"]
        for arg, h in POSITIONALS:
            positionals.append(args[arg])
            del args[arg]
        split(*positionals, **args)
        with open(os.path.join(output_dir, "build.conf"), "w") as f:
            json.dump(args, f)
    elif sys.argv[1] == 'ctf':
        parser = argparse.ArgumentParser(
            description='Generate CNTK Text Format file.'
        )
        parser.add_argument('dataset_file_path', help='path to dataset file')
        parser.add_argument("vocab_label_dir",
                            help="directory to vocabulary and label files")
        parser.add_argument(
            '--vocab_file', help='path to vocabulary file',
            default='vocabulary.txt'
        )
        parser.add_argument(
            '--label_file', help='path to labels file',
            default='labels.txt'
        )

        args = vars(parser.parse_args(sys.argv[2:]))

        generate_CTF(
            args['dataset_file_path'],
            vocab_file_path=os.path.join(
                args["vocab_label_dir"], args['vocab_file']),
            label_file_path=os.path.join(
                args["vocab_label_dir"], args['label_file'])
        )
    else:
        print(
            'First argument must be "segment", "split", or "ctf".',
            file=sys.stderr)
