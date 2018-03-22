from __future__ import (absolute_import, division, print_function)

from concurrent.futures import ProcessPoolExecutor
from random import shuffle
import multiprocessing
import collections
import itertools
import os
import csv
import json
import sys
import argparse

import numpy as np

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


def segment(sentence):
    words = jieba.cut(SnowNLP(sentence).han)
    valid_words = list(filter(lambda x: x and all(map(
        lambda y: is_Chinese_char(y) and y not in IGNORED_CHAR and not y.isspace(), x)),
                              words))
    del words
    return valid_words


def write_to_file(path, data):
    with open(path + ".raw", "w") as raw, open(path + ".txt", "w") as txt:
        for sentence, words, label in data:
            raw.write("{}\t{}\n".format(sentence, label))
            txt.write("{}\t{}\n".format(" ".join(words), label))


def replace_unknown(known_vocabs, records, train_size):
    print("[build]replacing unknown token...")
    for i in range(train_size, len(records)):
        sentence, words, label = records[i]
        words = list(
            map(lambda x: x if x in known_vocabs else UNKNOWN_TOKEN, words))
        records[i] = (sentence, words, label)


def read_csv(ignored_labels, csv_file, label_field, data_field):
    sentences = collections.defaultdict(list)
    ignored_labels = set(ignored_labels.split(','))
    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=',', quotechar='"')
        for row in reader:
            if row[label_field] not in ignored_labels and row[data_field].strip():
                sentences[row[label_field]].append(row[data_field].strip())
    return sentences


def to_ctf(output_dir, prefix, vocab_file, label_file):
    input_file = os.path.join(output_dir, prefix + ".txt")
    output_file = os.path.join(output_dir, prefix + ".ctf")
    vocab_file = os.path.join(output_dir, vocab_file)
    label_file = os.path.join(output_dir, label_file)
    os.system(('python /usr/local/cntk/Scripts/txt2ctf.py --map {} {} ' +
               '--annotated True --input {} --output {}')
              .format(
        vocab_file, label_file,
        input_file, output_file))


def build_tuple(sentence):
    return (sentence, segment(sentence))


def build_dataset(csv_file, data_field, label_field,
                  output_dir,
                  train_prefix="train",
                  test_prefix="test",
                  dev_prefix="dev",
                  train_ratio=0.8,
                  dev_ratio=0.1,
                  max_size=0,
                  vocab_file="vocabulary.txt",
                  label_file="labels.txt", ignored_labels="",
                  even=False, ctf=True):
#    jieba.enable_parallel(multiprocessing.cpu_count())
    assert(train_ratio + dev_ratio < 1)
    print("[build]\treading CSV file...")
    sentences = read_csv(ignored_labels, csv_file, label_field, data_field)
    if even or max_size:
        print("[build] truncating number of sentences from each label...")
        size = min(map(lambda x: len(x), sentences.values()))
        if max_size:
            size = min(max_size, size)
        for label in sentences:
            sentences[label] = sentences[label][:size]
    labels = sentences.keys()

    print("[build] segmenting dataset...")
    records = []
    with ProcessPoolExecutor() as pool:
        for label in sentences:
            print("[build] segmeting sentences from {}...".format(label))
            sentences_in_label = pool.map(build_tuple, sentences[label])
            records.extend((i[0], i[1], label) for i in sentences_in_label)
        del sentences
    print("[build] splitting to train/dev/test set...")
    shuffle(records)
    train_size = int(len(records) * train_ratio)
    dev_size = int(len(records) * dev_ratio)
    known_vocabs = set(itertools.chain.from_iterable(
        i[1] for i in records[:train_size]))
    known_vocabs.add(UNKNOWN_TOKEN)
    replace_unknown(known_vocabs, records, train_size)
    print("[build] writing to files...")
    write_to_file(os.path.join(output_dir, train_prefix),
                  records[:train_size])
    write_to_file(os.path.join(output_dir, dev_prefix),
                  records[train_size:train_size + dev_size])
    write_to_file(os.path.join(output_dir, test_prefix),
                  records[train_size + dev_size:])
    with open(os.path.join(output_dir, vocab_file), "w") as f:
        f.write("\n".join(known_vocabs))
    with open(os.path.join(output_dir, label_file), "w") as f:
        f.write("\n".join(labels))
    if ctf:
        print("[build]Converting to ctf files...")
        to_ctf(output_dir, train_prefix, vocab_file, label_file)
        to_ctf(output_dir, dev_prefix, vocab_file, label_file)
        to_ctf(output_dir, test_prefix, vocab_file, label_file)


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


def plot_log(
        log_file_path, x_field, y_field, smooth_factor, to_accuracy,
        x_label, y_label, title, color, transparent, dpi, min_x, max_x):
    import matplotlib.pyplot as plt
    def _running_average_smooth(y, window_size):
        kernel = np.ones(window_size) / window_size
        y_pad = np.lib.pad(y, (window_size, ), 'edge')
        y_smooth = np.convolve(y_pad, kernel, mode='same')
        return y_smooth[window_size:-window_size]

    x = list()
    y = list()

    print('[plot_log]\tReading CSV log file ...')

    with open(log_file_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if min_x >= 0 and int(row[x_field].strip()) < min_x:
                continue
            if max_x >= 0 and int(row[x_field].strip()) > max_x:
                break
            if row[y_field].strip().lower() != 'nan':
                try:
                    x.append(int(row[x_field].strip()))
                except ValueError:
                    raise ValueError('x-axis value error at line {}: {}'.format(
                        i + 1, row[x_field].strip()
                    ))
                try:
                    y.append(
                        1.0 - float(row[y_field].strip()) if to_accuracy else
                        float(row[y_field].strip())
                    )
                except ValueError:
                    raise ValueError('y-axis value error at line {}: {}'.format(
                        i + 1, row[y_field].strip()
                    ))

    x = np.asarray(x)
    y = np.asarray(y)

    print('[plot_log]\tPlotting data ...')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.plot(x, y, alpha=0.2, color=color)
    plt.plot(
        x, _running_average_smooth(y, smooth_factor), color=color, linewidth=2
    )

    plt.grid()

    print('[plot_log]\tSaving figure ...')

    plt.savefig(
        '{}.png'.format(title.replace(' ', '_')),
        bbox_inches='tight', transparent=transparent, dpi=dpi
    )


if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == 'build':
        parser = argparse.ArgumentParser(
            description='Build dataset from CSV. segment the sentences\
            and split dataset.')
        POSITIONALS = [("csv_file", "path to csv file"),
                       ("data_field", "field name of data entry in CSV"),
                       ("label_field", "field name of label entry in CSV"),
                       ("output_dir", "directory for output files")]
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
            ("ctf", "perform txt2ctf on the output", True, bool)
        ]
        for arg, h in POSITIONALS:
            parser.add_argument(arg, help=h)
        for arg, h, default, t in OPTIONALS:
            parser.add_argument("--" + arg, help=h, default=default, type=t)

        args = vars(parser.parse_args(sys.argv[2:]))

        csv_file = args["csv_file"]
        data_field = args["data_field"]
        label_field = args["label_field"]
        output_dir = args["output_dir"]
        pargs = []
        for i in POSITIONALS:
            pargs.append(args[i[0]])
            del args[i[0]]
        build_dataset(
            *pargs,
            **args
        )
        with open(os.path.join(pargs[-1], "build.conf"), "w") as f:
            json.dump(args, f)

    elif len(sys.argv) > 1 and sys.argv[1] == 'ctf':
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

    elif len(sys.argv) > 1 and sys.argv[1] == 'plot':
        parser = argparse.ArgumentParser(
            description='Plot CSV log file.'
        )
        parser.add_argument('log_file_path', help='path to log file')
        parser.add_argument(
            'x_field', help='field name of x-axis in CSV')
        parser.add_argument(
            'y_field', help='field name of y-axis in CSV')
        parser.add_argument(
            'x_label', help='label for x-axis in figure')
        parser.add_argument(
            'y_label', help='label for y-axis in figure')
        parser.add_argument(
            'title', help='title for figure')
        parser.add_argument(
            '--color', help='color of line', default='yellowgreen'
        )
        parser.add_argument(
            '--transparent', help='transparent background',
            type=bool, default=False
        )
        parser.add_argument(
            '--dpi', help='DPI of saved figure file',
            type=int, default=500
        )
        parser.add_argument(
            '--smooth_factor', help='smooth factor',
            type=int, default=9
        )
        parser.add_argument(
            '--to_accuracy', help='convert to accuracy and plot',
            type=bool, default=False
        )
        parser.add_argument(
            '--min_x', help='min x-value to plot',
            type=int, default=-1
        )
        parser.add_argument(
            '--max_x', help='max x-value to plot',
            type=int, default=-1
        )

        args = vars(parser.parse_args(sys.argv[2:]))

        plot_log(
            args['log_file_path'],
            args['x_field'],
            args['y_field'],
            smooth_factor=args['smooth_factor'],
            to_accuracy=args['to_accuracy'],
            x_label=args['x_label'],
            y_label=args['y_label'],
            title=args['title'],
            color=args['color'],
            transparent=args['transparent'],
            dpi=args['dpi'],
            min_x=args['min_x'],
            max_x=args['max_x']
        )
    else:
        print(
            'First argument must be "build", "ctf", or "plot".',
            file=sys.stderr)
