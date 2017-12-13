from __future__ import (absolute_import, division, print_function)

from functools import reduce
from random import shuffle
import itertools
import os
import collections
import csv

import numpy as np
import matplotlib.pyplot as plt

import jieba

UNKNOWN_TOKEN = "UNKNOWN"


def default(v, d):
    if v is not None:
        return v
    return d


def is_Chinese(s):
    return reduce(
        lambda x, y: x or y,
        map(
            lambda p: reduce(
                lambda x, y: x and y,
                map(lambda c: not c.isspace() and p[0] <= c <= p[1], s),
                True
            ),
            (
                ('\u4E00', '\u9FFF'),
                ('\u3400', '\u4DBF'),
                ('\u20000', '\u2A6DF'),
                ('\u2A700', '\u2B73F'),
                ('\u2B740', '\u2B81F'),
                ('\u2B820', '\u2CEAF'),
                ('\u2CEB0', '\u2EBEF'),
                ('\uF900', '\uFAFF')
            )
        ),
        False
    )


def has_Chinese(s):
    return reduce(
        lambda x, y: x or y,
        map(
            lambda p: reduce(
                lambda x, y: x or y,
                map(lambda c: not c.isspace() and p[0] <= c <= p[1], s),
                True
            ),
            (
                ('\u4E00', '\u9FFF'),
                ('\u3400', '\u4DBF'),
                ('\u20000', '\u2A6DF'),
                ('\u2A700', '\u2B73F'),
                ('\u2B740', '\u2B81F'),
                ('\u2B820', '\u2CEAF'),
                ('\u2CEB0', '\u2EBEF'),
                ('\uF900', '\uFAFF')
            )
        ),
        False
    )


def build_dataset(
        csv_file_path, data_field, label_field,
        delimiter=',', quotechar='"',
        output_file="data.txt",
        report_freq=100,
        segmnt_func=jieba.cut,
        filter_func=is_Chinese,
        unknown_number=0):

    jieba.enable_parallel(8)

    print('[build_dataset]\tReading CSV file ...')
    with open(csv_file_path, 'r', newline='') as f, open(output_file, 'w') as g:
        reader = csv.DictReader(f, delimiter=delimiter, quotechar=quotechar)
        i = 0
        for row in reader:
            sentence = list(filter(
                lambda x: x and filter_func(x),
                segmnt_func(row[data_field].strip())
            ))
            if sentence:
                g.write("{}\t{}\n".format(
                    ' '.join(sentence), row[label_field]))
                i += 1
            if i % report_freq == 0:
                print("[build_dataset]\tProcessing line #{} ...".format(i + 1))
    print('[build_dataset]\tCompleted!')


def split(data_file_name,
          train_file_name="train.txt",
          test_file_name="test.txt",
          train_ratio=0.8,
          unknown_number=0,
          vocab_file_name="vocabulary.txt",
          label_file_name="label.txt"):
    record = []
    with open(data_file_name, 'r') as f:
        for line in f:
            sentence, label = line[:-1].split('\t')
            record.append((sentence.split(' '), label))
    shuffle(record)
    trainNum = int(len(record) * train_ratio)
    allWords = itertools.chain.from_iterable([i[0] for i in record])
    vocabularyCounter = collections.Counter(allWords)
    f


def generate_CTF(dataset_file_path, vocab_file_path, label_file_path):
    dataset_name = dataset_file_path.strip().split('/')[-1].split('.')[0]
    print('[generate_CTF]\tGenerating CTF file for {} set ...'
          .format(dataset_name))
    output_file = dataset_name + '.ctf'
    os.system(('python /usr/local/cntk/Scripts/txt2ctf.py --map {} {} ' +
               '--annotated True --input {} --output {}')
              .format(
        vocab_file_path, label_file_path,
        dataset_file_path, output_file))
    print('[generate_CTF]\tCTF file {} successfully generated!'
          .format(output_file))


def plot_log(
        log_file_path, x_field, y_field, smooth_factor, to_accuracy,
        x_label, y_label, title, color, transparent, min_x, max_x):
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
                        i, row[x_field].strip()
                    ))
                try:
                    y.append(
                        1.0 - float(row[y_field].strip()) if to_accuracy else
                        float(row[y_field].strip())
                    )
                except ValueError:
                    raise ValueError('y-axis value error at line {}: {}'.format(
                        i, row[y_field].strip()
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
        '{}.png'.format(title), bbox_inches='tight', transparent=transparent
    )


if __name__ == "__main__":
    import sys
    import argparse

    if len(sys.argv) > 1 and sys.argv[1] == '--segment':
        parser = argparse.ArgumentParser(description='Build dataset from CSV.')
        parser.add_argument('csv_file_path', help='path to CSV file')
        parser.add_argument(
            'data_field', help='field name of data entry in CSV')
        parser.add_argument(
            'label_field', help='field name of label entry in CSV')
        parser.add_argument(
            "--output_file", help="output file",
            default="data.txt"
        )
        args = vars(parser.parse_args(sys.argv[2:]))

        build_dataset(
            args['csv_file_path'],
            args['data_field'],
            args['label_field'],
            output_file=args["output_file"]
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "--split":
        parser.add_argument(
            "data_file_name", help="input data",
            default="data.txt"
        )
        parser.add_argument(
            '--train_file_name', help='where to store training dataset',
            default='train.txt'
        )
        parser.add_argument(
            '--test_file_name', help='where to store testing dataset',
            default='test.txt'
        )
        parser.add_argument(
            '--train_ratio', help='training set ratio',
            type=float, default=0.8
        )
        parser.add_argument(
            '--vocab_file_name', help='where to store vocabulary',
            default='vocabulary.txt'
        )
        parser.add_argument(
            '--label_file_name', help='where to store labels',
            default='labels.txt'
        )

        parser.add_argument(
            "--unknown_number", help="top n words to be labelled as unknown",
            default=0, type=int
        )
        args = vars(parser.parse_args(sys.argv[2:]))
        split(args["data_file_name"], **args)

    elif len(sys.argv) > 1 and sys.argv[1] == '--ctf':
        parser = argparse.ArgumentParser(
            description='Generate CNTK Text Format file.'
        )
        parser.add_argument('dataset_file_path', help='path to dataset file')
        parser.add_argument(
            '--vocab_file_path', help='path to vocabulary file',
            default='vocabulary.txt'
        )
        parser.add_argument(
            '--label_file_path', help='path to labels file',
            default='labels.txt'
        )

        args = vars(parser.parse_args(sys.argv[2:]))

        generate_CTF(
            args['dataset_file_path'],
            vocab_file_path=args['vocab_file_path'],
            label_file_path=args['label_file_path']
        )

    elif len(sys.argv) > 1 and sys.argv[1] == '--plot':
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
            min_x=args['min_x'],
            max_x=args['max_x']
        )
    else:
        print(
            'First argument must be "--segment", "--ctf", or "--plot".', file=sys.stderr
        )
