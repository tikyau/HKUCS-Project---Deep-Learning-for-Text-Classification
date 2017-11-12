from __future__ import (absolute_import, division, print_function)

from functools import reduce
from random import shuffle

import os
import collections
import csv

import numpy as np
import matplotlib.pyplot as plt

import jieba


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
        dataset_file_name='data.txt',
        vocab_file_name='vocabulary.txt',
        label_file_name='labels.txt',
        report_freq=100,
        segmnt_func=jieba.cut,
        filter_func=is_Chinese):
    vocb = list()
    data = list()
    labl = set()

    jieba.enable_parallel(8)

    print('[build_dataset]\tReading CSV file ...')
    with open(csv_file_path, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter=delimiter, quotechar=quotechar)
        record = collections.defaultdict(list)
        for row in reader:
            record[row[label_field]].append(row[data_field].strip())

    minSize = min(len(record[i]) for i in record)
    for label in record:
        sentences = record[label][:minSize]
        shuffle(sentences)
        for i, sentence in enumerate(sentences):
            if i % report_freq == 0:
                print('[build_dataset]\tProcessing line #{} ...'.format(i + 1))
            sentence = list(filter(
                lambda x: x and filter_func(x),
                segmnt_func(sentence)
            ))
            if sentence:
                data.append((sentence, label))
                vocb += sentence

    print('[build_dataset]\tBuilding vocabulary ...')
    vocb = collections.Counter(vocb).most_common()

    print(('[build_dataset]\tSuccessfully extracted {} words for vocabulary ' +
           'and {} labels from {} entries for data.')
          .format(len(vocb), len(labl), len(data)))

    print('[build_dataset]\tWriting vocabulary to file {} ...'
          .format(vocab_file_name))
    with open(vocab_file_name, 'w') as vocabulary_file:
        for word, _ in vocb:
            vocabulary_file.write('{}\n'.format(word))

    print('[build_dataset]\tWriting labels to file {} ...'
          .format(label_file_name))
    with open(label_file_name, 'w') as label_file:
        for label in record.keys():
            label_file.write('{}\n'.format(label))

    print('[build_dataset]\tWriting dataset to file {} ...'
          .format(dataset_file_name))
    with open(dataset_file_name, 'w') as data_file:
        for sentence, label in data:
            data_file.write('{}\t{}\n'.format(' '.join(sentence), label))

    del vocb
    del labl
    del data

    print('[build_dataset]\tCompleted!')


def train_test_split(
        dataset_file_path, train_file_name='train.txt',
        test_file_name='test.txt', train_ratio=0.80):
    print('[train_test_split]\tReading data file ...')

    with open(dataset_file_path, 'r') as data_file:
        lines = data_file.readlines()

    print('[train_test_split]\tShuffling data entries ...')
    shuffle(lines)
    train_num = int(len(lines) * train_ratio)
    print(('[train_test_split]\tRandomly sampled {} entries for training, ' +
           '{} entries for testing.').format(train_num, len(lines) - train_num))

    print('[train_test_split]\tWriting training set ...')
    with open(train_file_name, 'w') as train_file:
        train_file.writelines(lines[:train_num])

    print('[train_test_split]\tWriting testing set ...')
    with open(test_file_name, 'w') as test_file:
        test_file.writelines(lines[train_num:])

    del lines


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
            if min_x >=0 and int(row[x_field].strip()) < min_x:
                continue
            if max_x >=0 and int(row[x_field].strip()) > max_x:
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

    if len(sys.argv) > 1 and sys.argv[1] == '--build':
        parser = argparse.ArgumentParser(description='Build dataset from CSV.')
        parser.add_argument('csv_file_path', help='path to CSV file')
        parser.add_argument(
            'data_field', help='field name of data entry in CSV')
        parser.add_argument(
            'label_field', help='field name of label entry in CSV')
        parser.add_argument(
            '--dataset_file_name', help='where to store dataset',
            default='data.txt'
        )
        parser.add_argument(
            '--vocab_file_name', help='where to store vocabulary',
            default='vocabulary.txt'
        )
        parser.add_argument(
            '--label_file_name', help='where to store labels',
            default='labels.txt'
        )

        args = vars(parser.parse_args(sys.argv[2:]))

        build_dataset(
            args['csv_file_path'],
            args['data_field'],
            args['label_field'],
            dataset_file_name=args['dataset_file_name'],
            vocab_file_name=args['vocab_file_name'],
            label_file_name=args['label_file_name']
        )

    elif len(sys.argv) > 1 and sys.argv[1] == '--split':
        parser = argparse.ArgumentParser(description='Preprocess datasets.')
        parser.add_argument('dataset_file_path', help='path to dataset file')
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

        args = vars(parser.parse_args(sys.argv[2:]))

        train_test_split(
            args['dataset_file_path'],
            train_file_name=args['train_file_name'],
            test_file_name=args['test_file_name'],
            train_ratio=args['train_ratio']
        )

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
            'First argument must be "--build", "--split", "--ctf", or "--plot".'
            , file=sys.stderr
        )
