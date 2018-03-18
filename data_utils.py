from __future__ import (absolute_import, division, print_function)

from functools import reduce
from random import shuffle
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import itertools
import os
import collections
import csv
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

import jieba

UNKNOWN_TOKEN = "UNKNOWN"

known_vocabularies = set()


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


def replace(i):
    return "{}\t{}".format(
        " ".join(
            list(map(lambda x: x if x in known_vocabularies else UNKNOWN_TOKEN,
                     i[0]))),
        i[1]
    )


def remove(i):
    return "{}\t{}".format(
        " ".join(list(filter(lambda x: x in known_vocabularies, i[0]))), i[1]
    )


def build_dataset(
        csv_file_path, data_field, label_field,
        delimiter=',', quotechar='"',
        output_file="data.txt",
        report_freq=100,
        segmnt_func=jieba.cut,
        filter_func=is_Chinese_char,
        unknown_number=0):

    jieba.enable_parallel(multiprocessing.cpu_count())

    print('[build_dataset]\tReading CSV file ...')
    with open(csv_file_path, 'r', newline='') as f, open(output_file, 'w') as g:
        reader = csv.DictReader(f, delimiter=delimiter, quotechar=quotechar)
        i = 0
        for row in reader:
            sentence = list(filter(
                lambda x: x and filter_func(x) and not x.isspace(),
                segmnt_func(row[data_field].strip())
            ))
            if sentence:
                g.write("{}\t{}\n".format(
                    ' '.join(sentence), row[label_field]))
                i += 1
            if i % report_freq == 0:
                print("[build_dataset]\tProcessing line #{} ...".format(i + 1))
    print('[build_dataset]\tCompleted!')


def filter_and_write(pool, records, file_name, func):
    with open(file_name, "w") as f:
        new_sentences = filter(lambda x: x[0] != '\t', pool.map(func, records))
        f.write("\n".join(new_sentences))
        f.write("\n")


def split(data_file_name,
          train_file_name="train.txt",
          test_file_name="test.txt",
          dev_file_name="dev.txt",
          train_ratio=0.8,
          dev_ratio=0.1,
          max_size=0,
          remove_unknown=False,
          unknown_number=0,
          output_dir=".",
          vocab_file_name="vocabulary.txt",
          label_file_name="labels.txt", ignored_labels="", even=False):
    assert(train_ratio + dev_ratio < 1)
    ignored_labels = set(ignored_labels.split(','))
    records = collections.defaultdict(list)
    print("[Split dataset]\tReading from file...")
    with open(data_file_name, 'r') as f:
        for line in f:
            sentence, label = line[:-1].split('\t')
            if label in ignored_labels:
                continue
            records[label].append(sentence.split())
    print("[Split dataset]\tBuilding vocabulary...")
    labels = records.keys()
    if even or max_size:
        min_size = min(map(lambda x: len(x), records.values()))
        if max_size:
            min_size = min(max_size, min_size)
        records = list(itertools.chain.from_iterable(
            [(i, label) for i in records[label][:min_size]]
            for label in records.keys()))
    else:
        records = list(itertools.chain.from_iterable(
            [(i, label) for i in records[label]]
            for label in records.keys()))
    shuffle(records)
    train_size = int(len(records) * train_ratio)
    dev_size = int(len(records) * dev_ratio)
    all_words = itertools.chain.from_iterable(
        i[0] for i in records[:train_size])
    vocabs = [i[0] for i in collections.Counter(
        all_words).most_common()[unknown_number:]]
    global known_vocabularies
    known_vocabularies = set(vocabs)
    if not remove_unknown:
        known_vocabularies.add(UNKNOWN_TOKEN)
    print("[Split dataset]\tWriting to files...")
    filter_func = remove if remove_unknown else replace
    with ProcessPoolExecutor() as pool:
        filter_and_write(pool, records[:train_size], os.path.join(
            output_dir, train_file_name), filter_func)
        filter_and_write(
            pool, records[train_size: train_size + dev_size],
            os.path.join(output_dir, dev_file_name), filter_func)
        filter_and_write(
            pool, records[train_size + dev_size:],
            os.path.join(output_dir, test_file_name), filter_func)
    with open(os.path.join(output_dir, vocab_file_name), "w") as f:
        f.write("\n".join(known_vocabularies))
        f.write("\n")
    with open(os.path.join(output_dir, label_file_name), "w") as f:
        f.write("\n".join(labels))
        f.write("\n")


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
    import sys
    import argparse

    if len(sys.argv) > 1 and sys.argv[1] == 'segment':
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
    elif len(sys.argv) > 1 and sys.argv[1] == "split":
        parser = argparse.ArgumentParser(
            description="split dataset and store vocabularies")
        parser.add_argument(
            "data_file_name", help="input data"
        )
        parser.add_argument(
            "--output_dir", help="output directory",
            default="."
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
            "--dev_file_name", help="where to store testing dataset",
            default="dev.txt"
        )
        parser.add_argument(
            '--train_ratio', help='training set ratio',
            type=float, default=0.8
        )
        parser.add_argument(
            "--dev_ratio", help="development set ratio",
            type=float, default=0.1
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
            "--even", help="whether to subsample to make all labels even",
            default=False,
            type=bool
        )
        parser.add_argument(
            "--ignored_labels", help="ingore labels, delimited by comma",
            default="",
            type=str
        )
        parser.add_argument(
            "--remove_unknown", help="whether to remove unknown words",
            default=False, type=bool
        )
        parser.add_argument(
            "--unknown_number", help="top n words to be labelled as unknown",
            default=0, type=int
        )
        parser.add_argument(
            "--max_size", help="maximum number of data from each label",
            default=0, type=int
        )
        args = vars(parser.parse_args(sys.argv[2:]))
        file_name = args["data_file_name"]
        del args["data_file_name"]
        split(file_name, **args)
        with open("{}/split.conf".format(args["output_dir"]), "w") as f:
            json.dump(args, f)
    elif len(sys.argv) > 1 and sys.argv[1] == 'ctf':
        parser = argparse.ArgumentParser(
            description='Generate CNTK Text Format file.'
        )
        parser.add_argument('dataset_file_path', help='path to dataset file')
        parser.add_argument("vocab_label_dir",
                            help="directory to vocabulary and label files")
        parser.add_argument(
            '--vocab_file_name', help='path to vocabulary file',
            default='vocabulary.txt'
        )
        parser.add_argument(
            '--label_file_name', help='path to labels file',
            default='labels.txt'
        )

        args = vars(parser.parse_args(sys.argv[2:]))

        generate_CTF(
            args['dataset_file_path'],
            vocab_file_path=os.path.join(
                args["vocab_label_dir"], args['vocab_file_name']),
            label_file_path=os.path.join(
                args["vocab_label_dir"], args['label_file_name'])
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
            'First argument must be "segment", "split", "ctf", or "plot".',
            file=sys.stderr)
