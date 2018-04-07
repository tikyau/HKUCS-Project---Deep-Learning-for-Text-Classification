#!/usr/bin/env python3
from __future__ import (absolute_import, division, print_function)

import os
import json
import sys
import argparse


from buildctf import ONEHOT_MODE


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print(
            'First argument must be "segment", "split", or "ctf".',
            file=sys.stderr)
        sys.exit(0)
    if sys.argv[1] == 'segment':
        from segment import segment_csv
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

            ("max_size", "maximum number of entries from each label", 0, int),
            ("ignored_labels", "labels to be ignored", "", str),
        ]
        FLAGS = [
            ("even", "keep numbers of entries from all labels balanced"),

        ]
        for arg, h in POSITIONALS:
            parser.add_argument(arg, help=h)
        for arg, h, default, t in OPTIONALS:
            parser.add_argument("--" + arg, help=h, default=default, type=t)
        for arg, h in FLAGS:
            parser.add_argument("--" + arg, help=h, action="store_true")
        args = vars(parser.parse_args(sys.argv[2:]))
        positionals = []
        output_dir = args["output_dir"]
        for arg, h in POSITIONALS:
            positionals.append(args[arg])
            del args[arg]
        from split import split_dataset
        split_dataset(*positionals, **args)
        with open(os.path.join(output_dir, "build.conf"), "w") as f:
            json.dump(args, f)
    elif sys.argv[1] == "process":
        parser = argparse.ArgumentParser(
            description='process dataset and build vocabulary/labels')
        parser.add_argument("input_dir", help="directory for splitted dataset")
        OPTIONALS = [
            ("train_prefix", "", "train", str),
            ("dev_prefix", "", "dev", str),
            ("test_prefix", "", "test", str),
            ("vocab_file", "filename for vocabulary", "vocabulary.txt", str),
            ("label_file", "filename for labels", "labels.txt", str),
            ("unknown_threshold", "threshold for marking word as UNKOWN", 0, int)
        ]
        FLAGS = [
            ("no_filter", "filter non-Chinese words")
        ]
        for arg, h, default, t in OPTIONALS:
            parser.add_argument("--" + arg, help=h, default=default, type=t)
        for arg, h in FLAGS:
            parser.add_argument("--" + arg, help=h, action="store_true")
        args = vars(parser.parse_args(sys.argv[2:]))
        input_dir = args["input_dir"]
        del args["input_dir"]
        from process_text import process
        process(input_dir, **args)
    elif sys.argv[1] == 'ctf':
        parser = argparse.ArgumentParser(
            description='Generate CNTK Text Format file.'
        )
        parser.add_argument('dataset_file_path', help='path to dataset file')
        parser.add_argument("--vocab_label_dir",
                            help="directory to vocabulary and label files",
                            default="", type=str)
        parser.add_argument(
            '--vocab_file', help='path to vocabulary file',
            default='vocabulary.txt', type=str
        )
        parser.add_argument(
            '--label_file', help='path to labels file',
            default='labels.txt', type=str
        )
        parser.add_argument(
            "--mode", help="mode for creating ctf file",
            default=ONEHOT_MODE, type=str)

        args = vars(parser.parse_args(sys.argv[2:]))
        from buildctf import generate_CTF
        generate_CTF(
            args["mode"],
            args['dataset_file_path'],
            args["vocab_label_dir"],
            args["vocab_file"],
            args["label_file"]
        )
    else:
        print(
            'First argument must be "segment", "split", "process" or "ctf".',
            file=sys.stderr)
