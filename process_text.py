import os
import collections
import itertools

UNKNOWN_TOKEN = "UNKNOWN"
IGNORED_CHAR = set(["\t", "…", "“", "”"])


def process(input_dir, no_filter=False, unknown_threshold=0,
            replace_unknown=True, train_prefix="train",
            test_prefix="test", dev_prefix="dev"):
    filter_tag = "no_filter" if no_filter else "filter"
    replace_tag = "replace" if replace_unknown else "remove"
    dir_name = "{}_{}_{}".format(filter_tag, replace_tag, unknown_threshold)
    output_dir = os.path.join(input_dir, dir_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    print("[process] reading dataset...")
    train_set = read_txt(os.path.join(input_dir, train_prefix + ".txt"))
    dev_set = read_txt(os.path.join(input_dir, dev_prefix + ".txt"))
    test_set = read_txt(os.path.join(input_dir, test_prefix + ".txt"))
    labels = set(map(lambda x: x[1], train_set))
    if not no_filter:
        print("[process] filtering non-Chinese words...")
    train_set = process_filter(train_set, no_filter)
    dev_set = process_filter(dev_set, no_filter)
    test_set = process_filter(test_set, no_filter)
    all_words = itertools.chain.from_iterable(
        (words for (words, label) in train_set))
    counter = collections.Counter(all_words)
    known_vocabs = set(
        [word for word in counter if counter[word] > unknown_threshold])
    if replace_unknown:
        known_vocabs.add(UNKNOWN_TOKEN)
    print("[process] processing unknown token...")
    train_set = process_unknown(train_set, known_vocabs, replace_unknown)
    dev_set = process_unknown(dev_set, known_vocabs, replace_unknown)
    test_set = process_unknown(test_set, known_vocabs, replace_unknown)
    print("[process] writing to files...")
    write_to_file(train_set, train_prefix, output_dir)
    write_to_file(dev_set, dev_prefix, output_dir)
    write_to_file(test_set, test_prefix, output_dir)
    with open(os.path.join(output_dir, "vocabulary.txt"), "w") as f:
        for vocab in known_vocabs:
            f.write("{}\n".format(vocab))
    with open(os.path.join(output_dir, "labels.txt"), "w") as f:
        for label in labels:
            f.write("{}\n".format(label))


def read_txt(input_file):
    records = []
    with open(input_file) as f:
        for line in f:
            words, label = line[:-1].split('\t')
            records.append((words.split(), label))
    return records


def process_filter(dataset, no_filter):
    if not no_filter:
        filter_func = filter_non_Chinese
    else:
        def filter_func(x): return x
    dataset = ((filter_func(words), label) for (words, label) in dataset)
    dataset = post_filter(dataset, no_filter)
    return dataset


def filter_non_Chinese(words):
    return list(
        filter(
            lambda x: all(
                map(
                    lambda y: is_Chinese_char(y) and y not in IGNORED_CHAR,
                    x
                ),
            ),
            words
        )
    )


def post_filter(records, no_filter):
    if not no_filter:
        records = filter_empty(records)
    else:
        records = list(records)
    return records


def filter_empty(dataset):
    return list(
        filter(
            lambda x: x[0] and all(
                map(
                    lambda y: y and not y.isspace(), x[0])
            ), dataset)
    )


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


def process_unknown(dataset, known_vocabs, replace_unknown):
    if replace_unknown:
        dataset = replace_unknowns(dataset, known_vocabs)
    else:
        dataset = remove_unknowns(dataset, known_vocabs)
        dataset = filter_empty(dataset)
    return dataset


def replace_unknowns(dataset, known_vocabs):
    print("[build]replacing unknown token...")
    new_dataset = []
    for words, label in dataset:
        new_words = list(
            map(lambda word: word if word in known_vocabs else UNKNOWN_TOKEN, words)
        )
        new_dataset.append((new_words, label))
    return new_dataset


def remove_unknowns(dataset, known_vocabs):
    new_dataset = []
    for words, label in dataset:
        new_words = list(
            filter(lambda word: word in known_vocabs, words)
        )
        new_dataset.append((new_words, label))
    return new_dataset


def write_to_file(dataset, prefix, output_dir):
    with open(os.path.join(output_dir, prefix + ".txt"), "w") as f:
        for words, label in dataset:
            f.write("{}\t{}\n".format(" ".join(words), label))
