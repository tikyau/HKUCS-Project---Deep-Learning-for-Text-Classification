import csv
import multiprocessing

import jieba
from snownlp import SnowNLP


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


def segment(data_field, label_field, row):
    sentence = row[data_field].strip()
    if not sentence:
        return (sentence, [], row[label_field])
    sentence = sentence.replace("\n", "\\n").replace(
        "\t", "\\t").replace('\r', "\\r")
    words = jieba.cut(SnowNLP(sentence).han)
    valid_words = list(filter(lambda x: x and all(
        map(lambda y: not y.isspace(), x)), words))
    return (sentence, valid_words, row[label_field])
