import os
import sys
import math
import argparse

import cntk as C
from snownlp import SnowNLP


def benchmark_cntk(data_path, model_path):
    model = C.load_model(model_path)
    raise NotImplementedError()


def read_sentences(data_path):
    sentences = []
    with open(data_path) as f:
        for line in f:
            sentence, label = line[:-1].split('\t')
            sentences.append((sentence, int(label)))


def benchmark_snownlp(data_path):
    sentences = read_sentences(data_path)
    labels = set([i[1] for i in sentences])
    min_label = min(labels)
    max_label = max(labels)

    def scale(i):
        return math.floor(i * len(labels)) + 1
    confuse_matrix = [[0 for i in range(len(labels))]
                      for j in range(len(labels))]
    total = 0
    correct = 0
    for sentence, correct_label in sentences:
        total += 1
        predicted = scale(SnowNLP(sentence).sentiments)
        correct += 1 if correct_label == predicted else 0
        confuse_matrix[correct_label - 1][predicted - 1] += 1
        print("{}/{} accuracy: {:.2f}".format(correct, total, correct / total))
    for i in range(len(labels)):
        print("\t".join((confuse_matrix[i][j]
                         for j in range(len(labels)))))
    accuracy = sum(confuse_matrix[i][i]
                   for i in range(len(labels))) / len(sentences)
    print("Accuracy {:.2f}".format(accuracy))


def benchmark_api(data_path):
    raise NotImplementedError()


def main():
    if len(sys.argv) <= 1:
        print('First argument must be "model", "snownlp", or "api".',
              file=sys.stderr)
        sys.exit(0)
    parser = argparse.ArgumentParser(description="benchmark several approches")
    parser.add_argument("data_path", help="path to the target file")
    if sys.argv[1] == "snownlp":
        parser.parse_args(sys.argv[2:])
        benchmark_snownlp(parser.data_path)
    elif sys.argv[1] == "model":
        parser.add_argument("model_path", help="path to the saved model")
        parser.parse_args(sys.argv[2:])
        benchmark_cntk(parser.data_path, parser.model_path)
    elif sys.argv[1] == "api":
        parser.parse_args(sys.argv[2:])
        benchmark_api(parser.data_path)


if __name__ == "__main__":
    main()
