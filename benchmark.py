#!/usr/bin/env python3
1;4205;0cimport sys
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
    return sentences


def benchmark_snownlp(data_path):
    sentences = read_sentences(data_path)
    labels = set([i[1] for i in sentences])

    def scale(i):
        return math.floor(i * len(labels)) + 1
    confuse_matrix = [[0 for i in range(len(labels))]
                      for j in range(len(labels))]
    total = 0
    correct = 0
    for sentence, correct_label in sentences:
        total += 1
        sentiment = SnowNLP(sentence).sentiments
        predicted = scale(sentiment)
        correct += 1 if correct_label == predicted else 0
        try:
            confuse_matrix[correct_label - 1][predicted - 1] += 1
        except Exception as e:
            print(correct_label - 1, predicted - 1)
        print("{}/{} accuracy: {:.2f}".format(correct, total, correct / total))
    for i in range(len(labels)):
        print("\t".join((str(confuse_matrix[i][j])
                         for j in range(len(labels)))))
    accuracy = sum(confuse_matrix[i][i]
                   for i in range(len(labels))) / len(sentences)
    print("Accuracy {:.2f}".format(accuracy))


def benchmark_api(data_path):
    raise NotImplementedError()


def train_snownlp(pos, neg):
    from snownlp import sentiment
    sentiment.train(pos, neg)
    sentiment.save("movie_12345_even.marshal")


def main():
    if len(sys.argv) <= 1:
        print('First argument must be "model", "snownlp", or "api".',
              file=sys.stderr)
        sys.exit(0)
    parser = argparse.ArgumentParser(description="benchmark several approches")
    parser.add_argument("data_path", help="path to the target file")
    if sys.argv[1] == "snownlp":
        args = parser.parse_args(sys.argv[2:])
        benchmark_snownlp(args.data_path)
    elif sys.argv[1] == "model":
        parser.add_argument("model_path", help="path to the saved model")
        args = parser.parse_args(sys.argv[2:])
        benchmark_cntk(args.data_path, args.model_path)
    elif sys.argv[1] == "api":
        args = parser.parse_args(sys.argv[2:])
        benchmark_api(args.data_path)


if __name__ == "__main__":
    train_snownlp(sys.argv[1], sys.argv[2])
