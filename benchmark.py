#!/usr/bin/env python3
import sys
import math
import argparse
import os
import random
import requests


def get_size(file_path):
    with open(file_path, 'r') as f:
        return len(f.readlines())


def benchmark_cntk(data_path, model_path):
    import cntk as C
    model = C.load_model(model_path)
    vocab_file_path = os.path.join(data_path, "vocabulary.txt")
    label_file_path = os.path.join(data_path, "labels.txt")
    dev_file_path = os.path.join(data_path, "dev.ctf")
    minibatch_size = 1024
    x_dim = get_size(vocab_file_path)
    y_dim = get_size(label_file_path)
    x = C.sequence.input_variable(x_dim, is_sparse=True)
    model = model(x)
    y = C.input_variable(y_dim, is_sparse=True)
    streamDefs = C.io.StreamDefs(
        sentence=C.io.StreamDef(
            field='S0', shape=x_dim, is_sparse=True),
        label=C.io.StreamDef(
            field='S1', shape=y_dim, is_sparse=True)
    )
    dev_reader = C.io.MinibatchSource(
        C.io.CTFDeserializer(dev_file_path,
                             streamDefs),
        randomize=True, max_sweeps=1)
    error = C.classification_error(model, y)
    progress_printer = C.logging.ProgressPrinter(
        freq=100, tag='evaluate', test_freq=1000
    )
    evaluator = C.eval.Evaluator(error, [progress_printer])
    input_map = {x: dev_reader.streams.sentence, y: dev_reader.streams.label}

    data = dev_reader.next_minibatch(minibatch_size, input_map=input_map)
    while data:
        evaluator.test_minibatch(data)
        data = dev_reader.next_minibatch(minibatch_size, input_map=input_map)
    evaluator.summarize_test_progress()


def read_sentences(data_path):
    sentences = []
    with open(data_path) as f:
        for line in f:
            sentence, label = line[:-1].split('\t')
            sentences.append((sentence, int(label)))
    return sentences


def benchmark_snownlp(data_path):
    from snownlp import SnowNLP
    sentences = read_sentences(data_path)
    labels = set([i[1] for i in sentences])

    def scale(i):
        return 0 if i < 0.5 else 1
    confuse_matrix = [[0 for i in range(len(labels))]
                      for j in range(len(labels))]
    total = 0
    correct = 0
    for sentence, correct_label in sentences:
        total += 1
        sentiment = SnowNLP(sentence).sentiments
        predicted = scale(sentiment)
        print("{}\t{}\t{}".format(sentence, predicted, correct_label))
        correct += 1 if correct_label == predicted else 0
        try:
            confuse_matrix[correct_label - 1][predicted - 1] += 1
        except Exception:
            print(correct_label - 1, predicted - 1)
        print("{}/{} accuracy: {:.2f}".format(correct, total, correct / total))
    for i in range(len(labels)):
        print("\t".join((str(confuse_matrix[i][j])
                         for j in range(len(labels)))))
    accuracy = sum(confuse_matrix[i][i]
                   for i in range(len(labels))) / len(sentences)
    print("Accuracy {:.2f}".format(accuracy))


def benchmark_api(data_path):
    MICROSOFT_KEY = "241d007895ea46a48b50edeb65be3ed1"
    REQUEST_URL = "https://westcentralus.api.cognitive.microsoft.com/text/analytics/v2.0/" + "sentiment"
    sentences = read_sentences(data_path)
    labels = set([i[1] for i in sentences])
    random.shuffle(sentences)

    headers = {
        "Ocp-Apim-Subscription-Key": MICROSOFT_KEY
    }
    total = 0
    total_correct = 0
    for i in range(min(len(sentences) // 10, 450)):
        part = sentences[i: i + 10]
        document = {"documents": toDoc(part)}
        target = {str(j + 1): part[j][1] for j in range(len(part))}
        try:
            response = requests.post(
                REQUEST_URL, headers=headers, json=document)
            sentiments = response.json()
            for doc in sentiments["documents"]:
                if match(doc, target):
                    total_correct += 1
            total += 10
            print("{}\t{}\t{:2f}".format(
                total_correct, total, total_correct / total))
        except Exception as e:
            print("ERROR!", e)


def match(doc, target):
    def scale(i):
        return i // 0.2
    predicted = doc["score"]
    expected = target[doc["id"]]
    return expected == scale(predicted)


def toDoc(sentences):
    result = []
    for i in range(len(sentences)):
        result.append({
            "id": str(i + 1),
            "language": "zh-Hans",
            "text": sentences[0]
        })
    return result


def train_snownlp(pos, neg):
    from snownlp import sentiment
    sentiment.train(pos, neg)
    sentiment.save("movie_even.marshal")


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
    main()
