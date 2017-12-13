from __future__ import (absolute_import, division, print_function)

import os

import numpy as np
import cntk.device
import warnings
import sys
import pyparsing
from datetime import datetime
from collections import Counter
import gensim
import itertools
import cntk as C
from cntk.train.training_session import CrossValidationConfig,\
    training_session, CheckpointConfig, TestConfig


def train_word2vec(sentences_matrix, index2Word,
                   num_features=300, min_word_count=1, context=10):
    """
    Trains, saves, loads Word2Vec model
    Returns initial weights for embedding layer.

    inputs:
    sentence_matrix # int matrix: num_sentences x max_sentence_len
    index2Word  # dict {int: str}
    num_features    # Word vector dimensionality
    min_word_count  # Minimum word count
    context         # Context window size
    """
    model_dir = 'w2vModels'
    model_name = "{:d}features_{:d}minwords_{:d}context".format(
        num_features, min_word_count, context)
    model_name = os.path.join(model_dir, model_name)
    if os.path.exists(model_name):
        embedding_model = gensim.models.Word2Vec.load(model_name)
        print('Load existing Word2Vec model {}'.format(model_name))
    else:
        # Set values for various parameters
        num_workers = 8  # Number of threads to run in parallel
        downsampling = 1e-3  # Downsample setting for frequent words

        # Initialize and train the model
        print('Training Word2Vec model...')
        embedding_model = gensim.models.Word2Vec(sentences_matrix, workers=num_workers,
                                                 size=num_features, min_count=min_word_count,
                                                 window=context, sample=downsampling)

        # If we don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        embedding_model.init_sims(replace=True)

        # Saving the model for later use. You can load it later using Word2Vec.load()
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        print('Saving Word2Vec model {}'.format(model_name))
        embedding_model.save(model_name)

    # add unknown words
    try:
        embedding_weights = {
            word: embedding_model[word] for word in index2Word.values()}
    except Exception:
        import pdb
        pdb.set_trace()
    return embedding_weights, embedding_model.vector_size


def load_data(base_dir):
    testFile = os.path.join(base_dir, 'test.txt')
    trainFile = os.path.join(base_dir, 'train.txt')
    x_test, y_test = read_data(testFile)
    x_train, y_train = read_data(trainFile)
    pad_sentence(x_test, x_train)
    index2Word, word2Index = build_vocab(x_train)

    def w2i(word):
        return word2Index[word] if word in word2Index else -1

    y_test = np.array(y_test)
    y_train = np.array(y_train)
    return {
        "x_test": np.array(x_test),
        "y_test": y_test,
        "x_train": np.array(x_train),
        "y_train": y_train,
        "word2Index": word2Index,
        "index2Word": index2Word
    }


def read_data(path):
    print("reading data from", path)
    with open(path, 'r') as f:
        data = (i[:-1].split('\t') for i in f.readlines())
    x = [i[0].split(' ') for i in data]
    print(x[0])
    y = [i[1] for i in data]
    labeller = {}
    uniqLabel = set(y)
    for index, label in enumerate(uniqLabel):
        labeller[label] = [0 for i in range(len(uniqLabel))]
        labeller[label][index] = 1
    y = [uniqLabel[i] for i in y]
    return x, y


def pad_sentence(x_test, x_train):
    sequenceLength = max(map(lambda x: len(x), x_train))
    print("padding sentence to", sequenceLength)
    PADDING_WORD = '<pad>'
    total_training_size = len(x_train)
    total_testing_size = len(x_test)
    for i, sentence in enumerate(x_train):
        sentence.extend([PADDING_WORD for i in range(
            sequenceLength - len(sentence))])
        if i % 100 == 0:
            print("padding training set {} / {}".format(i, total_training_size))
    for i, sentence in enumerate(x_test):
        if len(sentence) < sequenceLength:
            sentence.extend([PADDING_WORD for i in range(
                sequenceLength - len(sentence))])
        else:
            sentence = sentence[:sequenceLength]
        if i % 100 == 0:
            print("padding test set {} / {}".format(i, total_testing_size))


def build_vocab(x_train):
    print("Building vocabulary")
    word_counts = Counter(itertools.chain(*x_train))
    index2Word = {x[0]: x[1][0] for x in enumerate(word_counts.most_common())}
    # Mapping from word to index
    word2Index = {x: i for i, x in index2Word.items()}
    return index2Word, word2Index


def get_size(file_path):
    with open(file_path, 'r') as f:
        return len(f.readlines())


def BiRecurrence(fwd, bwd):
    F = C.layers.Recurrence(fwd)
    B = C.layers.Recurrence(bwd, go_backwards=True)
    x = C.placeholder()
    apply_x = C.splice(F(x), B(x))
    return apply_x


def create_model(y):
    parameters = {
        "embDim": 600,
        "hiddenDim": 300,
        "numFilters": 10,
        "filterShape": 5,

    }
    with C.layers.default_options(initial_state=0.1):
        return C.layers.Sequential([
            C.layers.Convolution1D(
                filter_shape=parameters["filterShape"],
                pad=True, num_filters=parameters["numFilters"],
                activation=C.relu),
            C.layers.MaxPooling(filter_shape=parameters["filterShape"]),
            C.layers.Dense(100),
            C.layers.Dense(y.shape, name='classify')
        ], name="CNN")


def create_criterion_function(model, labels):
    ce = C.cross_entropy_with_softmax(model, labels)
    errs = C.classification_error(model, labels)
    return ce, errs  # (model, labels) -> (loss, error metric)


def train(trainReader, testReader, model, x, y, max_epochs=20):

    # Instantiate the model function; x is the input (feature) variable
    logPath = "log/" + model.name + '/' + datetime.now().strftime("%h,%d_%H_%M")
    savePath = "model/" + model.name
    # Instantiate the loss and error function
    metric = create_criterion_function(model, y)
    # training config
    epoch_size = 180000 // 2    # half of training dataset size
    minibatch_size = 512

    # LR schedule over epochs
    lr_per_sample = [6e-5] * 4 + [3e-5] * 20 + [1e-5]
    lr_per_minibatch = [lr * minibatch_size for lr in lr_per_sample]
    lr_schedule = C.learning_rate_schedule(
        lr_per_minibatch,
        C.UnitType.minibatch,
        epoch_size
    )

    # Momentum schedule
    momentum_as_time_constant = C.momentum_as_time_constant_schedule(50)

    # Adam optimizer
    learner = C.adam(
        parameters=model.parameters,
        lr=lr_schedule,
        momentum=momentum_as_time_constant,
        gradient_clipping_threshold_per_sample=5,
        gradient_clipping_with_truncation=True
    )

    # Setup the progress updater
    progress_printer = C.logging.ProgressPrinter(
        freq=100, tag='Training', num_epochs=max_epochs,
        test_freq=500
    )
    tensorboard_writer = C.logging.TensorBoardProgressWriter(
        freq=10, log_dir=logPath, model=model
    )

    # Instantiate the trainer
    trainer = C.Trainer(
        model, metric,
        learner, [progress_printer, tensorboard_writer]
    )

    # process minibatches and perform model training
    C.logging.log_number_of_parameters(model)

    def inputMap(reader): return {
        x: reader.streams.sentence,
        y: reader.streams.label
    }

    cvConfig = CrossValidationConfig(
        minibatch_source=testReader, minibatch_size=minibatch_size,
        model_inputs_to_streams=inputMap(testReader),
        frequency=epoch_size
    )

    checkpointConfig = CheckpointConfig(
        savePath, frequency=epoch_size, restore=False)

    training_session(
        trainer=trainer, mb_source=trainReader, mb_size=minibatch_size,
        model_inputs_to_streams=inputMap(trainReader),
        progress_frequency=epoch_size, cv_config=cvConfig,
        max_samples=epoch_size * max_epochs, checkpoint_config=checkpointConfig
    ).train()


def evaluate(reader, model, x, y):

    # Instantiate the model function; x is the input (feature) variable
    # Create the loss and error functions
    loss, classification_error = create_criterion_function(model, y)
    # process minibatches and perform evaluation
    progress_printer = C.logging.ProgressPrinter(
        tag='Evaluation', num_epochs=100, freq=100, first=10)
    evaluator = C.eval.Evaluator(classification_error, progress_printer)
    while True:
        minibatch_size = 100
        # fetch minibatch
        data = reader.next_minibatch(minibatch_size, input_map={
            x: reader.streams.sentence,
            y: reader.streams.label
        })
        if not data:
            break
        evaluator.test_minibatch(data)
    evaluator.summarize_test_progress()


def main():
    cntk.device.try_set_default_device(cntk.device.gpu(0))
    warnings.filterwarnings("ignore")
    C.cntk_py.set_fixed_random_seed(1)
    np.random.seed(1)

    # dataset file path
    PATH_TO_DATASET = sys.argv[1]
    print("loading data")
    input_data = load_data(PATH_TO_DATASET)
    word_vectors, vector_size = train_word2vec(
        input_data["x_train"], input_data["index2Word"])

    def w2v(wordIndex):
        return word_vectors[wordIndex] if wordIndex in word_vectors else np.random.uniform(-0.25, 0.25,
                                                                                           vector_size)
    # input_data["x_train"] = np.stack([np.stack([w2v(wordIndex)
    #                                             for wordIndex in sentence]) for sentence in input_data["x_train"]])
    # input_data["x_test"] = np.stack([np.stack([w2v(wordIndex)
    #                                            for wordIndex in sentence]) for sentence in input_data["x_test"]])
    x_test = np.stack([np.stack([w2v(wordIndex)
                                 for wordIndex in sentence]) for sentence in input_data["x_train"]])
    print(x_test.shape)
    x_train = input_data["x_train"]
    y_test = input_data["y_test"]
    y_train = input_data["y_train"]
    input_dim = x_test.shape
    # dataset dimensions
    print('Input size :', input_dim)

    # Create the containers for input feature (x) and the label (y)
    x = C.sequence.input_variable(input_dim, is_sparse=False)
    y = C.input_variable(yDim)

    trainReader = C.io.MinibatchSourceFromData({'x': x_train, 'y': y_train})
    testReader = C.io.MinibatchSourceFromData({'x': x_test, 'y': y_test})
    model = create_model(y)(x)
    print(model.embed.E.shape)
    print(model.classify.b.value)
    train(trainReader, testReader, model, x, y)


if __name__ == '__main__':
    main()
