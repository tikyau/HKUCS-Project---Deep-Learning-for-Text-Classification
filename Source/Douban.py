from __future__ import (absolute_import, division, print_function)

import os

import numpy as np
import cntk as C
from cntk.train.training_session import CrossValidationConfig,\
    training_session, CheckpointConfig, TestConfig
import cntk.device
import warnings
import sys
import pyparsing


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
        "hiddenDim": 300
    }
    with C.layers.default_options(initial_state=0.1):
        return C.layers.Sequential([
            C.layers.Embedding(parameters["embDim"], name='embed'),
            C.layers.BatchNormalization(),
            # BiRecurrence(
            #     C.layers.LSTM(parameters["hiddenDim"] // 2),
            #     C.layers.LSTM(parameters["hiddenDim"] // 2),
            # ),
            C.layers.Recurrence(C.layers.GRU(parameters["hiddenDim"] // 2)),
            C.layers.BatchNormalization(),
            C.sequence.last,
            C.layers.Dropout(0.3),
            C.layers.Dense(y.shape, name='classify')
        ], name="GRU")


def create_criterion_function(model, labels):
    ce = C.cross_entropy_with_softmax(model, labels)
    errs = C.classification_error(model, labels)
    return ce, errs  # (model, labels) -> (loss, error metric)


def train(trainReader, testReader, model, x, y, max_epochs=20):

    # Instantiate the model function; x is the input (feature) variable
    logPath = "log/" + model.name
    savePath = "model/" + model.name
    # Instantiate the loss and error function
    metric = create_criterion_function(model, y)
    # training config
    epoch_size = 180000 // 2    # half of training dataset size
    minibatch_size = 512

    # LR schedule over epochs
    lr_per_sample = [3e-4] * 4 + [1.5e-4] * 20 + [1e-4]
    lr_per_minibatch = [lr * minibatch_size for lr in lr_per_sample]
    lr_schedule = C.learning_rate_schedule(
        lr_per_minibatch,
        C.UnitType.minibatch,
        epoch_size
    )

    # Momentum schedule
    momentum_as_time_constant = C.momentum_as_time_constant_schedule(100)

    # Adam optimizer
    learner = C.adam(
        parameters=model.parameters,
        lr=lr_schedule,
        momentum=momentum_as_time_constant,
        gradient_clipping_threshold_per_sample=15,
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

    training_session(
        trainer=trainer, mb_source=trainReader, mb_size=minibatch_size,
        model_inputs_to_streams=inputMap(trainReader),
        progress_frequency=epoch_size, cv_config=cvConfig,
        max_samples=epoch_size * max_epochs
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


# def show_samples(reader, model, x, num=5):
#     print('\nShowing {} samples:'.format(num))

#     vocab = [line.rstrip('\n') for line in open(vocab_file_path)]
#     label = [line.rstrip('\n') for line in open(label_file_path)]

#     vocab_dict = {i: vocab[i] for i in range(vocab_size)}
#     label_dict = {i: label[i] for i in range(label_dim)}

#     for i in range(num):
#         entry = reader.next_minibatch(1, input_map={
#             x: reader.streams.sentence,
#             y: reader.streams.label
#         })
#         while np.random.random() >= 0.5:
#             entry = reader.next_minibatch(1, input_map={
#                 x: reader.streams.sentence,
#                 y: reader.streams.label
#             })
#         inp_seq = list(entry.values())[0].as_sequences(x)
#         truth = list(entry.values())[1].as_sequences(y)[0]
#         pred = model.eval({x: inp_seq})[0]
#         pred_label = label_dict[np.argmax(pred)]
#         truth_label = label_dict[truth.argmax()]
#         word_indices = inp_seq[0].argmax(axis=1).T.tolist()[0]
#         sentence = ''.join(map(lambda i: vocab_dict[i], word_indices))
#         print('\nInput sentence:', sentence)
#         print('True label     :', truth_label)
#         print('Predicted label:', pred_label)


def main():
    cntk.device.try_set_default_device(cntk.device.gpu(0))
    warnings.filterwarnings("ignore")

    # dataset file path
    PATH_TO_DATASET = sys.argv[1]
    train_file_path = os.path.join(PATH_TO_DATASET, "train.ctf")
    test_file_path = os.path.join(PATH_TO_DATASET, "test.ctf")
    vocab_file_path = os.path.join(PATH_TO_DATASET, "vocabulary.txt")
    label_file_path = os.path.join(PATH_TO_DATASET, "labels.txt")
    model_file_path = 'DB/binary/Bi_LSTM_Douban.dnn'
    plain_train_file_path = os.path.join(PATH_TO_DATASET, "train.txt")

    # dataset dimensions
    xDim = get_size(vocab_file_path)  # xDim is the size of vocabulary
    yDim = get_size(label_file_path)  # yDim is the number of labels
    train_size = get_size(plain_train_file_path)

    print('Vocabulary size :', xDim)
    print('Number of labels:', yDim)
    print("Training size:", train_size)

    # Create the containers for input feature (x) and the label (y)
    x = C.sequence.input_variable(xDim, is_sparse=True)
    y = C.input_variable(yDim)

    streamDefs = C.io.StreamDefs(
        sentence=C.io.StreamDef(
            field='S0', shape=xDim, is_sparse=True),
        label=C.io.StreamDef(
            field='S1', shape=yDim, is_sparse=True)
    )
    trainReader = C.io.MinibatchSource(C.io.CTFDeserializer(train_file_path,
                                                            streamDefs),
                                       randomize=True,
                                       max_sweeps=C.io.INFINITELY_REPEAT)
    testReader = C.io.MinibatchSource(C.io.CTFDeserializer(test_file_path,
                                                           streamDefs),
                                      randomize=False,
                                      max_sweeps=1)
    model = create_model(y)(x)
    print(model.embed.E.shape)
    print(model.classify.b.value)
    train(trainReader, testReader, model, x, y)


if __name__ == '__main__':
    main()
