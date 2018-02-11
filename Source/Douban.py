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
import shutil
from datetime import datetime


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
            C.layers.Stabilizer(),
            BiRecurrence(
                C.layers.LSTM(parameters["hiddenDim"] // 2, activation=C.tanh),
                C.layers.LSTM(parameters["hiddenDim"] // 2, activation=C.tanh)),
            C.layers.BatchNormalization(),
            C.sequence.last,
            C.layers.Dense(y.shape, name='classify')
        ], name="BiLSTM_with_unk")


def create_criterion_function(model, labels):
    ce = C.cross_entropy_with_softmax(model, labels)
    errs = C.classification_error(model, labels)
    return ce, errs  # (model, labels) -> (loss, error metric)


def train_and_test(train_reader, dev_reader, model, x, y, output_dir, max_epochs=20):
    time_name = datetime.now().strftime("%h,%d_%H_%M")
    log_path = os.path.join(output_dir, time_name)
    chk_file = os.path.join(log_path, "checkpoint")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    shutil.copy(sys.argv[0], os.path.join(log_path, sys.argv[0]))
    # Instantiate the model function; x is the input (feature) variable
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
        freq=10, log_dir=log_path, model=model
    )

    # Instantiate the trainer
    trainer = C.Trainer(
        model, metric,
        learner, [progress_printer, tensorboard_writer]
    )

    # process minibatches and perform model training
    C.logging.log_number_of_parameters(model)

    def input_map(reader): return {
        x: reader.streams.sentence,
        y: reader.streams.label
    }

    cv_config = CrossValidationConfig(
        minibatch_source=dev_reader, minibatch_size=minibatch_size,
        model_inputs_to_streams=input_map(dev_reader),
        frequency=epoch_size
    )

    checkpoint_config = CheckpointConfig(
        chk_file, frequency=epoch_size, restore=False)
    test_config = TestConfig(minibatch_source=dev_reader,
                             minibatch_size=minibatch_size, model_inputs_to_streams=input_map(
                                 dev_reader),
                             criterion=metric[1])

    training_session(
        trainer=trainer, mb_source=train_reader, mb_size=minibatch_size,
        model_inputs_to_streams=input_map(train_reader),
        progress_frequency=epoch_size, cv_config=cv_config,
        max_samples=epoch_size * max_epochs, checkpoint_config=checkpoint_config, test_config=test_config
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

    # dataset file path
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    train_file_path = os.path.join(input_dir, "train.ctf")
    test_file_path = os.path.join(input_dir, "test.ctf")
    dev_file_path = os.path.join(input_dir, "dev.ctf")
    vocab_file_path = os.path.join(input_dir, "vocabulary.txt")
    label_file_path = os.path.join(input_dir, "labels.txt")
    plain_train_file_path = os.path.join(input_dir, "train.txt")

    # dataset dimensions
    xDim = get_size(vocab_file_path)  # xDim is the size of vocabulary
    yDim = get_size(label_file_path)  # yDim is the number of labels
    train_size = get_size(plain_train_file_path)
    C.cntk_py.set_fixed_random_seed(1)
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
    train_reader = C.io.MinibatchSource(C.io.CTFDeserializer(train_file_path,
                                                             streamDefs),
                                        randomize=True,
                                        max_sweeps=C.io.INFINITELY_REPEAT)
    dev_reader = C.io.MinibatchSource(C.io.CTFDeserializer(dev_file_path, streamDefs),
                                      randomize=False, max_sweeps=1)
    test_reader = C.io.MinibatchSource(C.io.CTFDeserializer(test_file_path,
                                                            streamDefs),
                                       randomize=False,
                                       max_sweeps=1)
    model = create_model(y)(x)
    print(model.embed.E.shape)
    print(model.classify.b.value)
    train(train_reader, dev_reader, model, x, y, output_dir)
    evaluate(test_reader, model, x, y)


if __name__ == '__main__':
    main()
