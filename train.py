#!/usr/bin/env python3
from __future__ import (absolute_import, division, print_function)
import os
import sys
from datetime import datetime
import signal
import argparse
import inspect
import json
import random

import cntk as C
import cntk.device
from models import LSTMClassificationWrapper, LSTMRegressionWrapper,\
    CNNClassificationWrapper
from buildctf import GAUSSIAN_MODE, SCALER_MODE, ONEHOT_MODE


def get_size(file_path):
    with open(file_path, 'r') as f:
        return len(f.readlines())


def get_y_dim(label_file_path, input_dir):
    with open(os.path.join(input_dir, "build.conf")) as f:
        j = json.load(f)
        return get_size(label_file_path) if j["mode"] != SCALER_MODE else 1


class CTFDataManager(object):
    def __init__(self, **kwargs):
        input_dir = kwargs['input_dir']
        train_file_path = os.path.join(input_dir, kwargs['train_file_name'])
        train_file_plain = os.path.join(input_dir, kwargs["train_file_plain"])
        test_file_path = os.path.join(input_dir, kwargs['test_file_name'])
        dev_file_path = os.path.join(input_dir, kwargs['dev_file_name'])
        vocab_file_path = os.path.join(input_dir, kwargs['vocab_file_name'])
        label_file_path = os.path.join(input_dir, kwargs['label_file_name'])
        self.x_dim = get_size(vocab_file_path)
        self.y_dim = get_y_dim(label_file_path, input_dir)
        self.train_size = get_size(train_file_plain)
        self.x = C.sequence.input_variable(self.x_dim, is_sparse=True)
        self.y = C.input_variable(self.y_dim)

        streamDefs = C.io.StreamDefs(
            sentence=C.io.StreamDef(
                field='S0', shape=self.x_dim, is_sparse=True),
            label=C.io.StreamDef(
                field='S1', shape=self.y_dim)
        )
        self.train_reader = C.io.MinibatchSource(
            C.io.CTFDeserializer(train_file_path,
                                 streamDefs),
            randomize=True,
            max_sweeps=C.io.INFINITELY_REPEAT)
        self.dev_reader = C.io.MinibatchSource(
            C.io.CTFDeserializer(dev_file_path,
                                 streamDefs),
            randomize=True, max_sweeps=1)

    def get_train_map(self):
        return self._get_input_map(self.train_reader)

    def get_dev_map(self):
        return self._get_input_map(self.dev_reader)

    def _get_input_map(self, reader):
        return {
            self.x: reader.streams.sentence,
            self.y: reader.streams.label
        }


class TrainManager(object):
    def __init__(self, model_wrapper, data_manager, log_path,
                 minibatch_size=1024, max_epochs=50):
        self.max_epochs = max_epochs
        self.log_path = log_path
        self.epoch_size = data_manager.train_size
        self.minibatch_size = minibatch_size
        self.dev_reader = data_manager.dev_reader
        self.train_reader = data_manager.train_reader
        self.dev_map = data_manager.get_dev_map()
        self.train_map = data_manager.get_train_map()
        self.metric = model_wrapper.metric
        self.model = model_wrapper.model
        self.x = data_manager.x
        self.y = data_manager.y
        learner = self._create_learner()
        self.loggers = self._create_loggers()
        self.trainer = C.Trainer(
            self.model, self.metric, learner, self.loggers)
        self.evaluator = C.eval.Evaluator(self.metric.accuracy, self.loggers)

    def _create_loggers(self):
        progress_printer = C.logging.ProgressPrinter(
            freq=100, tag='Training', num_epochs=self.max_epochs, test_freq=1000
        )
        tensorboard_writer = C.logging.TensorBoardProgressWriter(
            freq=10, log_dir=self.log_path, model=self.model
        )
        return [progress_printer, tensorboard_writer]

    def _create_learner(self):
        lr_per_sample = [3e-5] * 10 + [1.5e-5] * 20 + [1e-5]
        lr_per_minibatch = [lr * self.minibatch_size for lr in lr_per_sample]
        lr_schedule = C.learning_rate_schedule(
            lr_per_minibatch,
            C.UnitType.minibatch,
            self.epoch_size
        )
        momentum_as_time_constant = C.momentum_as_time_constant_schedule(20)
        learner = C.adam(
            parameters=self.model.parameters,
            lr=3e-4,
            momentum=momentum_as_time_constant,
            gradient_clipping_threshold_per_sample=0.21,
            gradient_clipping_with_truncation=True
        )
        return learner

    def train(self):
        checkpoint = os.path.join(self.log_path, "checkpoint")
        print("epoch sizes: ", self.epoch_size)
        best_result = 0
        for i in range(self.max_epochs):
            accumulated = 0
            while accumulated < self.epoch_size:
                self.trainer.train_minibatch(self.train_reader.next_minibatch(
                    self.minibatch_size, input_map=self.train_map))
                accumulated += self.minibatch_size
                if (accumulated // self.minibatch_size) % 5000 == 0:
                    self.evaluate()
            accuracy = self.evaluate()
            if accuracy > best_result:
                self.model.save("{}.best".format(checkpoint))
                best_result = accuracy
            self.model.save("{}.{}".format(checkpoint, i))
        self.trainer.summarize_training_progress()
        return best_result

    def evaluate(self):
        pos = self.dev_reader.current_position
        pos["minibatchSourcePosition"] = 0
        num_samples = 0
        correct_samples = 0
        self.dev_reader.current_position = pos
        batch = self.dev_reader.next_minibatch(
            self.minibatch_size, input_map=self.dev_map)
        while batch:
            num_samples += batch[self.x].num_samples
            avg_acc = self.evaluator.test_minibatch(batch)
            correct_samples += batch[self.x].num_samples * avg_acc
            batch = self.dev_reader.next_minibatch(
                self.minibatch_size, input_map=self.dev_map)
        self.evaluator.summarize_test_progress()
        return correct_samples / num_samples


def save_config(wrapper, train_manager, log_path, args):

    wrapper_code = inspect.getsource(wrapper.__class__)
    train_manager_code = inspect.getsource(train_manager.__class__)
    parameter_code = inspect.getsource(get_model)
    with open(os.path.join(log_path, "config"), "w") as f:
        f.write("hyperparameters:\n\n")
        f.write(parameter_code)
        f.write("model:\n\n")
        f.write(wrapper_code)
        f.write("training settings:\n\n")
        f.write(train_manager_code)
        f.write("arguments:\n\n")
        json.dump(args, f)


def get_log_path(input_dir, mode, output_dir, name, run_name):
    run_name = run_name or "{}_{}_{}".format(
        os.path.basename(os.path.normpath(input_dir)),
        mode, name)
    time_name = datetime.now().strftime("%h,%d_%H_%M")
    log_path = os.path.join(output_dir, "{}_{}".format(run_name, time_name))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    return log_path


def get_model(x_dim, y_dim):
    return LSTMClassificationWrapper(300, 500, x_dim=x_dim, y_dim=y_dim)


def train_model(data_manager, wrapper, log_path, args):

    wrapper.bind(data_manager.x, data_manager.y)
    setup_logger(log_path)
    train_manager = TrainManager(
        wrapper, data_manager, log_path)

    print('Vocabulary size :', data_manager.x_dim)
    print('Number of labels:', data_manager.y_dim)
    print("Training size:", data_manager.train_size)
    result = train_manager.train()
    save_config(wrapper, train_manager, log_path, args)
    return result


def get_args():
    parser = argparse.ArgumentParser(description='Train and test.')

    parser.add_argument('input_dir', help='Directory containing the dataset')
    parser.add_argument('output_dir', help='Directory to store logs etc.')
    parser.add_argument("mode", help="mode for ctf encoding")
    parser.add_argument(
        '--train_file_name', help='suffix of training set CTF file',
        default='train.ctf'
    )
    parser.add_argument(
        '--test_file_name', help='suffix of testing set CTF file',
        default='test.ctf'
    )
    parser.add_argument(
        '--dev_file_name', help='suffix of dev set CTF file',
        default="dev.ctf"
    )
    parser.add_argument(
        '--vocab_file_name', help='Name of vocabulary file',
        default='vocabulary.txt'
    )
    parser.add_argument(
        '--label_file_name', help='Name of label file',
        default='labels.txt'
    )
    parser.add_argument(
        '--train_file_plain', help='Name of training set plain file',
        default='train.txt'
    )
    parser.add_argument(
        '--run_name', help='Name of current run, default input_dir+model',
        default=''
    )
    parser.add_argument("--search", help="do random search",
                        default=False, type=bool)
    return vars(parser.parse_args(sys.argv[1:]))


def setup_logger(log_path):
    if signal.getsignal(signal.SIGHUP) != signal.SIG_DFL:
        sys.stdout = open(os.path.join(log_path, "run.log"), "a")
        sys.stderr = sys.stdout


def main():
    cntk.device.try_set_default_device(cntk.device.gpu(0))
    C.cntk_py.set_fixed_random_seed(1)
    args = get_args()
    args["dev_file_name"] = "{}_{}".format(args["mode"], args["dev_file_name"])
    args["test_file_name"] = "{}_{}".format(
        args["mode"], args["test_file_name"])
    args["train_file_name"] = "{}_{}".format(
        args["mode"], args["train_file_name"])
    data_manager = CTFDataManager(**args)
    log_path = get_log_path(
        args["input_dir"], args["mode"],
        args["output_dir"], "classification", args["run_name"])
    if not args["search"]:
        wrapper = get_model(data_manager.x_dim, data_manager.y_dim)
        train_model(data_manager, wrapper, log_path, args)
        return
    try_embedding = 10
    try_lstm = 10
    best_embedding = 0
    best_lstm = 0
    best_result = 0
    for i in range(try_embedding):
        for j in range(try_lstm):
            embedding = random.randrange(200, 1000, 20)
            lstm = random.randrange(200, 1000, 20)
            wrapper = LSTMClassificationWrapper(
                embedding, lstm, data_manager.x_dim, data_manager.y_dim)
            print("trying Embedding: {}\tLSTM: {}".format(embedding, lstm))
            result = train_model(data_manager, wrapper, log_path, args)
            print("Embedding: {}\tLSTM: {}\t Accuracy: {:.2f}".format(
                embedding, lstm, result))
            if result > best_result:
                best_result = result
                best_embedding = embedding
                best_lstm = lstm
    print("best result: ")
    print("Embedding: {}\tLSTM: {}\t Accuracy: {:.2f}".format(
        best_embedding, best_lstm, best_result))


if __name__ == '__main__':
    main()
