from __future__ import (absolute_import, division, print_function)

import os
import sys
from datetime import datetime
import argparse
import inspect
import json

import cntk as C
from cntk.train.training_session import CrossValidationConfig,\
    training_session, CheckpointConfig, TestConfig
import cntk.device
from models import LSTMRegressionWrapper


class CTFDataManager(object):
    def __init__(self, **kwargs):
        input_dir = kwargs['input_dir']
        train_file_path = os.path.join(input_dir, kwargs['train_file_name'])
        test_file_path = os.path.join(input_dir, kwargs['test_file_name'])
        dev_file_path = os.path.join(input_dir, kwargs['dev_file_name'])
        vocab_file_path = os.path.join(input_dir, kwargs['vocab_file_name'])
        label_file_path = os.path.join(input_dir, kwargs['label_file_name'])
        self.x_dim = self._get_size(vocab_file_path)
        self.y_dim = self._get_size(label_file_path)
        self.train_size = self._get_size(train_file_path)
        self.x = C.sequence.input_variable(self.x_dim, is_sparse=True)
        self.y = C.input_variable(self.y_dim)

        streamDefs = C.io.StreamDefs(
            sentence=C.io.StreamDef(
                field='S0', shape=self.x_dim, is_sparse=True),
            label=C.io.StreamDef(
                field='S1', shape=self.y_dim, is_sparse=True)
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

    def _get_size(self, file_path):
        with open(file_path, 'r') as f:
            return len(f.readlines())


class TrainManager(object):
    def __init__(self, model_wrapper, data_manager, log_path,
                 minibatch_size=1024, max_epochs=20):
        self.epoch_size = data_manager.train_size
        self.minibatch_size = minibatch_size
        self.dev_reader = data_manager.dev_reader
        self.train_reader = data_manager.train_reader
        self.dev_map = data_manager.get_dev_map()
        self.train_map = data_manager.get_train_map()
        self.metric = model_wrapper.metric
        self.model = model_wrapper.model
        learner = self._create_learner()
        trainer = self._create_trainer(log_path, learner, max_epochs)
        self.session = self._create_session(log_path, trainer, max_epochs)

    def _create_trainer(self, log_path, learner, max_epochs):
        progress_printer = C.logging.ProgressPrinter(
            freq=100, tag='Training', num_epochs=max_epochs,
            test_freq=500
        )
        tensorboard_writer = C.logging.TensorBoardProgressWriter(
            freq=10, log_dir=log_path, model=self.model
        )
        trainer = C.Trainer(
            self.model, self.metric,
            learner, [progress_printer, tensorboard_writer]
        )
        return trainer

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
            lr=lr_schedule,
            momentum=momentum_as_time_constant,
            gradient_clipping_threshold_per_sample=10,
            gradient_clipping_with_truncation=True
        )
        return learner

    def _create_session(self, log_path, trainer, max_epochs):
        cv_config = CrossValidationConfig(
            minibatch_source=self.dev_reader,
            minibatch_size=self.minibatch_size,
            model_inputs_to_streams=self.dev_map,
            frequency=self.epoch_size // 2
        )
        checkpoint = os.path.join(log_path, "checkpoint")
        checkpoint_config = CheckpointConfig(
            checkpoint, frequency=self.epoch_size, restore=False)
        test_config = TestConfig(minibatch_source=self.dev_reader,
                                 minibatch_size=self.minibatch_size,
                                 model_inputs_to_streams=self.dev_map,
                                 criterion=self.metric.error)
        return training_session(
            trainer=trainer, mb_source=self.train_reader,
            mb_size=self.minibatch_size,
            model_inputs_to_streams=self.train_map,
            progress_frequency=self.epoch_size, cv_config=cv_config,
            max_samples=self.epoch_size * max_epochs,
            checkpoint_config=checkpoint_config, test_config=test_config
        )


def save_config(output_dir, log_path, wrapper, train_manager, args):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    wrapper_code = inspect.getsource(wrapper)
    train_manager_code = inspect.getsource(train_manager)
    parameter_code = inspect.getsource(get_model)
    with open(os.path.join(log_path, "config", "w"), "w") as f:
        f.write("hyperparameters:\n\n")
        f.write(parameter_code)
        f.write("model:\n\n")
        f.write(wrapper_code)
        f.write("training settings:\n\n")
        f.write(train_manager_code)
        f.write("arguments:\n\n")
        json.dump(args, f)


def get_log_path(output_dir, run_name):
    time_name = datetime.now().strftime("%h,%d_%H_%M")
    run_name = run_name + '_' if run_name != '' else ''
    log_path = os.path.join(output_dir, run_name + time_name)
    return log_path


def get_model():
    return LSTMRegressionWrapper(300, 1000)


def train_model(args, wrapper):
    output_dir = args['output_dir']
    run_name = args['run_name']
    C.cntk_py.set_fixed_random_seed(1)
    data_manager = CTFDataManager(**args)

    print('Vocabulary size :', data_manager.x_dim)
    print('Number of labels:', data_manager.y_dim)
    print("Training size:", data_manager.train_size)

    wrapper.bind(data_manager.x, data_manager.y)
    wrapper.generate_metric(data_manager.y)
    log_path = get_log_path(output_dir, run_name)

    train_manager = TrainManager(wrapper, data_manager, log_path)
    save_config(output_dir, log_path, wrapper, train_manager, args)
    train_manager.session.train()


def get_args():
    parser = argparse.ArgumentParser(description='Train and test.')

    parser.add_argument('input_dir', help='Directory containing the dataset')
    parser.add_argument('output_dir', help='Directory to store logs etc.')

    parser.add_argument(
        '--train_file_name', help='Name of training set CTF file',
        default='train.ctf'
    )
    parser.add_argument(
        '--test_file_name', help='Name of testing set CTF file',
        default='test.ctf'
    )
    parser.add_argument(
        '--dev_file_name', help='Name of dev set CTF file',
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
        '--run_name', help='Name of current run',
        default='run'
    )
    return vars(parser.parse_args(sys.argv[1:]))


def main():
    cntk.device.try_set_default_device(cntk.device.gpu(0))
    args = get_args()
    train_model(args, get_model())


if __name__ == '__main__':
    main()
