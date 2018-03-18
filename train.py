from __future__ import (absolute_import, division, print_function)

import os
import sys
import shutil
from datetime import datetime
import argparse

import cntk as C
from cntk.train.training_session import CrossValidationConfig,\
    training_session, CheckpointConfig, TestConfig
import cntk.device
from models import LSTMRegression


class CTFDataManager(object):
    def __init__(self, input_dir, **kwargs):
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


def create_learner(model, epoch_size, minibatch_size):
    lr_per_sample = [3e-5] * 10 + [1.5e-5] * 20 + [1e-5]
    lr_per_minibatch = [lr * minibatch_size for lr in lr_per_sample]
    lr_schedule = C.learning_rate_schedule(
        lr_per_minibatch,
        C.UnitType.minibatch,
        epoch_size
    )
    momentum_as_time_constant = C.momentum_as_time_constant_schedule(20)
    learner = C.adam(
        parameters=model.parameters,
        lr=lr_schedule,
        momentum=momentum_as_time_constant,
        gradient_clipping_threshold_per_sample=10,
        gradient_clipping_with_truncation=True
    )
    return learner


def save_script(output_dir, log_path):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    shutil.copy(sys.argv[0], os.path.join(log_path, sys.argv[0]))


def get_log_path(output_dir, run_name):
    time_name = datetime.now().strftime("%h,%d_%H_%M")
    run_name = run_name + '_' if run_name != '' else ''
    log_path = os.path.join(output_dir, run_name + time_name)
    return log_path


def create_train_session(model, metric, data_manager, log_path, max_epochs=20):
    epoch_size = data_manager.train_size
    MINIBATCH_SIZE = 1024
    checkpoint = os.path.join(log_path, "checkpoint")
    dev_reader = data_manager.dev_reader
    train_reader = data_manager.train_reader

    C.logging.log_number_of_parameters(model)

    progress_printer = C.logging.ProgressPrinter(
        freq=100, tag='Training', num_epochs=max_epochs,
        test_freq=500
    )
    tensorboard_writer = C.logging.TensorBoardProgressWriter(
        freq=10, log_dir=log_path, model=model
    )
    learner = create_learner(model, epoch_size, MINIBATCH_SIZE)
    trainer = C.Trainer(
        model, metric,
        learner, [progress_printer, tensorboard_writer]
    )

    cv_config = CrossValidationConfig(
        minibatch_source=dev_reader, minibatch_size=MINIBATCH_SIZE,
        model_inputs_to_streams=data_manager.get_dev_map(),
        frequency=epoch_size // 2
    )

    checkpoint_config = CheckpointConfig(
        checkpoint, frequency=epoch_size, restore=False)
    test_config = TestConfig(minibatch_source=dev_reader,
                             minibatch_size=MINIBATCH_SIZE,
                             model_inputs_to_streams=data_manager.get_dev_map(),
                             criterion=metric.error)

    return training_session(
        trainer=trainer, mb_source=train_reader, mb_size=MINIBATCH_SIZE,
        model_inputs_to_streams=data_manager.get_train_map(),
        progress_frequency=epoch_size, cv_config=cv_config,
        max_samples=epoch_size * max_epochs,
        checkpoint_config=checkpoint_config, test_config=test_config
    )


def train_model(args, builder):
    input_dir = args['input_dir']
    output_dir = args['output_dir']
    run_name = args['run_name']
    C.cntk_py.set_fixed_random_seed(1)
    data_manager = CTFDataManager(input_dir, args)

    print('Vocabulary size :', data_manager.x_dim)
    print('Number of labels:', data_manager.y_dim)
    print("Training size:", data_manager.train_size)

    model = builder.get_model()(data_manager.y)(data_manager.x)
    metric = builder.get_metric(data_manager.y)

    log_path = get_log_path(output_dir, run_name)

    save_script(output_dir, log_path)
    train_session = create_train_session(
        model, metric, data_manager,
        log_path)
    train_session.train()


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
    builder = LSTMRegression(300, 1000)
    train_model(args, builder)


if __name__ == '__main__':
    main()
