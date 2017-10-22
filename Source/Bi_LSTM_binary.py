from __future__ import (absolute_import, division, print_function)

import os

import numpy as np
import cntk as C

import cntk.device
cntk.device.try_set_default_device(cntk.device.gpu(0))

import warnings
warnings.filterwarnings("ignore")

# dataset file path
vocab_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    '../Datasets/Dataset1/vocabulary_binary.txt')
label_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    '../Datasets/Dataset1/labels_binary.txt')
train_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    '../Datasets/Dataset1/train_binary.ctf')
test_file_path  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    '../Datasets/Dataset1/test_binary.ctf')
model_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    '../Trained/Bi_LSTM_binary.dnn')

# dataset dimensions
vocab_size = 5115
num_labels = 2

# model dimensions
input_dim  = vocab_size
label_dim  = num_labels
emb_dim    = 150
hidden_dim = 300

# Create the containers for input feature (x) and the label (y)
x = C.sequence.input_variable(input_dim, is_sparse=True)
y = C.input_variable(num_labels)

def BiRecurrence(fwd, bwd):
    F = C.layers.Recurrence(fwd)
    B = C.layers.Recurrence(bwd, go_backwards=True)
    x = C.placeholder()
    apply_x = C.splice(F(x), B(x))
    return apply_x

def create_model():
    '''
    Bi-LSTM model：
                                                label:   正面
                                                          ^
                                                          |
                                                      +-------+
                                                      | Dense |
                                                      +-------+
                                                          ^
                                                          |
                                                     +---------+
                                                     | Dropout |
                                                     +---------+
                                                          ^
                                                          |
          +------+   +------+   +------+   +------+   +------+
     0 -->| LSTM |<->| LSTM |<->| LSTM |<->| LSTM |<->| LSTM |
          +------+   +------+   +------+   +------+   +------+
              ^          ^          ^          ^          ^
              |          |          |          |          |
          +-------+  +-------+  +-------+  +-------+  +-------+
          | Embed |  | Embed |  | Embed |  | Embed |  | Embed |
          +-------+  +-------+  +-------+  +-------+  +-------+
              ^          ^          ^          ^          ^
              |          |          |          |          |
    word ---->+--------->+--------->+--------->+--------->+
            笔记本        用         的         很         流畅
    '''
    with C.layers.default_options(initial_state=0.1):
        return C.layers.Sequential([
            C.layers.Embedding(emb_dim, name='embed'),
            C.layers.BatchNormalization(),
            BiRecurrence(
                C.layers.LSTM(hidden_dim // 2),
                C.layers.LSTM(hidden_dim // 2)
            ),
            C.layers.BatchNormalization(),
            C.sequence.last,
            C.layers.Dropout(0.1),
            C.layers.Dense(num_labels, name='classify')
        ])

z = create_model()

def create_criterion_function(model, labels):
    ce   = C.cross_entropy_with_softmax(model, labels)
    errs = C.classification_error      (model, labels)
    return ce, errs # (model, labels) -> (loss, error metric)

def create_reader(path, is_training, randomize):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        sentence = C.io.StreamDef(field='S0', shape=vocab_size, is_sparse=True),
        label    = C.io.StreamDef(field='S1', shape=num_labels, is_sparse=True)
    )), randomize=randomize,
    max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)

def train(reader, model_func, max_epochs=10):

    # Instantiate the model function; x is the input (feature) variable
    model = model_func(x)

    # Instantiate the loss and error function
    loss, label_error = create_criterion_function(model, y)

    # training config
    epoch_size = 2239    # half of training dataset size
    minibatch_size = 50

    # LR schedule over epochs
    lr_per_sample = [3e-4]*4+[1.5e-4]
    lr_per_minibatch = [lr * minibatch_size for lr in lr_per_sample]
    lr_schedule = C.learning_rate_schedule(
        lr_per_minibatch,
        C.UnitType.minibatch,
        epoch_size
    )

    # Momentum schedule
    momentum_as_time_constant = C.momentum_as_time_constant_schedule(10)

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
        tag='Training', num_epochs=max_epochs
    )

    # Instantiate the trainer
    trainer = C.Trainer(model, (loss, label_error), learner, progress_printer)

    # process minibatches and perform model training
    C.logging.log_number_of_parameters(model)

    t = 0
    for epoch in range(max_epochs):        # loop over epochs
        epoch_end = (epoch+1) * epoch_size
        while t < epoch_end:               # loop over minibatches on the epoch
            # fetch minibatch
            data = reader.next_minibatch(minibatch_size, input_map={
                x: reader.streams.sentence,
                y: reader.streams.label
            })
            trainer.train_minibatch(data)
            t += data[y].num_samples
        trainer.summarize_training_progress()

def do_train():
    global z
    reader = create_reader(train_file_path, is_training=True, randomize=True)
    train(reader, z)

def evaluate(reader, model_func):

    # Instantiate the model function; x is the input (feature) variable
    model = model_func(x)

    # Create the loss and error functions
    loss, label_error = create_criterion_function(model, y)

    # process minibatches and perform evaluation
    progress_printer = C.logging.ProgressPrinter(tag='Evaluation', num_epochs=0)

    while True:
        minibatch_size = 1000
        # fetch minibatch
        data = reader.next_minibatch(minibatch_size, input_map={
            x: reader.streams.sentence,
            y: reader.streams.label
        })
        if not data:
            break
        evaluator = C.eval.Evaluator(label_error, progress_printer)
        evaluator.test_minibatch(data)
    evaluator.summarize_test_progress()

def do_test():
    reader = create_reader(test_file_path, is_training=False, randomize=False)
    evaluate(reader, z)

def show_samples(num=5):
    print('\nShowing {} samples:'.format(num))
    reader = create_reader(test_file_path, is_training=False, randomize=True)

    vocab = [line.rstrip('\n') for line in open(vocab_file_path)]
    label = [line.rstrip('\n') for line in open(label_file_path)]

    vocab_dict = {i: vocab[i] for i in range(vocab_size)}
    label_dict = {i: label[i] for i in range(label_dim)}

    for i in range(num):
        entry = reader.next_minibatch(1, input_map={
            x: reader.streams.sentence,
            y: reader.streams.label
        })
        while np.random.random() >= 0.5:
            entry = reader.next_minibatch(1, input_map={
                x: reader.streams.sentence,
                y: reader.streams.label
            })
        inp_seq = list(entry.values())[0].as_sequences(x)
        truth = list(entry.values())[1].as_sequences(y)[0]
        pred = z(x).eval({x: inp_seq})[0]
        pred_label  = label_dict[np.argmax(pred)]
        truth_label = label_dict[truth.argmax()]
        word_indices = inp_seq[0].argmax(axis=1).T.tolist()[0]
        sentence = ''.join(map(lambda i: vocab_dict[i], word_indices))
        print('\nInput sentence:', sentence)
        print('True label     :', truth_label)
        print('Predicted label:', pred_label)

if __name__ == '__main__':
    do_train()
    do_test()
    show_samples(10)
    z.save(model_file_path)
