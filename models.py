import collections


import cntk as C

Metric = collections.namedtuple("Metric", ["loss", "accuracy"])


def BiRecurrence(fwd, bwd):
    F = C.layers.Recurrence(fwd)
    B = C.layers.Recurrence(bwd, go_backwards=True)
    x = C.placeholder()
    apply_x = C.splice(F(x), B(x))
    return apply_x


class Classifier(object):
    def __init__(self):
        self.model = None
        self.metric = None

    def bind(self, x, y):
        self.model = self.model(x)
        loss = C.cross_entropy_with_softmax(self.model, y)
        accuracy = 1 - C.not_equal(C.argmax(self.model), C.argmax(y))
        self.metric = Metric(loss, accuracy)


class Regression(object):
    def __init__(self):
        self.model = None
        self.metric = None
        self.y_dim = 0

    def bind(self, x, y):
        self.model = self.model(x)
        loss = C.squared_error(self.model * 2.5, C.argmax(y) - 2)
        accuracy = 1 - C.not_equal(C.round(self.model), C.argmax(y) - 2)
        self.metric = Metric(loss, accuracy)


class LSTMRegressionWrapper(Regression):
    def __init__(self, embedding_dim, lstm_hidden_dim, reducer, x_dim, y_dim,
                 name="LSTM_linear_regression_tanh"):
        super().__init__()
        with C.layers.default_options(activation=C.tanh):
            self.model = C.layers.Sequential([
                C.layers.Embedding(embedding_dim, name='embed'),
                C.layers.Recurrence(
                    C.layers.LSTM(lstm_hidden_dim)
                ),
                reducer,
                C.layers.Dense(1, activation=C.tanh)
            ], name=name)
            self.y_dim = y_dim
            self.name = name
        self.metric = None


class LSTMClassificationWrapper(Classifier):
    def __init__(self, embedding_dim, lstm_hidden_dim, reducer, x_dim, y_dim,
                 name="LSTM_classification"):
        super().__init__()
        with C.layers.default_options(activation=C.tanh):
            self.model = C.layers.Sequential([
                C.layers.Embedding(embedding_dim, name='embed'),
                C.layers.Recurrence(
                    C.layers.LSTM(lstm_hidden_dim)
                ),
                reducer,
                C.layers.Dense((y_dim, ))
            ], name=name)
        self.name = name


class CNNClassificationWrapper(Classifier):
    def __init__(self, embedding_dim, conv_layers, x_dim, y_dim, name="CNN_classification"):
        super().__init__()
        with C.layers.default_options(activation=C.relu):
            self.model = C.layers.Sequential([
                C.layers.Embedding(embedding_dim),
                C.layers.For(range(conv_layers), lambda x: [
                    C.layers.Convolution(
                        (2, embedding_dim), embedding_dim,
                        sequential=True, reduction_rank=0,
                        pad=True, strides=(1, embedding_dim)
                    ),
                    C.ops.squeeze,
                    C.layers.BatchNormalization()
                ]),
                C.sequence.reduce_max,
                C.layers.Dense(50),
                C.layers.Dense(y_dim)
            ])
        self.metric = None
        self.name = name


class CNNRegressionWrapper(Regression):
    def __init__(self, embedding_dim, conv_layers, conv_words, x_dim, y_dim, reducer, name="CNN_regression"):
        super().__init__()
        with C.layers.default_options(activation=C.relu):
            self.model = C.layers.Sequential([
                C.layers.Embedding(embedding_dim),
                C.layers.For(range(conv_layers), lambda x: [
                    C.layers.Convolution(
                        (conv_words, embedding_dim), embedding_dim,
                        sequential=True, reduction_rank=0,
                        pad=True, strides=(1, embedding_dim)
                    ),
                    C.ops.squeeze,
                    C.layers.BatchNormalization()
                ]),
                reducer,
                C.layers.Dense(50),
                C.layers.Dense((1, ), activation=C.tanh)
            ], name=name)
        self.name = name
