import collections


import cntk as C

Metric = collections.namedtuple("Metric", ["loss", "accuracy"])


def BiRecurrence(fwd, bwd):
    F = C.layers.Recurrence(fwd)
    B = C.layers.Recurrence(bwd, go_backwards=True)
    x = C.placeholder()
    apply_x = C.splice(F(x), B(x))
    return apply_x


class LSTMRegressionWrapper(object):

    def __init__(self, embedding_dim, lstm_hidden_dim, x_dim, y_dim,
                 name="LSTM_linear_regression_tanh"):
        with C.layers.default_options(activation=C.tanh):
            self.model = C.layers.Sequential([
                C.layers.Embedding(embedding_dim, name='embed'),
                C.layers.Recurrence(
                    C.layers.LSTM(lstm_hidden_dim)
                ),
                C.sequence.last,
                C.layers.BatchNormalization(),
                C.layers.Dense(1, activation=C.tanh)
            ], name=name)
            self.y_dim = y_dim
            self.name = name
        self.metric = None

    def bind(self, x, y):
        self.model = self.model(x)
        loss = C.squared_error(self.model * 2.5, C.argmax(y) - 2)
        accuracy = 1 - C.not_equal(C.round(self.model), C.argmax(y) - 2)
        self.metric = Metric(loss, accuracy)


class LSTMClassificationWrapper(object):
    def __init__(self, embedding_dim, lstm_hidden_dim, x_dim, y_dim,
                 name="LSTM_classification"):
        with C.layers.default_options(activation=C.tanh):
            self.model = C.layers.Sequential([
                C.layers.Embedding(embedding_dim, name='embed'),
                C.layers.Stabilizer(),
                C.layers.Recurrence(
                    C.layers.LSTM(lstm_hidden_dim)
                ),
                C.sequence.last,
                C.layers.BatchNormalization(),
                C.layers.Dense((y_dim, ))
            ], name=name)
        self.metric = None
        self.name = name

    def bind(self, x, y):
        self.model = self.model(x)
        loss = C.cross_entropy_with_softmax(self.model, y)
        accuracy = 1 - C.not_equal(C.argmax(self.model), C.argmax(y))
        self.metric = Metric(loss, accuracy)


class CNNClassificationWrapper(object):
    def __init__(self, embedding_dim, x_dim, y_dim, name="CNN_classification"):
        with C.layers.default_options(activation=C.relu):
            self.model = C.layers.Sequential([
                C.layers.Embedding(300),
                C.layers.Convolution(
                    (1, 3), 20, sequential=True, reduction_rank=0),
                C.sequence.reduce_max,
                C.layers.BatchNormalization(),
                C.layers.Dropout(0.5),
                C.layers.Dense(50),
                C.layers.Dense(y_dim)
            ])
        self.metric = None
        self.name = name

    def bind(self, x, y):
        self.model = self.model(x)
        loss = C.cross_entropy_with_softmax(self.model, y)
        accuracy = 1 - C.not_equal(C.argmax(self.model), C.argmax(y))
        self.metric = Metric(loss, accuracy)


class GaussianClassificationWrapper(LSTMClassificationWrapper):
    def __init__(self, embedding_dim, lstm_hidden_dim,
                 x_dim, y_dim, name="clasification_with_gaussian"):
        super.__init__(embedding_dim, lstm_hidden_dim, x_dim, y_dim, name)

    def bind(self, x, y):
        self.model = self.model(x)
        loss = C.cross_entropy_with_softmax(self.model, y)
        accuracy = 1 - C.classification_error(self.model, y)
        self.metric = Metric(loss, accuracy)
