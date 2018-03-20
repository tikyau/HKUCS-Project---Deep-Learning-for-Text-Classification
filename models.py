import collections


import cntk as C

Metric = collections.namedtuple("Metric", ["loss", "error"])


def BiRecurrence(fwd, bwd):
    F = C.layers.Recurrence(fwd)
    B = C.layers.Recurrence(bwd, go_backwards=True)
    x = C.placeholder()
    apply_x = C.splice(F(x), B(x))
    return apply_x


class LSTMRegressionWrapper(object):

    def __init__(self, embedding_dim, lstm_hidden_dim, x_dim, y_dim,
                 name="LSTM_linear_regression_with_sigmoid"):
        with C.layers.default_options(activation=C.tanh):
            self.model = C.layers.Sequential([
                C.layers.Embedding(embedding_dim, name='embed'),
                C.layers.Stabilizer(),
                C.layers.Recurrence(
                    C.layers.LSTM(lstm_hidden_dim)
                ),
                C.sequence.last,
                C.layers.BatchNormalization(),
                C.layers.Dense(1, activation=C.sigmoid)
            ], name=name)
            self.y_dim = y_dim
            self.name = name
        self.metric = None

    def bind(self, x, y):
        self.model = self.model(x)
        loss = C.squared_error(self.model * self.y_dim, C.argmax(y) + 0.5)
        error = C.not_equal(C.floor(self.model * self.y_dim), C.argmax(y))
        self.metric = Metric(loss, error)


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
        error = C.classification_error(self.model, y)
        self.metric = Metric(loss, error)


class YAClassificationWrapper(object):
    def __init__(self, embedding_dim, lstm_hidden_dim, x_dim, y_dim,
                 name="yet_another_classifier"):
        self.model = C.layers.Sequential([
            C.layers.Embedding(embedding_dim),
            C.layers.Recurrence(
                C.layers.LSTM(hidden_dim)
            ),
            C.layers.Dropout(0.75),
            C.layers.Recurrence(
                C.layers.LSTM(hidden_dim)
            ),
            C.layers.BatchNormalization(),
            C.layers.Dense((y_dim, ))
        ])
        self.metric = None
        self.name

    def bind(self, x, y):
        self.model = self.model(x)
        loss = C.cross_entropy_with_softmax(self.model, y)
        error = C.classification_error(self.model, y)
        self.metric = Metric(loss, error)
