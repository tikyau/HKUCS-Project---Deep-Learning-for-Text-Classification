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
                 name="LSTM_linear_regression"):
        with C.layers.default_options(activation=C.tanh):
            self.model = C.layers.Sequential([
                C.layers.Embedding(embedding_dim, name='embed'),
                C.layers.Stabilizer(),
                C.layers.Recurrence(
                    C.layers.LSTM(lstm_hidden_dim)
                ),
                C.sequence.last,
                C.layers.BatchNormalization(),
                C.layers.Dense(1, activation=None, name='linear_reg')
            ], name=name)
        self.metric = None

    def bind(self, x, y):
        self.model = self.model(x)
        loss = C.not_equal(C.round(self.model), C.argmax(y) + 1)
        error = C.squared_error(self.model, C.argmax(y) + 1)
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
                C.layers.Dense(y_dim, name='classifier')
            ], name=name)
        self.metric = None

    def bind(self, x, y):
        self.model = self.model(x)
        loss = C.cross_entropy_with_softmax(self.model, y)
        error = C.classification_error(self.model, y)
        self.metric = Metric(loss, error)
