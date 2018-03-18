import collections


import cntk as C

Metric = collections.namedtuple("Metric", ["loss", "error"])


def BiRecurrence(fwd, bwd):
    F = C.layers.Recurrence(fwd)
    B = C.layers.Recurrence(bwd, go_backwards=True)
    x = C.placeholder()
    apply_x = C.splice(F(x), B(x))
    return apply_x


class LSTMRegression(object):

    def __init__(self, embedding_dim, lstm_hidden_dim,
                 name="LSTM_linear_regression", **kwargs):
        with C.layers.default_options(activation=C.tanh):
            self._model = C.layers.Sequential([
                C.layers.Embedding(embedding_dim, name='embed'),
                C.layers.Stabilizer(),
                C.layers.Recurrence(
                    C.layers.LSTM(lstm_hidden_dim)
                ),
                C.sequence.last,
                C.layers.BatchNormalization(),
                C.layers.Dense(1, activation=None, name='linear_reg')
            ], name=name)

    def get_model(self):
        return self._model

    def get_metric(self, labels):
        loss = C.not_equal(C.round(self._model), C.argmax(labels) + 1)
        error = C.squared_error(self._model, C.argmax(labels) + 1)
        return Metric(loss, error)
