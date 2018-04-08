import random
import os
import json
import cntk as C

from train import CTFDataManager, TrainManager, train_model
from models import LSTMClassificationWrapper, LSTMRegressionWrapper,\
    CNNClassificationWrapper, CNNRegressionWrapper


def search_lstm(data_manager, log_path, args):
    best_embedding = 0
    best_lstm = 0
    best_result = 0
    best_reducer = None
    all_result = []
    reducers = {"reduce_sum": C.sequence.reduce_sum}
    for embedding in range(1000, 1300, 100):
        for lstm in range(1000, 1300, 100):
            for reducer in reducers:
                j = {"embedding": embedding, "lstm": lstm, "reducer": reducer}
                new_log_path = os.path.join(
                    log_path, "{}_{}_{}".format(embedding, lstm, reducer))
                if not os.path.exists(new_log_path):
                    os.mkdir(new_log_path)
                wrapper = LSTMClassificationWrapper(
                    embedding, lstm, reducers[reducer],
                    data_manager.x_dim, data_manager.y_dim)
                print("trying Embedding: {}\tLSTM: {}\treducer: {}".format(
                    embedding, lstm, reducer))
                result = train_model(data_manager, wrapper,
                                     new_log_path, args, max_epochs=20)
                print("Embedding: {}\tLSTM: {}\t reducer: {}\tAccuracy: {:.2f}".format(
                    embedding, lstm, reducer, result))
                j["accuracy"] = result
                all_result.append(j)
                if result > best_result:
                    best_result = result
                    best_embedding = embedding
                    best_lstm = lstm
                    best_reducer = reducer
                with open(os.path.join(log_path, "result.json"), "w") as f:
                    json.dump(all_result, f)
    print("best result: ")
    print("Embedding: {}\tLSTM: {}\t reducer:{}\tAccuracy: {:.2f}".format(
        best_embedding, best_lstm, best_reducer, best_result))


def search_cnn(data_manager, log_path, args):
    try_conv_layers = 5
    try_conv_size_range = (2, 5)
    all_result = []
    reducers = {"reduce_max": C.sequence.reduce_max,
                "reduce_sum": C.sequence.reduce_sum}
    embedding = 1000
    for conv_layer in range(1, try_conv_layers):
        for conv_size in range(try_conv_size_range[0], try_conv_size_range[1]):
            for reducer in reducers:
                j = {"embedding": embedding,
                     "conv_layer": conv_layer,
                     "conv_size": conv_size,
                     "reducer": reducer}
                new_log_path = os.path.join(
                    log_path, "{}_{}_{}_{}".format(embedding, conv_layer,
                                                   conv_size, reducer))
                if not os.path.exists(new_log_path):
                    os.mkdir(new_log_path)
                wrapper = CNNClassificationWrapper(embedding, conv_layer,
                                                   conv_size, reducers[reducer],
                                                   data_manager.x_dim,
                                                   data_manager.y_dim)
                print("trying {}".format(j))
                result = train_model(data_manager, wrapper, new_log_path, args)
                print("{}\taccuracy: {:2f}".format(j, result))
                j["accuracy"] = result
                all_result.append(j)
                with open(os.path.join(log_path, "cnn.json"), "w") as f:
                    json.dump(all_result, f)


def search_model(data_manager, log_path, args):
    search_cnn(data_manager, log_path, args)
