import random
import os
import json
import cntk as C

from train import CTFDataManager, TrainManager, train_model
from models import LSTMClassificationWrapper, LSTMRegressionWrapper,\
    CNNClassificationWrapper, CNNRegressionWrapper


def search_lstm(data_manager, log_path, args):
    try_embedding = 10
    try_lstm = 10
    best_embedding = 0
    best_lstm = 0
    best_result = 0
    best_reducer = None
    all_result = []
    reducers = {"reduce_max": C.sequence.reduce_max,
                "reduce_sum": C.sequence.reduce_sum,
                "last": C.sequence.last}
    for _ in range(try_embedding):
        for _ in range(try_lstm):
            for reducer in reducers:
                embedding = random.randrange(200, 1000, 20)
                lstm = random.randrange(200, 1000, 20)
                j = {"embedding": embedding, "lstm": lstm, "reducer": reducer}
                new_log_path = os.path.join(
                    log_path, "{}_{}_{}".format(embedding, lstm, reducer))
                if not new_log_path:
                    os.mkdir(new_log_path)
                wrapper = LSTMClassificationWrapper(
                    embedding, lstm, reducers["reducer"],
                    data_manager.x_dim, data_manager.y_dim)
                print("trying Embedding: {}\tLSTM: {}\treducer: {}".format(
                    embedding, lstm, reducer))
                result = train_model(data_manager, wrapper, log_path, args)
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


def search_model(data_manager, log_path, args):
    search_lstm(data_manager, log_path, args)
