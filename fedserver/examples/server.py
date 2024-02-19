import ast
import pathlib

import flwr as fl
import ai4flwr.auth.bearer

FEDERATED_ROUNDS = 2
FEDERATED_METRIC = "accuracy"
FEDERATED_MIN_CLIENTS = 2
FEDERATED_STRATEGY = "Federated Averaging"
FEDERATED_TOKENS = "1234"

# Weighted average of the metric:
def wavg_metric(metrics):
    global FEDERATED_METRIC
    list_metrics = []
    try:
        list_metrics = ast.literal_eval(FEDERATED_METRIC)
    except ValueError:
        print("Only one metric has been entered.")
    if len(list_metrics) == 0:
        n = sum([i for i, _ in metrics])
        wavg_metric = sum([i * metric[FEDERATED_METRIC] / n for i, metric in metrics])
        return {FEDERATED_METRIC: wavg_metric}
    else:
        n = sum([i for i, _ in metrics])
        dict_metrics = {}
        for fed_metric in list_metrics:
            wavg_metric = sum([i * metric[fed_metric] / n for i, metric in metrics])
            dict_metrics[fed_metric] = wavg_metric
        return dict_metrics


if FEDERATED_STRATEGY == "Federated Averaging" or FEDERATED_STRATEGY is None:
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=FEDERATED_MIN_CLIENTS,
        min_fit_clients=FEDERATED_MIN_CLIENTS,
        evaluate_metrics_aggregation_fn=wavg_metric,
    )
elif FEDERATED_STRATEGY == "FedProx Strategy":
    strategy = fl.server.strategy.FedProx(
        min_available_clients=FEDERATED_MIN_CLIENTS,
        min_fit_clients=FEDERATED_MIN_CLIENTS,
        evaluate_metrics_aggregation_fn=wavg_metric,
    )
elif FEDERATED_STRATEGY == "Federated Optim Strategy":
    strategy = fl.server.strategy.FedOpt(
        min_available_clients=FEDERATED_MIN_CLIENTS,
        min_fit_clients=FEDERATED_MIN_CLIENTS,
        evaluate_metrics_aggregation_fn=wavg_metric,
    )
elif FEDERATED_STRATEGY == "Federated Optimization with Adam":
    strategy = fl.server.strategy.FedAdam(
        min_available_clients=FEDERATED_MIN_CLIENTS,
        min_fit_clients=FEDERATED_MIN_CLIENTS,
        evaluate_metrics_aggregation_fn=wavg_metric,
    )
elif FEDERATED_STRATEGY == "Adaptive Federated Optimization using Yogi":
    strategy = fl.server.strategy.FedYogi(
        min_available_clients=FEDERATED_MIN_CLIENTS,
        min_fit_clients=FEDERATED_MIN_CLIENTS,
        evaluate_metrics_aggregation_fn=wavg_metric,
    )

# Include token interceptor
token_interceptor = ai4flwr.auth.bearer.BearerTokenInterceptor(FEDERATED_TOKENS)
# Flower server:
fl.server.start_server(
    server_address="0.0.0.0:5000",
    config=fl.server.ServerConfig(num_rounds=FEDERATED_ROUNDS),
    strategy=strategy,
    certificates=(
        pathlib.Path(
            "fedserver", "examples", ".cache", "certificates", "ca.crt"
        ).read_bytes(),
        pathlib.Path(
            "fedserver", "examples", ".cache", "certificates", "server.pem"
        ).read_bytes(),
        pathlib.Path(
            "fedserver", "examples", ".cache", "certificates", "server.key"
        ).read_bytes(),
    ),
    interceptors=[token_interceptor],
)
