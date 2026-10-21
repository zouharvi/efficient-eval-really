import random
import statistics
from collections.abc import Callable

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def weighted_sampling_with_priors(data: Data, budgets: Budgets, gamma: float = 0.95, coldstart=5, seed=42) -> ModelScoresAtBudget:
    """
    gamma [0, 1]: how much of the ranking is based on human evaluations only as opposed to metrics
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    data = list(data)
    models = list(data[0]["scores"].keys())
    n_models = len(models)
    n_items = len(data)

    scores_mat = np.array([[item["scores"][m] for m in models] for item in data], dtype=float)
    costs_arr = np.array([item["cost"] for item in data], dtype=float)

    metric_key = next(iter(data[0]["scores_metrics"][models[0]]))
    metrics_mat = np.array([[item["scores_metrics"][m][metric_key] for m in models] for item in data], dtype=float)
    metrics_mean = np.mean(metrics_mat, axis=0)
    # Ranks: 0 is highest metric score
    ranking_metrics = np.zeros(n_models, dtype=float)
    ranking_metrics[np.argsort(-metrics_mean)] = np.arange(n_models)

    model_scores = [[] for _ in range(n_models)]
    sums = np.zeros(n_models, dtype=float)
    counts = np.zeros(n_models, dtype=int)
    cost = 0.0

    k_cold = min(coldstart, n_items)
    for m in range(n_models):
        for i in range(k_cold):
            sc = scores_mat[i, m]
            model_scores[m].append(sc)
            sums[m] += sc
            counts[m] += 1
            cost += costs_arr[i]

    results_map = {}
    sorted_budgets = sorted(set(budgets))
    BATCH_SIZE = max(10, min(100, n_models // 2))

    for b in sorted_budgets:
        while cost < b:
            avail = np.where(counts < n_items)[0]
            if len(avail) == 0:
                break

            means = sums[avail] / counts[avail]
            # Empirical ranking among available models
            ranking_real = np.zeros(len(avail), dtype=float)
            ranking_real[np.argsort(-means)] = np.arange(len(avail))

            gamma_now = gamma ** (cost / b)
            model_est = (gamma_now * ranking_real) + ((1.0 - gamma_now) * ranking_metrics[avail])
            order = avail[np.argsort(model_est)]

            n_avail = len(order)
            weights = 1.0 / (np.arange(n_avail) + 1.0)
            probs = weights / np.sum(weights)

            batch_limit = min(BATCH_SIZE, int(b - cost) + 1)
            chosen_indices = rng.choice(n_avail, size=batch_limit, p=probs)
            added = 0

            for idx in chosen_indices:
                if cost >= b:
                    break
                m = order[idx]
                if counts[m] >= n_items:
                    continue
                item_idx = counts[m]
                sc = scores_mat[item_idx, m]
                model_scores[m].append(sc)
                sums[m] += sc
                counts[m] += 1
                cost += costs_arr[item_idx]
                added += 1

            if added == 0:
                break

        results_map[b] = {models[i]: list(model_scores[i]) for i in range(n_models)}

    return [results_map[b] for b in budgets]


def weighted_sampling_with_priors_corrected(data: Data, budgets: Budgets, coldstart: int = 5, batch_size: int = 10) -> ModelScoresAtBudget:
    """
    A variant of weighted_sampling_with_priors that corrects for the bias introduced by the sampling method.
    This method uses a linear regression model to predict the missing scores based on the available scores and metrics.
    """
    import sklearn.linear_model

    data = list(data)
    # cold start phase
    model_scores = {
        model: [x["scores"][model] for x in data[:coldstart]]
        for model in data[0]["scores"]
    }
    models = list(data[0]["scores"])
    cost = sum(x["cost"] for x in data[:coldstart]) * len(models)

    # one-hot encoding
    one_hot_item = {item_i: [0] * len(data) for item_i in range(len(data))}
    for i in range(len(data)):
        one_hot_item[i][i] = 1
    one_hot_model = {model: [0] * len(models) for model in models}
    for i, model in enumerate(models):
        one_hot_model[model][i] = 1

    output = []
    # active learning phase
    while budgets:
        if cost >= budgets[0] or not models:
            budgets = budgets[1:]
            output.append({model: list(model_scores[model]) for model in model_scores})
            continue

        estimator_model = sklearn.linear_model.LinearRegression()
        # TODO: we need to make this faster
        estimator_model.fit(
            [
                # encode item_i one-hot
                [x["scores_metrics"][model]["metric"]] + one_hot_item[item_i]
                for model in list(data[0]["scores"])
                for item_i, x in enumerate(data[:len(model_scores[model])])
            ],
            [
                x["scores"][model]
                for model in list(data[0]["scores"])
                for x in data[:len(model_scores[model])]
            ],
        )
        # predict missing values
        estimated_scores = {
            model: list(estimator_model.predict([
                [x["scores_metrics"][model]["metric"]] + one_hot_item[item_i]
                for item_i, x in enumerate(data)
            ])) # type: ignore
            for model in models
        }

        # skip recomputation sometimes
        for _ in range(batch_size):
            model_estimate = {
                # use predictions for the missing values
                model: statistics.mean(model_scores[model] + estimated_scores[model][len(model_scores[model]):])
                for model in models
            }
            models.sort(key=model_estimate.get, reverse=True)  # type: ignore

            model = random.choices(
                models,
                weights=[
                    1 / (rank + 1)
                    for model, rank in zip(models, range(len(models)))
                ],
                k=1,
            )[0]
            item = data[len(model_scores[model])]
            model_scores[model].append(item["scores"][model])
            cost += item["cost"]

            models = [
                model for model in model_scores if len(model_scores[model]) < len(data)
            ]
            if not models:
                break
    return output
