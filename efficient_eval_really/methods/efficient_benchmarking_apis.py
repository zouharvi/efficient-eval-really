import numpy as np

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def efficient_benchmarking_apis_budgets(
    data: Data,
    budgets: Budgets,
    seed: int = 0,
    **kwargs,
) -> ModelScoresAtBudget:
    """Thompson Sampling Bandit for Efficient Benchmarking.

    Algorithm from (EACL 2017):
    'Efficient Benchmarking of NLP APIs using Multi-armed Bandits'

    Maintains a Gaussian posterior distribution over the expected evaluation performance
    of each candidate system/model. At each decision step, samples from the model posteriors
    (Thompson Sampling) to balance exploration and exploitation, dynamically allocating more
    evaluation budget to top-performing candidates.
    """
    rng = np.random.default_rng(seed)
    models = list(data[0]["scores"].keys())
    n_models = len(models)
    n_items = len(data)

    scores_mat = np.array([[item["scores"][m] for m in models] for item in data], dtype=float)
    costs_arr = np.array([item["cost"] for item in data], dtype=float)

    model_scores = [[] for _ in range(n_models)]
    item_indices = np.zeros(n_models, dtype=int)
    counts = np.zeros(n_models, dtype=int)
    sums = np.zeros(n_models, dtype=float)
    sum_sqs = np.zeros(n_models, dtype=float)
    cost = 0.0

    # Initial warm-up: 2 items per model
    warmup = min(2, n_items)
    for m in range(n_models):
        for i in range(warmup):
            sc = scores_mat[i, m]
            model_scores[m].append(sc)
            counts[m] += 1
            sums[m] += sc
            sum_sqs[m] += sc * sc
            cost += costs_arr[i]
        item_indices[m] = warmup

    results_map = {}
    sorted_budgets = sorted(set(budgets))
    BATCH_SIZE = max(10, min(100, n_models // 2))

    for budget in sorted_budgets:
        while cost < budget:
            avail = np.where(item_indices < n_items)[0]
            if len(avail) == 0:
                break

            c = counts[avail]
            means = sums[avail] / c
            var = np.maximum(1e-8, (sum_sqs[avail] - (sums[avail] ** 2) / c) / np.maximum(1, c - 1))
            stds = np.where(c > 1, np.sqrt(var), 0.2)
            sems = np.maximum(1e-4, stds / np.sqrt(c))

            batch_limit = min(BATCH_SIZE, int(budget - cost) + 1)
            samples = rng.normal(loc=means, scale=sems, size=(batch_limit, len(avail)))
            winners = avail[np.argmax(samples, axis=1)]
            added = 0

            for best_idx in winners:
                if cost >= budget or item_indices[best_idx] >= n_items:
                    continue
                item_idx = item_indices[best_idx]
                sc = scores_mat[item_idx, best_idx]
                model_scores[best_idx].append(sc)
                item_indices[best_idx] += 1
                counts[best_idx] += 1
                sums[best_idx] += sc
                sum_sqs[best_idx] += sc * sc
                cost += costs_arr[item_idx]
                added += 1

            if added == 0:
                break

        results_map[budget] = {models[i]: list(model_scores[i]) for i in range(n_models)}

    return [results_map[b] for b in budgets]
