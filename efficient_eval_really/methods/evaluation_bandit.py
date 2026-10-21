import math

import evaluation_bandit.algorithms
import numpy as np

from efficient_eval_really.methods import (
    Budget,
    Budgets,
    Data,
    ModelScoresAtBudget,
    ModelScoresSubset,
)


def _fast_upper_confidence_bound(
    data: Data,
    budgets: Budgets,
    c: float = 100 * math.sqrt(2),
    coldstart: int = 5,
) -> ModelScoresAtBudget:
    models = list(data[0]["scores"].keys())
    n_models = len(models)
    n_items = len(data)

    scores_mat = np.array([[item["scores"][m] for m in models] for item in data], dtype=float)
    costs_arr = np.array([item["cost"] for item in data], dtype=float)

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

            total_counts = np.sum(counts)
            ln_total = math.log(max(1, total_counts))
            means = sums[avail] / counts[avail]
            exploration = c * np.sqrt(ln_total / counts[avail])
            ucb = means + exploration

            batch_limit = min(BATCH_SIZE, int(b - cost) + 1)
            order = avail[np.argsort(-ucb)]
            added = 0

            for m in order[:batch_limit]:
                if cost >= b or counts[m] >= n_items:
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


def _fast_weighted_sampling(
    data: Data,
    budgets: Budgets,
    coldstart: int = 5,
    seed: int = 42,
) -> ModelScoresAtBudget:
    rng = np.random.default_rng(seed)
    models = list(data[0]["scores"].keys())
    n_models = len(models)
    n_items = len(data)

    scores_mat = np.array([[item["scores"][m] for m in models] for item in data], dtype=float)
    costs_arr = np.array([item["cost"] for item in data], dtype=float)

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
            order = avail[np.argsort(-means)]
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


def evaluation_bandit_to_ours(data: Data, budget: Budget, method: str, **kwargs) -> ModelScoresSubset:
    return evaluation_bandit_to_ours_budgets(data=data, budgets=[budget], method=method, **kwargs)[0]


def evaluation_bandit_to_ours_budgets(data: Data, budgets: Budgets, method: str, **kwargs) -> ModelScoresAtBudget:
    if method == "upper_confidence_bound":
        return _fast_upper_confidence_bound(data=data, budgets=budgets, **kwargs)
    elif method == "weighted_sampling":
        return _fast_weighted_sampling(data=data, budgets=budgets, **kwargs)
    elif method == "uniform":
        return evaluation_bandit.algorithms.uniform(data=data, budgets=budgets)
    elif method == "uniform_nonsquare":
        return evaluation_bandit.algorithms.uniform_nonsquare(data=data, budgets=budgets)
    else:
        raise ValueError(f"Unknown method_name: {method}")
