import random
import statistics
from collections.abc import Callable

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def weighted_sampling_with_priors(data: Data, budgets: Budgets, gamma: float = 0.95, coldstart=5, seed=0, **kwargs) -> ModelScoresAtBudget:
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


def weighted_sampling_with_priors_corrected(
    data: Data,
    budgets: Budgets,
    coldstart: int = 5,
    batch_size: int = 10,
    seed: int = 0,
    **kwargs,
) -> ModelScoresAtBudget:
    """A variant of weighted_sampling_with_priors that corrects for the bias introduced by the sampling method.

    This method uses a linear regression model (one-way item fixed effects + metric) to predict missing scores.
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

    mask = np.zeros((n_items, n_models), dtype=bool)
    mask[:k_cold, :] = True

    results_map = {}
    sorted_budgets = sorted(set(budgets))
    effective_batch_size = max(batch_size, min(100, n_models // 2))

    for b in sorted_budgets:
        while cost < b:
            avail = np.where(counts < n_items)[0]
            if len(avail) == 0:
                break

            # Closed-form One-Way Fixed Effects OLS (within-item centering)
            item_counts = mask.sum(axis=1)
            eval_items = item_counts > 0

            y_eval = np.where(mask, scores_mat, 0.0)
            z_eval = np.where(mask, metrics_mat, 0.0)

            y_item_mean = np.zeros(n_items, dtype=float)
            z_item_mean = np.zeros(n_items, dtype=float)
            y_item_mean[eval_items] = y_eval[eval_items].sum(axis=1) / item_counts[eval_items]
            z_item_mean[eval_items] = z_eval[eval_items].sum(axis=1) / item_counts[eval_items]

            n_total_eval = np.sum(item_counts)
            overall_y = np.sum(y_eval) / max(1, n_total_eval)
            overall_z = np.sum(z_eval) / max(1, n_total_eval)
            y_item_mean[~eval_items] = overall_y
            z_item_mean[~eval_items] = overall_z

            y_tilde = np.where(mask, scores_mat - y_item_mean[:, None], 0.0)
            z_tilde = np.where(mask, metrics_mat - z_item_mean[:, None], 0.0)

            denom = np.sum(z_tilde**2)
            beta = (np.sum(z_tilde * y_tilde) / denom) if denom > 1e-12 else 0.0

            # Predict unobserved entries for each model
            pred_mat = y_item_mean[:, None] + beta * (metrics_mat - z_item_mean[:, None])
            unobs_pred = np.where(~mask, pred_mat, 0.0).sum(axis=0)
            model_est = (sums + unobs_pred) / n_items

            order = avail[np.argsort(-model_est[avail])]
            n_avail = len(order)
            weights = 1.0 / (np.arange(n_avail) + 1.0)
            probs = weights / np.sum(weights)

            batch_limit = min(effective_batch_size, int(b - cost) + 1)
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
                mask[item_idx, m] = True
                cost += costs_arr[item_idx]
                added += 1

            if added == 0:
                break

        results_map[b] = {models[i]: list(model_scores[i]) for i in range(n_models)}

    return [results_map[b] for b in budgets]
