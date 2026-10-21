import numpy as np
from sklearn.linear_model import Ridge

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def misses_the_mark_budgets(
    data: Data,
    budgets: Budgets,
    alpha: float = 10.0,
    seed: int = 0,
    **kwargs,
) -> ModelScoresAtBudget:
    """Augmented Inverse Propensity Weighting (AIPW) Benchmark Prediction.

    Algorithm from Zhang, Dorner & Hardt (2025):
    'How Benchmark Prediction from Fewer Data Misses the Mark'

    Selects an evaluation coreset, fits a Ridge regression model mapping proxy metric
    scores to target model evaluations, and applies a doubly-robust AIPW / PPI estimator
    to accurately estimate full-benchmark performance.
    """
    rng = np.random.default_rng(seed)
    models = list(data[0]["scores"].keys())
    metric = next(iter(data[0]["scores_metrics"][models[0]]))
    n_models = len(models)
    n_items = len(data)

    # S_metric: (n_items, n_models), S_true: (n_models, n_items)
    S_metric = np.array([[item["scores_metrics"][m][metric] for m in models] for item in data], dtype=float)
    S_true = np.array([[item["scores"][m] for item in data] for m in models], dtype=float)

    results: ModelScoresAtBudget = []
    for budget in budgets:
        coreset_size = max(2, min(budget // n_models, n_items))

        if coreset_size >= n_items:
            results.append({m: [float(item["scores"][m]) for item in data] for m in models})
            continue

        coreset_indices = rng.choice(n_items, size=coreset_size, replace=False)
        rest_mask = np.ones(n_items, dtype=bool)
        rest_mask[coreset_indices] = False
        rest_indices = np.where(rest_mask)[0]

        n = len(coreset_indices)
        N = len(rest_indices)

        x_train = S_metric[coreset_indices, :]  # (coreset_size, n_models)
        x_test = S_metric[rest_indices, :]  # (N, n_models)

        model_scores = {}
        for m_idx, m in enumerate(models):
            y_train = S_true[m_idx, coreset_indices]

            rgs = Ridge(alpha=alpha)
            rgs.fit(x_train, y_train.reshape(-1, 1))

            y_pred_train = rgs.predict(x_train).squeeze()
            y_pred_test = rgs.predict(x_test).squeeze()

            # AIPW estimator: ppi = mean(y_train) + (mean(y_pred_test) - mean(y_pred_train)) / (1 + n / N)
            ppi_part = float(np.mean(y_train) + (np.mean(y_pred_test) - np.mean(y_pred_train)) / (1.0 + n / max(1, N)))
            model_scores[m] = [ppi_part]

        results.append(model_scores)

    return results
