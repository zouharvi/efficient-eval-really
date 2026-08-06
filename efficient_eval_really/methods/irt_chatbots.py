import numpy as np
from efficient_eval_really.methods import Data, Budgets, ModelScoresAtBudget


def _fit_2pl_irt(Y: np.ndarray) -> tuple:
    """Fit 2PL IRT via MLE on Y (n_models, n_items). Returns a, b, theta."""
    from scipy.optimize import minimize

    n_models, n_items = Y.shape
    eps = 1e-8

    mean_p = np.clip(Y.mean(axis=0), eps, 1 - eps)
    x0 = np.concatenate([
        np.ones(n_items),                  # a
        -np.log(mean_p / (1 - mean_p)),    # b
        np.zeros(n_models),                # theta
    ])

    def nll(x):
        a = x[:n_items]
        b = x[n_items:2 * n_items]
        theta = x[2 * n_items:]
        logits = a[None, :] * (theta[:, None] - b[None, :])
        p = np.clip(1 / (1 + np.exp(-np.clip(logits, -30, 30))), eps, 1 - eps)
        return -(Y * np.log(p) + (1 - Y) * np.log(1 - p)).sum()

    res = minimize(nll, x0, method="L-BFGS-B", options={"maxiter": 500})
    x = res.x
    a = x[:n_items]
    b = x[n_items:2 * n_items]
    theta = x[2 * n_items:]
    return a, b, theta


def irt_chatbots_budgets(data: Data, budgets: Budgets) -> ModelScoresAtBudget:
    """Approximate pairwise comparison with a single binary threshold c = mean(S).
      Fit general 2PL and then prompts selected by their fitted discrimination `a`.
    """
    models = list(data[0]["scores"].keys())
    metric = next(iter(data[0]["scores_metrics"][models[0]]))
    S = np.array([[item["scores_metrics"][m][metric] for m in models] for item in data])  # (n_items, n_models)
    n_items, n_models = S.shape

    c = np.mean(S)  # threshold: proxy for the paper's c
    Y = (S >= c).astype(float).T  # (n_models, n_items)
    a, b, theta = _fit_2pl_irt(Y)
    order = np.argsort(a)[::-1]  # descending discrimination

    results: ModelScoresAtBudget = []
    for budget in budgets:
        k = max(1, min(budget // n_models, n_items))
        if k >= n_items:
            results.append({m: [item["scores"][m] for item in data] for m in models})
            continue
        selected = order[:k].tolist()
        results.append({m: [data[i]["scores"][m] for i in selected] for m in models})
    return results
