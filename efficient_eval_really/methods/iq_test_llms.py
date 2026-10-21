import numpy as np
from sklearn.decomposition import FactorAnalysis

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def iq_test_llms_budgets(
    data: Data,
    budgets: Budgets,
    n_factors: int = 3,
    seed: int = 42,
) -> ModelScoresAtBudget:
    """Core Skills Latent Factor Evaluation (IQ Test for LLMs).

    Algorithm from Maimon et al. (2025):
    'IQ Test for LLMs: An Evaluation Framework for Uncovering Core Skills in LLMs'

    Applies Exploratory Factor Analysis (EFA) to model performance proxy data to uncover
    latent skill dimensions, selecting a diverse coreset of items with highest loadings
    across discovered core factors.
    """
    models = list(data[0]["scores"].keys())
    metric = next(iter(data[0]["scores_metrics"][models[0]]))
    n_models = len(models)
    n_items = len(data)

    S = np.array([[item["scores_metrics"][m][metric] for m in models] for item in data], dtype=float)

    # Number of latent factors
    d = min(n_factors, n_models - 1 if n_models > 1 else 1, n_items)
    if d >= 1:
        fa = FactorAnalysis(n_components=d, random_state=seed)
        transformed = fa.fit_transform(S)  # (n_items, d)
    else:
        transformed = S

    factor_loadings = np.abs(transformed)
    max_k = max(min(b // n_models, n_items) for b in budgets)

    selected: list[int] = []
    used = set()
    num_f = factor_loadings.shape[1] if len(factor_loadings.shape) > 1 else 1

    for step in range(max_k):
        f_idx = step % num_f
        col = factor_loadings[:, f_idx] if len(factor_loadings.shape) > 1 else factor_loadings.ravel()
        order = np.argsort(col)[::-1]
        for item_idx in order:
            if item_idx not in used:
                selected.append(int(item_idx))
                used.add(int(item_idx))
                break
        if len(selected) >= max_k or len(used) >= n_items:
            break

    results: ModelScoresAtBudget = []
    for budget in budgets:
        k = max(1, min(budget // n_models, n_items))
        if k >= n_items:
            results.append({m: [float(item["scores"][m]) for item in data] for m in models})
            continue

        coreset = selected[:k]
        results.append({m: [float(data[idx]["scores"][m]) for idx in coreset] for m in models})

    return results
