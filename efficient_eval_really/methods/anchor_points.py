import numpy as np
from efficient_eval_really.methods import Data, Budgets, ModelScoresAtBudget
from efficient_eval_really.methods.tailored_benchmarks import _kmedoids


def anchor_points_budgets(data: Data, budgets: Budgets) -> ModelScoresAtBudget:
    """Anchor Points (Vivek et al., 2024): K-Medoids on item correlation distance."""
    models = list(data[0]["scores"].keys())
    S = np.array([[item["scores"][m] for m in models] for item in data])  # (n_items, n_models)
    n_items, n_models = S.shape

    # 1 - Pearson correlation as distance; zero-variance items (nan) treated as uncorrelated
    with np.errstate(invalid="ignore"):
        D = 1 - np.nan_to_num(np.corrcoef(S), nan=0.0)
    np.fill_diagonal(D, 0.0)

    results: ModelScoresAtBudget = []
    for budget in budgets:
        k = max(1, min(budget // n_models, n_items))

        if k >= n_items:
            results.append({m: [item["scores"][m] for item in data] for m in models})
            continue

        medoids = _kmedoids(D, k)
        results.append({m: [data[i]["scores"][m] for i in medoids] for m in models})

    return results
