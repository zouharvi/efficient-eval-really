import numpy as np
from sklearn.metrics import pairwise_distances

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def bias_bounded_budgets(data: Data, budgets: Budgets, seed: int = 0, **kwargs) -> ModelScoresAtBudget:
    """Bias-Bounded Submodular Subset Selection.

    Algorithm from (2025):
    'Efficient Benchmarking via Bias-Bounded Subset Selection'

    Formulates coreset selection for model evaluation as maximizing a submodular
    facility location / coverage objective on proxy response vectors. Uses a greedy
    algorithm with provable (1 - 1/e) approximation to identify representative evaluation
    subsets bounding ranking bias.
    """
    models = list(data[0]["scores"].keys())
    metric = next(iter(data[0]["scores_metrics"][models[0]]))
    n_models = len(models)
    n_items = len(data)

    S = np.array([[item["scores_metrics"][m][metric] for m in models] for item in data], dtype=float)
    D = pairwise_distances(S, metric="euclidean")
    max_d = float(np.max(D)) if len(D) > 0 and np.max(D) > 0 else 1.0
    Sim = 1.0 - (D / max_d)  # (n_items, n_items)

    max_k = max(min(b // n_models, n_items) for b in budgets)
    selected: list[int] = []
    cur_max_sim = np.zeros(n_items, dtype=float)
    remaining = set(range(n_items))

    # First item: item maximizing total similarity to the benchmark pool
    first_item = int(np.argmax(Sim.sum(axis=1)))
    selected.append(first_item)
    cur_max_sim = Sim[first_item].copy()
    remaining.remove(first_item)

    for _ in range(1, max_k):
        rem_arr = np.array(list(remaining), dtype=int)
        new_max = np.maximum(cur_max_sim[None, :], Sim[rem_arr, :])
        gains = (new_max - cur_max_sim[None, :]).sum(axis=1)

        best_cand = int(rem_arr[np.argmax(gains)])
        selected.append(best_cand)
        cur_max_sim = np.maximum(cur_max_sim, Sim[best_cand])
        remaining.remove(best_cand)

    results: ModelScoresAtBudget = []
    for budget in budgets:
        k = max(1, min(budget // n_models, n_items))
        if k >= n_items:
            results.append({m: [float(item["scores"][m]) for item in data] for m in models})
            continue

        coreset = selected[:k]
        results.append({m: [float(data[idx]["scores"][m]) for idx in coreset] for m in models})

    return results
