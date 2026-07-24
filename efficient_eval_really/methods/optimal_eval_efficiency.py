from __future__ import annotations
import math
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from efficient_eval_really.methods import Data, Budgets, ModelScoresAtBudget


def optimal_eval_efficiency_budgets(
    data: Data,
    budgets: Budgets,
    n_iter: int = 10000,
    T_start: float = 1.0,
    T_end: float = 1e-3,
    seed: int = 42,
) -> ModelScoresAtBudget:
    """
    Item subset selection via simulated annealing to minimise the L2 error
    between subset-mean and full-benchmark-mean model scores.
    """
    models = list(data[0]["scores"].keys())
    metric = next(iter(data[0]["scores_metrics"][models[0]]))
    S = np.array(
        [[item["scores_metrics"][m][metric] for m in models] for item in data],
        dtype=float,
    )  
    n_items, n_models = S.shape
    true_means = S.mean(axis=0)  # full benchmark mean per model

    rng = np.random.default_rng(seed)

    results: ModelScoresAtBudget = []
    for budget in budgets:
        k = max(1, min(budget // n_models, n_items))

        if k >= n_items:
            results.append({m: [item["scores"][m] for item in data] for m in models})
            continue

        # initialise with a random subset
        current = list(rng.choice(n_items, k, replace=False).tolist())
        not_in = list(set(range(n_items)) - set(current))
        current_sum = S[current].sum(axis=0)  

        def _obj(s: np.ndarray) -> float:
            return float(np.sum((s / k - true_means) ** 2))

        current_obj = _obj(current_sum)
        best, best_obj = list(current), current_obj

        # simulated annealing search
        for step in range(n_iter):
            T = T_start * (T_end / T_start) ** (step / n_iter)

            # swapping items 
            ri = int(rng.integers(k))
            ai = int(rng.integers(n_items - k))
            remove, add = current[ri], not_in[ai]

            # check if swap improves objective
            candidate_sum = current_sum - S[remove] + S[add]
            c_obj = _obj(candidate_sum)
            delta = c_obj - current_obj

            # accept if better, or with metropolis probability
            if delta < 0 or (T > 1e-10 and rng.random() < math.exp(-delta / T)):
                current[ri] = add
                not_in[ai] = remove
                current_sum = candidate_sum
                current_obj = c_obj
                if current_obj < best_obj:
                    best, best_obj = list(current), current_obj

        results.append({m: [data[i]["scores"][m] for i in best] for m in models})

    return results
