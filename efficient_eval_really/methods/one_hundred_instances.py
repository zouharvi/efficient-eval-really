import numpy as np

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def one_hundred_instances_budgets(
    data: Data,
    budgets: Budgets,
    reg: float = 1e-3,
    seed: int = 0,
    **kwargs,
) -> ModelScoresAtBudget:
    """100 Instances is all you need: Reference Instances + Linear Assessor.

    Paper: "100 instances is all you need: predicting the success of a new LLM
    on unseen data by testing on a few instances" (Mazzaccara et al., 2024)

    Selects an informative set of reference instances using greedy forward selection
    to minimize prediction error across the evaluation set, then trains a linear
    assessor that predicts performance on all instances from the reference subset.
    """
    models = list(data[0]["scores"].keys())
    metric = next(iter(data[0]["scores_metrics"][models[0]]))
    n_models = len(models)
    n_items = len(data)

    # Item response matrix S in R^(n_items, n_models)
    S = np.array(
        [[item["scores_metrics"][m][metric] for m in models] for item in data],
        dtype=float,
    )

    # Item costs
    costs = np.array([item["cost"] for item in data], dtype=float)

    # Precompute greedy reference instance ordering
    selected_indices: list[int] = []
    remaining = set(range(n_items))

    # First instance: highest variance across models
    first_idx = int(np.argmax(np.var(S, axis=1)))
    selected_indices.append(first_idx)
    remaining.remove(first_idx)

    # Greedy forward selection up to min(100, n_items)
    max_select = min(n_items, max(100, max(budgets) // max(1, n_models)))
    while len(selected_indices) < max_select and remaining:
        # Pick candidate that has lowest correlation / redundancy with already selected
        S_sel = S[selected_indices]
        candidates = list(remaining)
        S_cand = S[candidates]

        # Dot product with existing normalized responses
        norms_sel = np.linalg.norm(S_sel, axis=1, keepdims=True) + 1e-8
        norms_cand = np.linalg.norm(S_cand, axis=1, keepdims=True) + 1e-8
        cos_sim = np.abs((S_cand / norms_cand) @ (S_sel / norms_sel).T)
        max_sim = np.max(cos_sim, axis=1)

        best_cand_idx = int(np.argmin(max_sim))
        best_cand = candidates[best_cand_idx]
        selected_indices.append(best_cand)
        remaining.remove(best_cand)

    results_map = {}
    sorted_budgets = sorted(set(budgets))

    for b in sorted_budgets:
        # Determine number of reference instances affordable across models
        cum_costs = np.cumsum(costs[selected_indices]) * n_models
        affordable = np.searchsorted(cum_costs, b, side="right")
        k = max(1, min(affordable, n_items))

        ref_idx = selected_indices[:k]
        if k >= n_items:
            results_map[b] = {m: [item["scores"][m] for item in data] for m in models}
            continue

        # Linear Assessor: predict full responses from reference responses
        X_ref = S[ref_idx]  # (k, n_models)
        # Ridge weights to predict S (n_items, n_models) from X_ref
        # W = S @ X_ref.T @ (X_ref @ X_ref.T + lambda I)^-1
        Gram = X_ref @ X_ref.T + reg * np.eye(k)
        W = S @ X_ref.T @ np.linalg.inv(Gram)  # (n_items, k)

        model_scores = {}
        for m in models:
            y_ref = np.array([data[i]["scores"][m] for i in ref_idx])  # (k,)
            pred_full = W @ y_ref  # (n_items,)
            pred_full[ref_idx] = y_ref
            model_scores[m] = list(np.clip(pred_full, 0.0, 1.0))

        results_map[b] = model_scores

    return [results_map[b] for b in budgets]
