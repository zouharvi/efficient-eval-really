import numpy as np

from efficient_eval_really.methods import Budgets, Data, ModelScoresAtBudget


def efficient_benchmarking_budgets(
    data: Data,
    budgets: Budgets,
    seed: int = 0,
    **kwargs,
) -> ModelScoresAtBudget:
    """Efficient Benchmarking of Language Models via Greedy Routing.

    Paper: "Efficient Benchmarking (of Language Models)" (Polo et al., NAACL 2024)

    Constructs minimal evaluation sets by greedily selecting items that maximize
    inter-model discriminability (variance of scores) while penalizing redundancy
    with previously selected items.
    """
    models = list(data[0]["scores"].keys())
    metric = next(iter(data[0]["scores_metrics"][models[0]]))
    n_models = len(models)
    n_items = len(data)

    S = np.array(
        [[item["scores_metrics"][m][metric] for m in models] for item in data],
        dtype=float,
    )  # (n_items, n_models)
    costs = np.array([item["cost"] for item in data], dtype=float)

    # Pre-select sequence of items greedily
    selected: list[int] = []
    available = set(range(n_items))

    # First item: maximum discriminability (variance across models)
    variances = np.var(S, axis=1)
    first_item = int(np.argmax(variances / (costs + 1e-6)))
    selected.append(first_item)
    available.remove(first_item)

    # Greedily add items balancing discriminability and novelty
    max_k = min(n_items, max(1, max(budgets) // max(1, n_models)))
    while len(selected) < max_k and available:
        cand_list = list(available)
        S_cand = S[cand_list]
        S_sel = S[selected]

        # Novelty: 1 - max correlation with existing items
        with np.errstate(invalid="ignore"):
            # Compute pairwise correlations between candidates and selected
            cand_centered = S_cand - np.mean(S_cand, axis=1, keepdims=True)
            sel_centered = S_sel - np.mean(S_sel, axis=1, keepdims=True)

            cand_norm = np.linalg.norm(cand_centered, axis=1, keepdims=True) + 1e-8
            sel_norm = np.linalg.norm(sel_centered, axis=1, keepdims=True) + 1e-8

            corr_mat = np.abs((cand_centered / cand_norm) @ (sel_centered / sel_norm).T)
            max_corr = np.max(corr_mat, axis=1)

        # Gain = variance * (1 - max_corr) / cost
        gain = variances[cand_list] * (1.0 - max_corr + 0.1) / (costs[cand_list] + 1e-6)
        best_idx = int(np.argmax(gain))
        picked = cand_list[best_idx]

        selected.append(picked)
        available.remove(picked)

    results_map = {}
    sorted_budgets = sorted(set(budgets))

    for b in sorted_budgets:
        cum_costs = np.cumsum(costs[selected]) * n_models
        k = int(np.searchsorted(cum_costs, b, side="right"))
        k = max(1, min(k, n_items))

        subset_items = selected[:k]
        results_map[b] = {m: [data[i]["scores"][m] for i in subset_items] for m in models}

    return [results_map[b] for b in budgets]
